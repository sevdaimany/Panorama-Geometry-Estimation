import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch_compat_patch # remove it if not using romav2
import torch
import cv2
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from einops import rearrange
import kornia as K
from correspondence_extractor import CorrespondenceExtractor

import DAP.test.infer as DAP_infer 
from DAP.test.infer import load_model as DAP_load_model

import yaml
import torchvision.utils as vutils
import geometry as geometry
import gc
import torch.nn as nn
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import AlphaCompositor, PerspectiveCameras, PointsRasterizationSettings, PointsRasterizer, PointsRenderer
import numpy as np
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import torch.nn.functional as F

class DFRMPoseEstimator:
    """
    Encapsulates the Feature Registration Module (DFRM) pipeline, including
    depth estimation, correspondence extraction, and batch preparation.
    """
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        
        print(f"[INFO] Initializing DFRMPoseEstimator on {self.device}...")
        
        self.feature_warper = DifferentiableFeatureWarper()
        self.depth_predictor = self._load_dap_depth_model()
        self.correspondence_extractor = CorrespondenceExtractor(
            matching_model=cfg.matching_model, 
            use_magsac=cfg.use_magsac
        )

    def _load_dap_depth_model(self):
        config_path = self.cfg.dap_config_path
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config["load_weights_dir"] = self.cfg.dap_load_weights_dir
        model, _ = DAP_load_model(config)
        print("[INFO] DAP Depth Config loaded.")
        return model

    @staticmethod
    def _read_image_as_pilrgb(path_to_image):
        assert path_to_image is not None
        with open(path_to_image, "rb") as file:
            return Image.open(file).convert("RGB")

    @staticmethod
    def _normalise_image(img_as_tensor):
        imagenet_normalisation = K.enhance.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = rearrange(img_as_tensor, "c h w -> 1 c h w")
        img = imagenet_normalisation(img)
        return img.squeeze()

    @staticmethod
    def _read_image_as_tensor(pil_image):
        return pil_to_tensor(pil_image).float()
    
    def preprocess_single_image(self, image_path):
        """Phase 1: Load image, resize, crop, and predict depth (Runs ONCE per image)."""

        img_pil = self._read_image_as_pilrgb(image_path)

        target_size = tuple(self.cfg.input_size)
        if img_pil.size != target_size:
            img_pil = img_pil.resize(self.cfg.input_size, resample=Image.Resampling.LANCZOS)
        
        img_tensor = self._read_image_as_tensor(img_pil)

        # Zero out the bottom 23%
        h, w = img_tensor.shape[-2:]
        img_tensor[:, int(h*0.77):, :] = 0
        
        # First Normalization
        img_tensor = img_tensor / 255.0
        
        return {"image": img_tensor, 
                 "path": image_path}

    def prepare_pair_from_cache(self, data_A, data_B):
        """Phase 2: Extract correspondences and finalize batch using cached data (Runs per PAIR)."""
        item = {
            "image1_path": data_A["path"],
            "image2_path": data_B["path"],
            "image1": data_A["image"].clone(),
            "image2": data_B["image"].clone(),
            "depth1": data_A["depth"].clone() if data_A["depth"] is not None else None,
            "depth2": data_B["depth"].clone() if data_B["depth"] is not None else None,
            "registration_strategy": "3d",
        }
        
        for key in ["intrinsics1", "intrinsics2", "rotation1", "rotation2", "position1", "position2", "transfm2d_1_to_2", "transfm2d_2_to_1", "focale1", "focale2"]:
            item[key] = None
            
        # Correspondence Extraction
        dummy_batch = {k: [v] for k, v in item.items()}
        dummy_batch = self.correspondence_extractor(dummy_batch)
        for k, v in dummy_batch.items():
            item[k] = v[0]
            
      
        item["image1"] = self._normalise_image(item["image1"])
        item["image2"] = self._normalise_image(item["image2"])

        # Package into Batch
        batch = {}
        for k, v in item.items():
            if torch.is_tensor(v):
                batch[k] = v.unsqueeze(0).to(self.device)
            elif isinstance(v, (int, float)):
                batch[k] = torch.tensor([v], dtype=torch.float32).to(self.device)
            else:
                batch[k] = [v]
        return batch

    def prepare_batch(self, image_a_path, image_b_path, sequence_id):
        """Prepares a pair of images for DFRM extraction."""
        item = {
            "image1_path": image_a_path,
            "image2_path": image_b_path,
            "registration_strategy": "3d", 
        }

        # 1. Load Images
        img1_pil = self._read_image_as_pilrgb(image_a_path)
        img2_pil = self._read_image_as_pilrgb(image_b_path)

        # resize if doesnt match cfg.input_size
        target_size = tuple(self.cfg.input_size)
        if img1_pil.size != target_size:
            print(f"[WARNING] Resizing image1 from {img1_pil.size} to {self.cfg.input_size}.")
            img1_pil = img1_pil.resize(self.cfg.input_size, resample=Image.Resampling.LANCZOS)
        if img2_pil.size != target_size:
            print(f"[WARNING] Resizing image2 from {img2_pil.size} to {self.cfg.input_size}.")
            img2_pil = img2_pil.resize(self.cfg.input_size, resample=Image.Resampling.LANCZOS)
        
        item['image1'] = self._read_image_as_tensor(img1_pil)
        item['image2'] = self._read_image_as_tensor(img2_pil)

        # 3. Depth Estimation
        print("[INFO] Predicting Depth...")
        if self.cfg.depth_model != "unidepth":
            item["depth1"] = torch.from_numpy(DAP_infer.infer_raw(self.depth_predictor, self.device, item["image1"].permute(1, 2, 0).numpy()))
            item["depth2"] = torch.from_numpy(DAP_infer.infer_raw(self.depth_predictor, self.device, item["image2"].permute(1, 2, 0).numpy()))
        
        # 2. Zero out the bottom 23% (Crucial for panoramas to hide tripod/car)
        h1, w1 = item["image1"].shape[-2:]
        h2, w2 = item["image2"].shape[-2:]

        item["image1"][:, int(h1*0.77):, :] = 0
        item["image2"][:, int(h2*0.77):, :] = 0

        # 4. First Normalization
        item['image1'] = item['image1'] / 255.0
        item['image2'] = item['image2'] / 255.0

        for key in ["intrinsics1", "intrinsics2", "rotation1", "rotation2", "position1", "position2", "transfm2d_1_to_2", "transfm2d_2_to_1", "focale1", "focale2"]:
            item[key] = None
        
        # 5. Correspondence Extraction
        print("[INFO] Extracting Correspondences...")
        dummy_batch = {k: [v] for k, v in item.items()}
        dummy_batch = self.correspondence_extractor(dummy_batch)
        for k, v in dummy_batch.items():
            item[k] = v[0]

        
        # save correspondences visualization
        if self.cfg.save_correspondences:
            os.makedirs(os.path.join(self.cfg.output_dir_ext, 'correspondences', sequence_id), exist_ok=True)
            # num_to_plot = min(400, len(item["points1"]))
            num_to_plot = len(item["points1"])
            
            best_pts1 = item["points1"][:num_to_plot].cpu()
            best_pts2 = item["points2"][:num_to_plot].cpu()
            
            save_path = os.path.join(self.cfg.output_dir_ext, "correspondences", sequence_id, f"{os.path.basename(image_a_path).split('.')[0]}_{os.path.basename(image_b_path).split('.')[0]}.png")
            self.plot_correspondences( item["image1"].cpu(), item["image2"].cpu(), best_pts1, best_pts2, 
                save_path=save_path
            ) 
            print(f"  Correspondences visualization saved → {save_path}\n")

        item["image1"] = self._normalise_image(item["image1"])
        item["image2"] = self._normalise_image(item["image2"])

        # 7. Package into Batch
        print("[INFO] Computing RT Matrix...")
        batch = {}
        for k, v in item.items():
            if torch.is_tensor(v):
                batch[k] = v.unsqueeze(0).to(self.device)
            elif isinstance(v, (int, float)):
                batch[k] = torch.tensor([v], dtype=torch.float32).to(self.device)
            else:
                batch[k] = [v]
        return batch
    
    def plot_correspondences(self, source_image, target_image, source_points, target_points, save_path="./correspondences.png"):
        """
        Helper function to plot correspondences.
        """
        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        if source_image.min() < 0 or source_image.max() > 1:
            source_image = (source_image * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
        if target_image.min() < 0 or target_image.max() > 1:
            target_image = (target_image * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)

        if torch.is_tensor(source_image):
            source_image = K.tensor_to_image(source_image)
            
        if torch.is_tensor(target_image):
            target_image = K.tensor_to_image(target_image)

        fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
        axarr[0].imshow(source_image)
        axarr[0].axis('off')
        axarr[1].imshow(target_image)
        axarr[1].axis('off')
        
        source_points = source_points * torch.tensor([source_image.shape[1], source_image.shape[0]])
        target_points = target_points * torch.tensor([target_image.shape[1], target_image.shape[0]])

        for i, (pt_q, pt_t) in enumerate(zip(source_points, target_points)):
            # Cast PyTorch tensors to standard Python floats to prevent Matplotlib crashes
            x1, y1 = float(pt_q[0]), float(pt_q[1])
            x2, y2 = float(pt_t[0]), float(pt_t[1])
            
            col = (np.random.random(), np.random.random(), np.random.random())
            
            con = ConnectionPatch(
                xyA=(x2, y2), xyB=(x1, y1),
                coordsA='data', coordsB='data',
                axesA=axarr[1], axesB=axarr[0],
                color=col, linewidth=0.7 
            )
            axarr[1].add_artist(con)
            
            # Draw the markers
            axarr[0].plot(x1, y1, c=col, marker='x')
            axarr[1].plot(x2, y2, c=col, marker='x')

        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig(save_path, bbox_inches="tight", dpi=300, pad_inches=0)
        plt.close(fig)
    
    def estimate_Rt_using_points_panorama(self, points1, points2, depth1, depth2):
        """
        Estimates the Rigid Transformation (Rt) between two panoramic images,
        utilizing the existing geometry.equirect_to_3d function.
        """
        b = len(points1)
        device_type = points1[0]
        
        # Identity matrices for K (panoramas have no pinhole focal length)
        K1_inv = torch.eye(3).unsqueeze(0).repeat(b, 1, 1).type_as(device_type)
        K2_inv = torch.eye(3).unsqueeze(0).repeat(b, 1, 1).type_as(device_type)
        
        batch_points1_in_world = []
        batch_points2_in_world = []
        
        for i in range(b):
            if depth1.dim() == 4:
                H, W = depth1[i, 0].shape
            elif depth1.dim() == 3:
                H, W = depth1[i].shape
            else:
                H, W = depth1.shape[-2:]
                
            pts1 = points1[i] # [N, 2] in absolute pixels
            pts2 = points2[i] # [N, 2] in absolute pixels
            
            # Sample Depth
            d1 = geometry.sample_depth_for_given_points(depth1[i].unsqueeze(0), pts1.unsqueeze(0)).view(-1)
            d2 = geometry.sample_depth_for_given_points(depth2[i].unsqueeze(0), pts2.unsqueeze(0)).view(-1)
            
            # Lift to 3D using YOUR function
            # Returns [N, 4] homogeneous coordinates [X, Y, Z, 1]
            pts1_3d_homo = geometry.equirect_to_3d(pts1, d1, W, H)
            pts2_3d_homo = geometry.equirect_to_3d(pts2, d2, W, H)
            
            # Drop the homogeneous '1' to get [N, 3] [X, Y, Z]
            pts1_3d = pts1_3d_homo[..., :3]
            pts2_3d = pts2_3d_homo[..., :3]
            
            batch_points1_in_world.append(pts1_3d)
            batch_points2_in_world.append(pts2_3d)

        #  Estimate Rigid Transformation (Rotation & Translation)
        Rt_1_to_2 = geometry.estimate_linear_warp(batch_points1_in_world, batch_points2_in_world)
        Rt_2_to_1 = geometry.estimate_linear_warp(batch_points2_in_world, batch_points1_in_world)
        
        return K1_inv, K2_inv, Rt_1_to_2, Rt_2_to_1


    def estimate_trajectory(self, image_paths, sequence_id, return_batch_for_one_pair=False):
        """
        Computes DFRM RT sequentially across a list of N raw images (0->1, 1->2, ...).
        Returns a list of accumulated RT matrices relative to the first image.
        """
        gc.collect()
        torch.cuda.empty_cache()
        
        accumulated_rts = [torch.eye(4, dtype=torch.float32).unsqueeze(0).to('cpu')]
        current_global_rt = accumulated_rts[0].to(self.device)

        for i in range(len(image_paths) - 1):
            print(f"[INFO] Processing pair: {i} -> {i+1}")

            with torch.no_grad(), torch.inference_mode():
                batch = self.prepare_batch(image_paths[i], image_paths[i+1], sequence_id)
                
                K_inv_1, K_inv_2, Rt_1_to_2, Rt_2_to_1 = self.estimate_Rt_using_points_panorama(
                    batch["points1"], batch["points2"], batch["depth1"], batch["depth2"]
                )
                
                if Rt_2_to_1 is not None:
                    current_global_rt = current_global_rt @ Rt_2_to_1
                    accumulated_rts.append(current_global_rt.detach().cpu())
                else:
                    print(f"!! [WARNING] Failed to estimate RT for pair {i}->{i+1}")
                    accumulated_rts.append(None)
            
            if return_batch_for_one_pair:
                return accumulated_rts, batch

            del batch, K_inv_1, K_inv_2, Rt_1_to_2, Rt_2_to_1
            gc.collect()
            torch.cuda.empty_cache()
                
        return accumulated_rts
    

    def generate_occlusion_mask(self, panorama_A_path, panorama_B_path, sequence_id, save_dir, Rt_1_to_2_tensor, Rt_2_to_1_tensor, model_name="", batch=None):
        """Uses the Feature Warper to generate occlusion masks."""
        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3,1,1)
        IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3,1,1)

        def denorm(img):
            return (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0,1)

        panorama_A = cv2.imread(panorama_A_path)
        h, w, _ = panorama_A.shape

        if batch is None:
            batch = self.prepare_batch(panorama_A_path, panorama_B_path)

        visibility = torch.ones((1, 1, h, w), requires_grad=False).type_as(batch['image1'])
        image1_with_visibility = torch.cat([batch['image1'], visibility], dim=1)
        image2_with_visibility = torch.cat([batch['image2'], visibility], dim=1)

        K1_inv = torch.eye(3).unsqueeze(0).repeat(1, 1, 1).type_as(batch['image1'])
        K2_inv = torch.eye(3).unsqueeze(0).repeat(1, 1, 1).type_as(batch['image2'])

        if Rt_1_to_2_tensor.dim() == 2: Rt_1_to_2_tensor = Rt_1_to_2_tensor.unsqueeze(0)
        if Rt_2_to_1_tensor.dim() == 2: Rt_2_to_1_tensor = Rt_2_to_1_tensor.unsqueeze(0)

        Rt_1_to_2_tensor = Rt_1_to_2_tensor.to(self.device, dtype=batch['image1'].dtype)
        Rt_2_to_1_tensor = Rt_2_to_1_tensor.to(self.device, dtype=batch['image2'].dtype)

        image1_warped = self.feature_warper.warp(image1_with_visibility, batch['depth1'], K1_inv, K2_inv, Rt_1_to_2_tensor)
        image2_warped = self.feature_warper.warp(image2_with_visibility, batch['depth2'], K2_inv, K1_inv, Rt_2_to_1_tensor)

        image1_warped_rgb = image1_warped[:, :3, :, :]
        visibility_mask_1 = image1_warped[:, -1:, :, :]

        image2_warped_rgb = image2_warped[:, :3, :, :]
        visibility_mask_2 = image2_warped[:, -1:, :, :]

        os.makedirs(save_dir, exist_ok=True)
        i = 0 
        side_by_side_1 = vutils.make_grid([denorm(batch['image2'][i]), denorm(batch['image1'][i]), denorm(image1_warped_rgb[i])], nrow=3, padding=5, pad_value=0.8)  
        side_by_side_2 = vutils.make_grid([denorm(batch['image1'][i]), denorm(batch['image2'][i]), denorm(image2_warped_rgb[i])], nrow=3, padding=5, pad_value=0.8)

        name = os.path.basename(panorama_A_path).split('.')[0]
        vutils.save_image(side_by_side_1, os.path.join(save_dir, f"{name}{model_name}_side_by_side_1.png"))
        vutils.save_image(side_by_side_2, os.path.join(save_dir, f"{name}{model_name}_side_by_side_2.png"))
        vutils.save_image(visibility_mask_1[i], os.path.join(save_dir, f"{name}{model_name}_mask_1.png"))
        vutils.save_image(visibility_mask_2[i], os.path.join(save_dir, f"{name}{model_name}_mask_2.png"))
        print(f"\n[SUCCESS] Saved masks to '{save_dir}'")

    def generate_occlusion_mask_tensors(self, batch, Rt_1_to_2_tensor, Rt_2_to_1_tensor):
        """Warps images to generate occlusion masks and returns tensors to main loop."""
        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3,1,1)
        IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3,1,1)

        def denorm(img):
            return (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0,1)    

        h, w = batch['image1'].shape[-2:]

        if Rt_1_to_2_tensor.dim() == 2: Rt_1_to_2_tensor = Rt_1_to_2_tensor.unsqueeze(0)
        if Rt_2_to_1_tensor.dim() == 2: Rt_2_to_1_tensor = Rt_2_to_1_tensor.unsqueeze(0)

        Rt_1_to_2_tensor = Rt_1_to_2_tensor.to(self.device, dtype=batch['image1'].dtype)
        Rt_2_to_1_tensor = Rt_2_to_1_tensor.to(self.device, dtype=batch['image2'].dtype)

        depth1_safe = batch['depth1']
        depth2_safe = batch['depth2'] 

        visibility = torch.ones((1, 1, h, w), requires_grad=False).type_as(batch['image1'])
        
        image1_with_visibility = torch.cat([batch['image1'], visibility], dim=1)
        image2_with_visibility = torch.cat([batch['image2'], visibility], dim=1)

        K1_inv = torch.eye(3).unsqueeze(0).repeat(1, 1, 1).type_as(batch['image1'])
        K2_inv = torch.eye(3).unsqueeze(0).repeat(1, 1, 1).type_as(batch['image2'])

        image1_warped = self.feature_warper.warp(image1_with_visibility, depth1_safe, K1_inv, K2_inv, Rt_1_to_2_tensor)
        image2_warped = self.feature_warper.warp(image2_with_visibility, depth2_safe, K2_inv, K1_inv, Rt_2_to_1_tensor)
      
        image1_warped_rgb = image1_warped[:, :3, :, :]
        visibility_mask_1 = image1_warped[:, -1:, :, :]

        image2_warped_rgb = image2_warped[:, :3, :, :]
        visibility_mask_2 = image2_warped[:, -1:, :, :]

        i = 0 
        side_by_side_1 = vutils.make_grid([denorm(batch['image2'][i]), denorm(batch['image1'][i]), denorm(image1_warped_rgb[i])], nrow=3, padding=5, pad_value=0.8)  
        side_by_side_2 = vutils.make_grid([denorm(batch['image1'][i]), denorm(batch['image2'][i]), denorm(image2_warped_rgb[i])], nrow=3, padding=5, pad_value=0.8)

        return {
            "side_by_side_1": side_by_side_1,
            "side_by_side_2": side_by_side_2,
            "mask_1": visibility_mask_1[i],
            "mask_2": visibility_mask_2[i]
        }

      
class DifferentiableFeatureWarper(nn.Module):
    def __init__(self):
        super().__init__()

    def render(self, point_cloud, device, image_hw):
        # unchanged
        raster_settings = PointsRasterizationSettings(
            image_size=image_hw,
            radius=float(1.5) / min(image_hw) * 2.0,
            bin_size=0,
            points_per_pixel=8,
        )
        canonical_cameras = PerspectiveCameras(
            R=rearrange(torch.eye(3), "r c -> 1 r c"),
            T=rearrange(torch.zeros(3), "n -> 1 n"),
        )
        canonical_rasterizer = PointsRasterizer(cameras=canonical_cameras, raster_settings=raster_settings)
        canonical_renderer = PointsRenderer(rasterizer=canonical_rasterizer, compositor=AlphaCompositor())
        canonical_renderer.to(device)
        rendered_features = rearrange(canonical_renderer(point_cloud, eps=1e-5), "b h w c -> b c h w")
        return rendered_features

    def warp(self, features_src, depth_src, src_camera_K_inv, dst_camera_K_inv, Rt_src_to_dst):
        b, _, h, w = features_src.shape
        image_coords = rearrange(
            geometry.get_index_grid(h, w, batch=b, type_as=features_src),
            "b h w t -> b (h w) t",
        )

        # Step 1: lift to 3D using spherical back-projection
        # depth_src is [B, H, W] -> [B, H*W]
        if depth_src.dim() == 4:
            depth_flat = rearrange(depth_src, "b 1 h w -> b (h w)")
        else:
            depth_flat = rearrange(depth_src, "b h w -> b (h w)")
        # Use original W,H of panorama (features may be downsampled)
        points_3d = geometry.equirect_to_3d(image_coords, depth_flat, W=w, H=h)
        # [B, N, 4] homogeneous
        # Step 2: apply the 3D rigid transform Rt_src_to_dst
        # points_3d: [B, N, 4], Rt: [B, 4, 4]
        points_3d_warped = torch.einsum(
            "bij,bnj->bni", Rt_src_to_dst, points_3d
        )  # [B, N, 4]
        # Step 3: project warped points back to equirectangular coords
        src_points_in_dst = geometry.project_3d_to_equirect(
            points_3d_warped, W=w, H=h
        )  # [B, N, 3] -> (u_norm, v_norm, depth)

        return self.render_features_from_points(src_points_in_dst, features_src)

    def render_features_from_points(self, points_in_3d, features):
        # unchanged
        b, _, h, w = features.shape
        src_point_cloud = Pointclouds(
            # points=geometry.convert_to_pytorch3d_coordinate_system(points_in_3d),
            # FIX: Pass the specific height and width to prevent the gray border squash
            points=geometry.convert_to_pytorch3d_coordinate_system(points_in_3d, image_hw=(h, w)),
            features=rearrange(features, "b c h w -> b (h w) c"),
        )
        # 1. Get the raw render (with the tears at the poles)
        rendered = self.render(src_point_cloud, features.device, (h, w))
        return rendered