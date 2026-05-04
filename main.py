import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from easydict import EasyDict
import torch
import cv2
import yaml
import copy
import gc
from DFRMCameraPoseEstimator import DFRMPoseEstimator
from DeepCameraPoseEstimator import CameraPoseEstimator
from DepthEstimation import DepthEstimation
from PIL import Image
import torchvision.utils as vutils
import itertools
import matplotlib.pyplot as plt
import random
import numpy as np



def seed_everything(seed=42):
    """Locks all sources of randomness for reproducible results."""
    # 1. Python built-in randomness
    random.seed(seed)
    
    # 2. NumPy randomness (Used heavily in data prep)
    np.random.seed(seed)
    
    # 3. OpenCV randomness (CRITICAL for MAGSAC/RANSAC)
    cv2.setRNGSeed(seed)
    
    # 4. PyTorch randomness
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU
    
    # 5. Lock down CuDNN (Makes GPU operations deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_numpy_matrices(tensor_list):
        matrices = []
        for t in tensor_list:
            if t is None:
                matrices.append(None)
            else:
                # DFRM might return batch tensors (1, 4, 4), while Extrinsic might return (4, 4)
                mat = t[0].cpu().numpy() if t.dim() == 3 else t.cpu().numpy()
                matrices.append(mat)
        return matrices

def get_easy_dict_from_yaml_file(path_to_yaml_file):
    with open(path_to_yaml_file, "r") as stream:
        return EasyDict(yaml.safe_load(stream))

def load_panoramas(panorama_paths, target_size_wh, save_dir=None):
    panoramas = []
    for path in panorama_paths:
        pano = Image.open(path).convert('RGB')
        if pano.size != target_size_wh:
            print(f"[WARNING] Resizing panorama from {pano.size[0]}x{pano.size[1]} to {target_size_wh[0]}x{target_size_wh[1]}.")
            pano = pano.resize(target_size_wh, Image.Resampling.LANCZOS)
        panoramas.append(np.array(pano))

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, os.path.basename(path))
            pano.save(save_path)
            print(f"Saved resized panorama to {save_path}")
    return panoramas

def compare_RT_computed(image_paths, sequence_id, cfg, estimator_da3, estimator_mapanything, estimator_dfrm):
    """
    Compares the trajectory (RTs) computed sequentially by DFRM and the CameraPoseEstimators 
    (using MapAnything or DepthAnything3) across N images.
    """
    num_images = len(image_paths)
    print(f"\n{'='*50}\nComparing Trajectories for {num_images} images...\n{'='*50}")

    output_dir = cfg['output_dir_ext']
    cubemap_dir = os.path.join(output_dir, 'trajectory', sequence_id)
    os.makedirs(cubemap_dir, exist_ok=True)
    
    # Extract names for saving and plotting
    image_names = [os.path.basename(p).split('.')[0] for p in image_paths]

    # 1. Compute DFRM RT (Sequential)
    torch.set_float32_matmul_precision('highest')
    print(f"\n{'='*50}\n[INFO] Running DFRM Estimator...\n{'='*50}")
    accumulated_rts_dfrm = estimator_dfrm.estimate_trajectory(image_paths, sequence_id)

    # 2. Load Panoramas for Extrinsic Estimation
    panoramas = load_panoramas(image_paths, target_size_wh=tuple(cfg.input_size), save_dir=cubemap_dir)

    # 3. Compute Extrinsic Estimation RT (using MapAnything and DepthAnything3)
    print(f"\n{'='*50}\n[INFO] Running MapAnything Estimator...\n{'='*50}")
    rts_mapanything = estimator_mapanything.run(
        panoramas, sequence_id, save_results=False, output_dir=output_dir, image_names=image_names
    )    
    print(f"\n{'='*50}\n[INFO] Running DepthAnything3 Estimator...\n{'='*50}")
    rts_da3 = estimator_da3.run(panoramas, sequence_id)


    dfrm_matrices = to_numpy_matrices(accumulated_rts_dfrm)
    ma_matrices = to_numpy_matrices(rts_mapanything)
    da3_matrices = to_numpy_matrices(rts_da3)


    print(f"\n{'='*20} RESULTS {'='*20}")
    
    for mat_list, model_label in zip(
        [dfrm_matrices, ma_matrices, da3_matrices], 
        ['DFRM', 'MapAnything', 'DepthAnything3']
    ):
        print(f"\n--- {model_label} Estimated Trajectory ---")
        for i, (mat, name) in enumerate(zip(mat_list, image_names)):
            if mat is not None and i > 0:
                estimator_mapanything.print_human_readable_pose(mat, name=f"{image_names[0]} ➔ {name} ({model_label})")
        
        unique_image_name = "_".join(image_names) + model_label
        estimator_mapanything.verify_poses_visually(mat_list, image_names, sequence_id=sequence_id, image_name=unique_image_name)

def compare_occlusion_masks(image_a_path, image_b_path, sequence_id, save_dir, cfg, device, estimator_da3, estimator_mapanything, estimator_dfrm):
    print(f"\n{'='*50}\nComparing Occlusion Masks...\n{'='*50}")

    torch.set_float32_matmul_precision('highest')
    dfrm_trajectory, batch = estimator_dfrm.estimate_trajectory([image_a_path, image_b_path], sequence_id, return_batch_for_one_pair=True)
    Rt_1_to_2 = torch.inverse(dfrm_trajectory[1])
    Rt_2_to_1 = dfrm_trajectory[1] 
    estimator_dfrm.generate_occlusion_mask(image_a_path, image_b_path, sequence_id, save_dir, Rt_1_to_2, Rt_2_to_1, model_name="_dfrm", batch=batch)
    

    pano_A, pano_B = load_panoramas([image_a_path, image_b_path], target_size_wh=tuple(cfg.input_size))

    ma_traj = estimator_mapanything.run([pano_A, pano_B], sequence_id)
    estimator_dfrm.generate_occlusion_mask(image_a_path, image_b_path, sequence_id, save_dir, torch.inverse(ma_traj[1]), ma_traj[1], model_name="_mapanything", batch=batch)

    da3_traj = estimator_da3.run([pano_A, pano_B], sequence_id)
    estimator_dfrm.generate_occlusion_mask(image_a_path, image_b_path, sequence_id, save_dir, torch.inverse(da3_traj[1]), da3_traj[1], model_name="_depthanything3", batch=batch)


def run_full_pipeline(image_names_list, sequence_ids, cfg, estimators, device):
    plot_depth_maps = True
    plot_correspondences = True
    use_map_depth = True

    for seq_id, image_names in zip(sequence_ids, image_names_list):
        print(f"\n{'#'*80}\nProcessing Sequence: {seq_id}\n{'#'*80}")
        
        image_names_for_print = [p.split('.')[0] for p in image_names]

        image_paths = [os.path.join(cfg['data_path_ext'], seq_id, name) for name in image_names]
        
        folder_name = f"pipeline_visualization_{'mapdepth' if use_map_depth else 'dapdepth'}"
        vis_dir = os.path.join(cfg['output_dir_ext'], folder_name, seq_id)
        occlusion_dir = os.path.join(vis_dir, 'occlusion_masks')
        poses_dir = os.path.join(vis_dir, 'trajectory')
        correspondences_dir = os.path.join(vis_dir, 'correspondences')
        depth_dir = os.path.join(vis_dir, 'depth')

        os.makedirs(correspondences_dir, exist_ok=True)
        os.makedirs(occlusion_dir, exist_ok=True)
        os.makedirs(poses_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        # ---------------------------------------------------------
        # PHASE 1: Pre-Cache Depths & Compute Global RTs
        # ---------------------------------------------------------
        print(f"\n--- Phase 1: Caching Depths and MapAnything Poses ---")
        
        image_cache = {}
        panoramas = []
        target_size_wh = tuple(cfg.input_size)
        target_size_hw = target_size_wh[::-1]

        # A) Prep panoramas for MapAnything Pose Estimation
        panoramas = load_panoramas(image_paths, target_size_wh=target_size_wh, save_dir=vis_dir)

        # B) Run MapAnything for Poses
        rts_mapanything = estimators.mapanything.run(
            panoramas, 
            seq_id, 
            save_results=True, 
            output_dir=poses_dir, 
            image_names=image_names_for_print
        )

        # run DFRM pose estimation to get relative RTs for all pairs (for occlusion mask generation)
        # rts_mapanything = estimators.dfrm.estimate_trajectory(image_paths, seq_id)
        rts_numpy = to_numpy_matrices(rts_mapanything)

        
        # C) Run DepthEstimation class for MapAnything Depth
        if use_map_depth:
            print(f"[INFO] Extracting MapAnything Depth batch...")
            mapanything_depths, scaling_factors = estimators.depth.run_mapanything_batch(image_paths)

        # D) Cache DFRM single-image requirements (Image tensor + MapAnything depth)
        for idx, path in enumerate(image_paths):
            print(f"Caching MapAnything depth & image for {os.path.basename(path)}...")
            with torch.no_grad():
                cached_data = estimators.dfrm.preprocess_single_image(path)
            
            if use_map_depth:
                ma_depth_numpy = mapanything_depths[idx]
                if (ma_depth_numpy.shape[1], ma_depth_numpy.shape[0]) != target_size_wh:
                    ma_depth_numpy = cv2.resize(ma_depth_numpy, target_size_wh, interpolation=cv2.INTER_NEAREST)

                depth_tensor = torch.from_numpy(ma_depth_numpy)
            else:
                depth_tensor = cached_data["depth"]  # [1, H, W], DAP

            print(f"Original Depth Tensor Shape: {depth_tensor.shape}, Min: {depth_tensor.min().item():.4f}, Max: {depth_tensor.max().item():.4f}, Median: {torch.median(depth_tensor).item():.4f}")

            # Post-process to fix zero depth values in the sky region
            height, width = depth_tensor.shape
            is_top_half = torch.zeros_like(depth_tensor, dtype=torch.bool)
            is_top_half[..., :height//2, :] = True
            zero_mask_1 = depth_tensor < 2.5 # DAP
            # zero_mask_1 = depth_tensor < 1e-3 # MapAnything

            if zero_mask_1.any():
                print("[INFO] Found zero depth values in the depth map. Applying post-processing to fix sky regions.")
                depth_tensor[zero_mask_1 & is_top_half] = 100.0  # Force Sky to Max Depth
            
            cached_data["depth"] = depth_tensor
            # cached_data["ma_scale"] = scaling_factors[idx]

            image_cache[path] = cached_data

        # plots relative poses
        for i, (rt, name) in enumerate(zip(rts_numpy, image_names_for_print)):
            if rt is not None and i > 0:
                estimators.mapanything.print_human_readable_pose(rt, name=f"{image_names_for_print[0]} ➔ {name})")
        
        unique_image_name = "_".join(image_names_for_print) 
        estimators.mapanything.verify_poses_visually(rts_numpy, image_names_for_print, sequence_id=seq_id, image_name=unique_image_name, output_dir=poses_dir)

        
        del panoramas
        gc.collect()
        torch.cuda.empty_cache()


        # save plot depth maps for all images in the sequence in image_cache depth
        if plot_depth_maps:
            for path in image_paths:
                depth_tensor = image_cache[path]['depth']
                depth_map = depth_tensor.squeeze().cpu().numpy()
                plt.figure(figsize=(8, 6))
                plt.imshow(depth_map, cmap='inferno')
                plt.title(f"Depth Map for {os.path.basename(path).split('.')[0]}")
                plt.colorbar(label='Depth Value')
                plt.axis('off')
                plt.savefig(os.path.join(depth_dir, f"{os.path.basename(path).split('.')[0]}.png"))
                plt.close()


        # ---------------------------------------------------------
        # PHASE 2: Combinatorial Occlusion Masks
        # ---------------------------------------------------------
        print(f"\n--- Phase 2: Computing Occlusion Masks for All Pairs ---")
        
        # Create an index lookup to easily grab the corresponding MapAnything RT
        path_to_idx = {path: idx for idx, path in enumerate(image_paths)}
        all_pairs = list(itertools.combinations(image_paths, 2))
        
        for img_a_path, img_b_path in all_pairs:
            name_a = os.path.basename(img_a_path).split('.')[0]
            name_b = os.path.basename(img_b_path).split('.')[0]
            idx_a, idx_b = path_to_idx[img_a_path], path_to_idx[img_b_path]
            
            print(f"  -> Processing pair: {name_a} & {name_b}")

            # 1. Fetch from cache and prepare the pair batch (only calculates correspondences)
            with torch.no_grad():
                batch = estimators.dfrm.prepare_pair_from_cache(
                    image_cache[img_a_path], 
                    image_cache[img_b_path]
                )
            
            # # 2. Calculate RELATIVE RTs from MapAnything Global Poses
            pose_A = rts_mapanything[idx_a].to(device)
            pose_B = rts_mapanything[idx_b].to(device)
            
            Rt_1_to_2 = torch.inverse(pose_B) @ pose_A 
            Rt_2_to_1 = torch.inverse(pose_A) @ pose_B

            # 3. Generate visual tensors
            with torch.no_grad():
                vis_outputs = estimators.dfrm.generate_occlusion_mask_tensors(batch, Rt_1_to_2, Rt_2_to_1)
            
            # 4. Save results
            pair_label = f"{name_a}_to_{name_b}"
            vutils.save_image(vis_outputs["side_by_side_1"], os.path.join(occlusion_dir, f"{pair_label}_side_by_side_1.png"))
            vutils.save_image(vis_outputs["side_by_side_2"], os.path.join(occlusion_dir, f"{pair_label}_side_by_side_2.png"))
            vutils.save_image(vis_outputs["mask_1"], os.path.join(occlusion_dir, f"{pair_label}_mask_1.png"))
            vutils.save_image(vis_outputs["mask_2"], os.path.join(occlusion_dir, f"{pair_label}_mask_2.png"))


            # save correspondences visualization
            if plot_correspondences:
                # [1, N, 2] -> [N, 2]
                pts1 = batch["points1"][0].cpu() 
                pts2 = batch["points2"][0].cpu()
                
                num_to_plot = len(pts1) 
                
                best_pts1 = pts1[:num_to_plot]
                best_pts2 = pts2[:num_to_plot]
                
                save_path = os.path.join(correspondences_dir, f"{pair_label}.png")
                estimators.dfrm.plot_correspondences( 
                    batch["image1"][0].cpu(), # [1, C, H, W] -> [C, H, W]
                    batch["image2"][0].cpu(), 
                    best_pts1, 
                    best_pts2, 
                    save_path=save_path
                ) 
                print(f"  Correspondences visualization saved → {save_path}\n")
            
            # Cleanup memory per pair
            del batch, vis_outputs, pose_A, pose_B, Rt_1_to_2, Rt_2_to_1
            gc.collect()
            torch.cuda.empty_cache()

        print(f"[SUCCESS] Finished sequence {seq_id}\n")


if __name__ == "__main__":

    config_file = os.path.join(os.path.dirname(__file__), "config.yml")
    cfg = get_easy_dict_from_yaml_file(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup configurations
    estimator_dfrm = DFRMPoseEstimator(cfg, device=device)
    estimator_mapanything = CameraPoseEstimator(cfg, model_name="mapanything", device=device)
    estimator_da3 = CameraPoseEstimator(cfg, model_name="depthanything", device=device)

    panorama_pairs = [
        # ('1.png', ),
        ('01.jpg','02.jpg', '03.jpg', '04.jpg', '05.jpg'),
        ('02_to_03_buildings_before.png', '02_to_03_buildings_inpainted.png'),
        ('01_prev_2.jpg', '02_prev_1.jpg', '03_center.jpg', '04_next_1.jpg', '05_next_2.jpg'),
        ('1.jpg', '2.jpg', '3.jpg', '4.jpg'),
        ('01_prev_2.jpg', '02_prev_1.jpg', '03_center.jpg', '04_next_1.jpg', '05_next_2.jpg',),
        ('01_prev_2.jpg', '02_prev_1.jpg', '03_center.jpg', '04_next_1.jpg', '05_next_2.jpg',),
        ('panorama3_original.png','panorama3_same street, heavy snow, _canny_depth_segmented_sd_strength0.8.png'),
        ('panorama7_original.png','panorama7_same street, heavy snow, _canny_depth_segmented_sd_strength0.8.png'),
        ('panorama8_original.png','panorama8_same street, heavy snow, _canny_depth_segmented_sd_strength0.8.png'),

    ]

    sequence_ids = [
        # "samy",
        "montreuil_rue_du_berger",
        "90fe3d58-5255-4542-bb5b-9ad552160f3f",
        "0b1aefa9-2a60-4ae7-a208-f6a934065086", 
        "argentina_835-Calle-57-La-Plata_11-2024", 
        "0b0838fe-7e19-4099-a77b-bc09fb406873",
        "0555c731-9dfb-4c23-8440-283d2fa20f69",
        "fake",
        "fake",
        "fake",

    ]

    # compare DFRM RT with CameraPoseEstimator RT on the same panorama pair
    if cfg.execution_mode == "compare_pose":
        for seq_id , panorama_names in zip(sequence_ids, panorama_pairs):
            print(f"\n\n{'#'*80}\nProcessing Sequence: {seq_id}, Panoramas: {panorama_names}\n{'#'*80}\n")
            panorama_paths = [os.path.join(cfg['data_path_ext'], seq_id, name) for name in panorama_names]
            compare_RT_computed(panorama_paths, seq_id, cfg, estimator_da3, estimator_mapanything, estimator_dfrm)    
            gc.collect()
            torch.cuda.empty_cache()

    # occlusion mask generation for a given panorama pair
    if cfg.execution_mode == "occlusion_mask":
        for seq_id , (panorama_a_name, panorama_b_name) in zip(sequence_ids, panorama_pairs):
            print(f"\n\n{'#'*80}\nProcessing Sequence: {seq_id}, Panoramas: {panorama_a_name} -> {panorama_b_name}\n{'#'*80}\n")
            panorama_a_path = os.path.join(cfg['data_path_ext'], seq_id, panorama_a_name)
            panorama_b_path = os.path.join(cfg['data_path_ext'], seq_id, panorama_b_name)
            save_dir = os.path.join(cfg['output_dir_ext'], 'occlusion_masks', seq_id)
            compare_occlusion_masks(panorama_a_path, panorama_b_path, seq_id, save_dir, cfg, device, estimator_da3, estimator_mapanything, estimator_dfrm)

    # compare depth maps generated by mapanything and DepthAnything3 for the same panorama
    if cfg.execution_mode == "compare_depth":
        depth_estimator = DepthEstimation(cfg)
        for seq_id , panorama_names in zip(sequence_ids, panorama_pairs):
            print(f"\n\n{'#'*80}\nProcessing Sequence: {seq_id}, Panoramas: {panorama_names}\n{'#'*80}\n")
            panorama_paths = [os.path.join(cfg['data_path_ext'], seq_id, name) for name in panorama_names]
            for path in panorama_paths:
                depth_estimator.compare(path, seq_id)
    
    # full pipeline execution: computes global RTs with MapAnything, generates occlusion masks for all pairs, and saves visualizations
    if cfg.execution_mode == "pipeline":
        estimators = EasyDict()
        estimators.dfrm = estimator_dfrm
        estimators.mapanything = estimator_mapanything
        estimators.depth = DepthEstimation(cfg, mode="single")
        torch.set_float32_matmul_precision('highest')
        run_full_pipeline(panorama_pairs, sequence_ids, cfg, estimators, device)
    