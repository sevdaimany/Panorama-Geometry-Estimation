import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import cv2

# DAP Imports
import DAP.test.infer as DAP_infer 
from DAP.test.infer import load_model as DAP_load_model

# MapAnything Imports
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils.image import preprocess_inputs

from PIL import Image


class DepthEstimation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load both models
        self.dap_model = self._load_dap_depth_model()
        self.mapanything_model = self._load_mapanything_model()

    def _load_dap_depth_model(self):
        config_path = self.cfg.dap_config_path
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config["load_weights_dir"] = self.cfg.dap_load_weights_dir
        
        model, _ = DAP_load_model(config)
        model.to(self.device)
        model.eval()
        print("[INFO] DAP Depth Config loaded.")
        return model

    def _load_mapanything_model(self):
        print("[INFO] Loading MapAnything Config...")
        model = MapAnything.from_pretrained(self.cfg.mapanything_model_name).to(self.device)
        model.eval()
        return model

    def _load_and_preprocess_panorama(self, panorama_path):
        img = Image.open(panorama_path).convert('RGB')
        if img is None:
            raise FileNotFoundError(f"Could not load image at {panorama_path}")
        
        width, height = self.cfg.input_size
        img = img.resize((width, height), resample=Image.BICUBIC)
        
        return np.array(img)
        
    def run_dap(self, panorama_path):
        # numpy array (H, W, 3)
        panorama_img_u8 = self._load_and_preprocess_panorama(panorama_path)
        
        with torch.no_grad():
            depth_map = DAP_infer.infer_raw(self.dap_model, self.device, panorama_img_u8)
            
        # Ensure it's a 2D numpy array (H, W)
        if torch.is_tensor(depth_map):
            depth_map = depth_map.squeeze().cpu().numpy()
            
        return depth_map
    
    def run_dap_batch(self, panorama_paths):
        # Load and preprocess all panoramas
        panorama_tensors = [self._load_and_preprocess_panorama(path) for path in panorama_paths]
        panorama_tensors = torch.stack(panorama_tensors)
        
        # panorama_tensors should be shape (B, C, H, W)
        with torch.no_grad():
            depth_maps = DAP_infer.infer_raw(self.dap_model, self.device, panorama_tensors)
        
        # Squeeze out the channel dimension and convert to a list of numpy arrays
        # Shape goes from (B, 1, H, W) -> (B, H, W)
        if torch.is_tensor(depth_maps):
            depth_maps = depth_maps.squeeze(1).cpu().numpy()
            
        # Convert the numpy batch array into a list of individual 2D arrays
        return [depth_map for depth_map in depth_maps]

    
    def run_mapanything(self, panorama_path):
        """
        Runs MapAnything using a specifically requested input size.
        """
        width, height = self.cfg.input_size
        views = load_images([panorama_path], resize_mode = "fixed_size", size= (width, height))

        with torch.no_grad():
            predictions = self.mapanything_model.infer(
                views, 
                memory_efficient_inference=True,
                use_amp=True,
                apply_mask=True
            )
            
        # 5. Extract Z-depth (Batch, Height, Width, 1) -> (Height, Width)
        depth_z = predictions[0]["depth_z"].squeeze().cpu().numpy()
        
        if (depth_z.shape[1], depth_z.shape[0]) != (width, height):
            print(f"[WARNING] MapAnything output size {depth_z.shape} does not match requested size {(height, width)}. Resizing output depth map to match.")
            depth_z = cv2.resize(
                depth_z, 
                (width, height), 
                interpolation=cv2.INTER_LINEAR
            )
            
        return depth_z
    
    def run_mapanything_batch(self, panorama_paths):
        # panorama_paths is a list: ["img1.jpg", "img2.jpg", ...]
        width, height = self.cfg.input_size
        views = load_images(panorama_paths, resize_mode = "fixed_size", size= (width, height))
        
        with torch.no_grad():
            predictions = self.mapanything_model.infer(
                views, 
                memory_efficient_inference=True,
                use_amp=True,
                apply_mask=True
            )
        
        # Extract Z-depth for all images in the batch
        # Returns a list of 2D numpy arrays
        depth_maps = []
        for pred in predictions:
            depth_maps.append(pred["depth_z"].squeeze().cpu().numpy())
            
        return depth_maps

    def compare(self, panorama_path, seq_id):
        """
        Runs both models, aligns their scales, calculates error metrics, 
        and plots a visual comparison.
        """
        print(f"[INFO] Running inference on: {panorama_path}")
        original_image_rgb = Image.open(panorama_path).convert('RGB')
        
        # Inference
        dap_depth = self.run_dap(panorama_path)
        ma_depth = self.run_mapanything(panorama_path)

        # MapAnything might output a different resolution. Resize to match DAP.
        print(f"[INFO] DAP Depth Shape: {dap_depth.shape}, MapAnything Depth Shape: {ma_depth.shape}")
        if dap_depth.shape != ma_depth.shape:
            ma_depth = cv2.resize(
                ma_depth, 
                (dap_depth.shape[1], dap_depth.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )

        print(f"Median DAP Depth (Masked): {np.median(dap_depth[dap_depth > 0]):.4f}")
        print(f"Median MapAnything Depth (Masked): {np.median(ma_depth[ma_depth > 0]):.4f}")
        
        mask = (dap_depth > 0) & (ma_depth > 0)
        print(f"\n--- Numerical Comparison ---")
        print(f"Scale Factor: {np.median(dap_depth[mask]) / np.median(ma_depth[mask]):.4f}")

        diff_map = np.abs(dap_depth - ma_depth)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        axes[0, 0].imshow(original_image_rgb)
        axes[0, 0].set_title("Original Panorama (RGB)")
        axes[0, 0].axis('off')

        im1 = axes[0, 1].imshow(dap_depth, cmap='inferno')
        axes[0, 1].set_title("DAP Output")
        axes[0, 1].axis('off')
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[1, 0].imshow(ma_depth, cmap='inferno')
        axes[1, 0].set_title("MapAnything Output")
        axes[1, 0].axis('off')
        fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im3 = axes[1, 1].imshow(diff_map, cmap='hot')
        axes[1, 1].set_title("Absolute Difference Map")
        axes[1, 1].axis('off')
        fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        os.makedirs(os.path.join(self.cfg.output_dir_ext, "depth"), exist_ok=True)
        plt.savefig(os.path.join(self.cfg.output_dir_ext, "depth", f"{seq_id}_{os.path.basename(panorama_path)}_depth_comparison.png"))
        plt.show()

        return dap_depth, ma_depth