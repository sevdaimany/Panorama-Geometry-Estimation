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
    random.seed(seed)
    np.random.seed(seed)    
    cv2.setRNGSeed(seed)    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)     
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


def create_folders(cfg, city_name, seq_id):
    vis_dir = os.path.join(cfg['output_dir_root'], "pipeline_visualization", city_name, seq_id)
    occlusion_dir = os.path.join(vis_dir, 'occlusion_masks')
    poses_dir = os.path.join(vis_dir, 'trajectory')
    depth_dir = os.path.join(vis_dir, 'depth')

    output_data_dir = os.path.join(cfg['output_dir_root'], 'pipeline_data', city_name, seq_id)
    depth_data_dir = os.path.join(output_data_dir, 'depth')
    poses_data_dir = os.path.join(output_data_dir, 'poses')
    occlusion_data_dir = os.path.join(output_data_dir, 'occlusion')

    for d in [occlusion_dir, poses_dir, depth_dir,
          depth_data_dir, poses_data_dir, occlusion_data_dir]:
        os.makedirs(d, exist_ok=True)

    return {
        "vis_dir": vis_dir,
        "occlusion_dir": occlusion_dir,
        "poses_dir": poses_dir,
        "depth_dir": depth_dir,
        "depth_data_dir": depth_data_dir,
        "poses_data_dir": poses_data_dir,
        "occlusion_data_dir": occlusion_data_dir
    }

def save_depth_data(depth_tensor, save_path):
    depth_numpy = depth_tensor.squeeze().cpu().numpy()
    np.savez_compressed(save_path, depth=depth_numpy)
    print(f"Saved depth data to {save_path}")

def save_depth_visualization(depth_tensor, save_path):
    depth_numpy = depth_tensor.squeeze().cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_numpy, cmap='inferno')
    plt.title(f"Depth Map Visualization")
    plt.colorbar(label='Depth Value')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved depth visualization to {save_path}")

def save_occlusion_visualization(vis_outputs, save_path, pair_label):
    vutils.save_image(vis_outputs["side_by_side_1"], os.path.join(save_path, f"{pair_label}_side_by_side_1.png"))
    vutils.save_image(vis_outputs["side_by_side_2"], os.path.join(save_path, f"{pair_label}_side_by_side_2.png"))
    vutils.save_image(vis_outputs["mask_1"], os.path.join(save_path, f"{pair_label}_mask_1.png"))
    vutils.save_image(vis_outputs["mask_2"], os.path.join(save_path, f"{pair_label}_mask_2.png"))

def save_occlusion_data(vis_outputs, save_path, pair_label):
    mask_1_np = vis_outputs["mask_1"].squeeze().cpu().numpy()
    mask_2_np = vis_outputs["mask_2"].squeeze().cpu().numpy()
    
    np.savez_compressed(
        os.path.join(save_path, f"{pair_label}_masks.npz"),
        mask_1=mask_1_np,
        mask_2=mask_2_np
    )

def save_pose_global_data(rts_numpy, save_path, sequence_id, image_names):
    poses_clean = np.array(rts_numpy, dtype=np.float32)
    np.savez(
        os.path.join(save_path, f"global_poses.npz"),
        poses=poses_clean,    # (N, 4, 4)
        image_names=np.array(image_names),   # (N,)
        reference_frame=image_names[0],
    )
    
def save_pose_relative_data(relative_poses_dict, save_path, sequence_id, image_names):
    pair_keys = list(relative_poses_dict.keys())
    np.savez(
        os.path.join(save_path, f"relative_poses.npz"),
        pair_indices=np.array(pair_keys, dtype=np.int32),  # (M, 2)
        pair_names=np.array([f"{image_names[a]}__{image_names[b]}" for a, b in pair_keys]),
        relative_poses=np.stack([relative_poses_dict[k] for k in pair_keys]).astype(np.float32),  # (M, 4, 4)
    )
    


def run_pipeline(city_name, seq_id, cfg, estimators, device):
    # input
    input_dir = os.path.join(cfg['input_dir_root'], city_name, seq_id) 
    image_exts = ('.jpg', '.jpeg', '.png')
    image_names = sorted([ f for f in os.listdir(input_dir) if f.lower().endswith(image_exts)]) 
    
    print(f"\n{'#'*80}\nFound {len(image_names)} images for sequence {seq_id} in {input_dir}.\n{'#'*80}\n")
    image_paths = [os.path.join(input_dir, name) for name in image_names]
    path_to_name = {path: name.split('.')[0] for path, name in zip(image_paths, image_names)}
    image_names_for_print = list(path_to_name.values())

    folder_dict = create_folders(cfg, city_name, seq_id)
    vis_dir, occlusion_dir, poses_dir, depth_dir = folder_dict["vis_dir"], folder_dict["occlusion_dir"], folder_dict["poses_dir"], folder_dict["depth_dir"]
    depth_data_dir, poses_data_dir, occlusion_data_dir = folder_dict["depth_data_dir"], folder_dict["poses_data_dir"], folder_dict["occlusion_data_dir"]

    # ---------------------------------------------------------
    # PHASE 1: Pre-Cache Depths & Compute Global RTs
    # ---------------------------------------------------------
    print(f"\n--- Phase 1: Caching Depths and MapAnything Poses ---")
    image_cache = {}
    target_size_wh = tuple(cfg.input_size)

    # Run MapAnything for Poses
    panoramas = load_panoramas(image_paths, target_size_wh=target_size_wh, save_dir=None)
    rts_mapanything = estimators.mapanything.run(panoramas, seq_id)
    rts_numpy = to_numpy_matrices(rts_mapanything)

    # --- Save global poses ---
    # 1. Convert to an array with float dtype (None becomes np.nan in float arrays)
    save_pose_global_data(rts_numpy, poses_data_dir, seq_id, image_names)
    
    # info relative poses
    for i, (rt, name) in enumerate(zip(rts_numpy, image_names_for_print)):
        if rt is not None and i > 0:
            estimators.mapanything.print_human_readable_pose(rt, name=f"{image_names_for_print[0]} ➔ {name})")
    
    estimators.mapanything.verify_poses_visually(rts_numpy, image_names_for_print, sequence_id=seq_id, image_name=seq_id, output_dir=poses_dir)

    # Run DepthEstimation class for MapAnything Depth
    print(f"[INFO] Extracting MapAnything Depth batch...")
    mapanything_depths, scaling_factors = estimators.depth.run_mapanything_batch(image_paths)

    # (Image tensor + MapAnything depth)
    for idx, path in enumerate(image_paths):
        print(f"Caching MapAnything depth & image for {os.path.basename(path)}...")
        with torch.no_grad():
            cached_data = estimators.dfrm.preprocess_single_image(path)
        
        ma_depth_numpy = mapanything_depths[idx]
        if (ma_depth_numpy.shape[1], ma_depth_numpy.shape[0]) != target_size_wh:
            ma_depth_numpy = cv2.resize(ma_depth_numpy, target_size_wh, interpolation=cv2.INTER_NEAREST)
        depth_tensor = torch.from_numpy(ma_depth_numpy)

        # Sky-only fix. The road is fine as-is.
        H_d, W_d = depth_tensor.shape
        v_grid = torch.linspace(0, 1, H_d, device=depth_tensor.device).view(H_d, 1).expand(H_d, W_d)
        is_above_horizon = v_grid < 0.48
        sky_broken = (depth_tensor <= 1e-3) & is_above_horizon   # DAP floor pixels above horizon
        depth_tensor = torch.where(sky_broken, torch.full_like(depth_tensor, 100.0), depth_tensor)
    

        cached_data["depth"] = depth_tensor
        image_cache[path] = cached_data
        base_name = path_to_name[path]
        save_depth_data(depth_tensor, os.path.join(depth_data_dir, f"{base_name}.npz"))
        save_depth_visualization(depth_tensor, os.path.join(depth_dir, f"{base_name}.png"))

    del panoramas
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # PHASE 2: Combinatorial Occlusion Masks
    # ---------------------------------------------------------
    print(f"\n--- Phase 2: Computing Occlusion Masks for All Pairs ---")
    
    path_to_idx = {path: idx for idx, path in enumerate(image_paths)}
    all_pairs = list(itertools.combinations(image_paths, 2))
    relative_poses = {}  # (idx_a, idx_b) -> 4x4
    
    for img_a_path, img_b_path in all_pairs:
        name_a = path_to_name[img_a_path]
        name_b = path_to_name[img_b_path]
        idx_a, idx_b = path_to_idx[img_a_path], path_to_idx[img_b_path]
        
        print(f"  -> Processing pair: {name_a} & {name_b}")

        with torch.no_grad():
            batch = estimators.dfrm.prepare_pair_from_cache(
                image_cache[img_a_path], 
                image_cache[img_b_path]
            )
        
        # Calculate RELATIVE RTs from MapAnything Global Poses
        pose_A = rts_mapanything[idx_a].to(device)
        pose_B = rts_mapanything[idx_b].to(device)
        
        Rt_1_to_2 = torch.inverse(pose_B) @ pose_A 
        Rt_2_to_1 = torch.inverse(pose_A) @ pose_B

        relative_poses[(idx_a, idx_b)] = Rt_1_to_2.cpu().numpy()
        relative_poses[(idx_b, idx_a)] = Rt_2_to_1.cpu().numpy()

        # Generate visual tensors
        with torch.no_grad():
            vis_outputs = estimators.dfrm.generate_occlusion_mask_tensors(batch, Rt_1_to_2, Rt_2_to_1)
        
        # Save results
        pair_label = f"{name_a}_to_{name_b}"
        save_occlusion_visualization(vis_outputs, occlusion_dir, pair_label)
        save_occlusion_data(vis_outputs, occlusion_data_dir, pair_label)
    
        # Cleanup memory per pair
        del batch, vis_outputs, pose_A, pose_B, Rt_1_to_2, Rt_2_to_1
        gc.collect()
        torch.cuda.empty_cache()
    
    save_pose_relative_data(relative_poses, poses_data_dir, seq_id, image_names_for_print)
    print(f"[SUCCESS] Finished sequence {seq_id}\n")



def run_full_pipeline(cfg, estimators, device):

    # loop through main folder to find city folders, then sequence folders, and run the pipeline for each sequence
    city_folders = sorted(os.listdir(cfg['input_dir_root']))
    for city_name in city_folders:
        city_path = os.path.join(cfg['input_dir_root'], city_name)
        if not os.path.isdir(city_path):
            continue
        sequence_folders = sorted(os.listdir(city_path))
        
        print(f"\n\n{'#'*80}\nStarting pipeline for City: {city_name}\n{'#'*80}\n")
        for seq_id in sequence_folders:
            seq_path = os.path.join(city_path, seq_id)
            if not os.path.isdir(seq_path):
                continue

            run_pipeline(city_name, seq_id, cfg, estimators, device)


def input_examples():
    panorama_pairs = [
        # ('1.png', ),
        ('01_prev_2.jpg', '02_prev_1.jpg', '03_center.jpg', '04_next_1.jpg', '05_next_2.jpg', 'buffer_01.jpg', 'buffer_02.jpg', 'buffer_03.jpg', 'buffer_04.jpg', 'buffer_05.jpg'),
        ('01_prev_2.jpg', '02_prev_1.jpg', '03_center.jpg', '04_next_1.jpg', '05_next_2.jpg', 'buffer_01.jpg', 'buffer_02.jpg', 'buffer_03.jpg', 'buffer_04.jpg', 'buffer_05.jpg'),
        ('00000090_before.png', '00000090_after.png'),
        ('01.jpg','02.jpg', '03.jpg', '04.jpg', '05.jpg', '06.jpg', '07.jpg'),
        ('02_to_03_buildings_before.png', '02_to_03_buildings_inpainted.png'),
        ('01_prev_2.jpg', '02_prev_1.jpg', '03_center.jpg', '04_next_1.jpg', '05_next_2.jpg', 'buffer_01.jpg', 'buffer_02.jpg', 'buffer_03.jpg', 'buffer_04.jpg', 'buffer_05.jpg'),
        ('1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg'),
        ('01_prev_2.jpg', '02_prev_1.jpg', '03_center.jpg', '04_next_1.jpg', '05_next_2.jpg',),
        ('01_prev_2.jpg', '02_prev_1.jpg', '03_center.jpg', '04_next_1.jpg', '05_next_2.jpg',),
        ('panorama3_original.png','panorama3_same street, heavy snow, _canny_depth_segmented_sd_strength0.8.png'),
        ('panorama7_original.png','panorama7_same street, heavy snow, _canny_depth_segmented_sd_strength0.8.png'),
        ('panorama8_original.png','panorama8_same street, heavy snow, _canny_depth_segmented_sd_strength0.8.png'),

    ]

    sequence_ids = [
        # "samy",
        "4b07592d-8a72-4f25-be2c-5ea3da6b7fcc",
        "1cef0942-8e2f-4c9d-9cde-deca11c63a0c",
        "PSCD_1",
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
    return panorama_pairs, sequence_ids 

if __name__ == "__main__":

    config_file = os.path.join(os.path.dirname(__file__), "config.yml")
    cfg = get_easy_dict_from_yaml_file(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # panorama_pairs, sequence_ids = input_examples()


    # compare DFRM RT with CameraPoseEstimator RT on the same panorama pair
    if cfg.execution_mode == "compare_pose":
        estimator_dfrm = DFRMPoseEstimator(cfg, device=device)
        estimator_mapanything = CameraPoseEstimator(cfg, model_name="mapanything", device=device)   
        estimator_da3 = CameraPoseEstimator(cfg, model_name="depthanything", device=device)
    
        for seq_id , panorama_names in zip(sequence_ids, panorama_pairs):
            print(f"\n\n{'#'*80}\nProcessing Sequence: {seq_id}, Panoramas: {panorama_names}\n{'#'*80}\n")
            panorama_paths = [os.path.join(cfg['data_path_ext'], seq_id, name) for name in panorama_names]
            compare_RT_computed(panorama_paths, seq_id, cfg, estimator_da3, estimator_mapanything, estimator_dfrm)    
            gc.collect()
            torch.cuda.empty_cache()

    # occlusion mask generation for a given panorama pair
    if cfg.execution_mode == "occlusion_mask":
        estimator_dfrm = DFRMPoseEstimator(cfg, device=device)
        estimator_mapanything = CameraPoseEstimator(cfg, model_name="mapanything", device=device)   
        estimator_da3 = CameraPoseEstimator(cfg, model_name="depthanything", device=device)

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
        estimators.dfrm = DFRMPoseEstimator(cfg, device=device, mode="pipeline")
        estimators.mapanything = CameraPoseEstimator(cfg, model_name="mapanything", device=device)
        estimators.depth = DepthEstimation(cfg, mode="single")
        torch.set_float32_matmul_precision('highest')
        run_full_pipeline(cfg, estimators, device)
    