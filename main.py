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

def compare_RT_computed(image_paths, sequence_id, cfg, estimator_da3, estimator_mapanything, estimator_dfrm):
    """
    Compares the trajectory (RTs) computed sequentially by DFRM and the CameraPoseEstimators 
    (using MapAnything or DepthAnything3) across N images.
    """
    num_images = len(image_paths)
    print(f"\n{'='*50}\nComparing Trajectories for {num_images} images...\n{'='*50}")

    output_dir = cfg['output_dir_ext']
    cubemap_dir = os.path.join(output_dir, sequence_id, 'cubemaps')
    os.makedirs(cubemap_dir, exist_ok=True)
    
    # Extract names for saving and plotting
    image_names = [os.path.basename(p).split('.')[0] for p in image_paths]

    # 1. Compute DFRM RT (Sequential)
    torch.set_float32_matmul_precision('highest')
    print(f"\n{'='*50}\n[INFO] Running DFRM Estimator...\n{'='*50}")
    accumulated_rts_dfrm = estimator_dfrm.estimate_trajectory(image_paths, sequence_id)

    # 2. Load Panoramas for Extrinsic Estimation
    panoramas = []
    for path, name in zip(image_paths, image_names):
        pano = cv2.imread(path)

        # resize if doesnt match cfg.input_size
        target_size_wh = tuple(cfg.input_size) # (W, H)
        target_size_hw = target_size_wh[::-1]  # (H, W)
        if pano.shape != target_size_hw: # OpenCV uses (height, width)
            print(f"[WARNING] Resizing panorama '{name}' from {pano.shape[1]}x{pano.shape[0]} to {target_size_wh[0]}x{target_size_wh[1]}.")
            pano = cv2.resize(pano, target_size_wh, interpolation=cv2.INTER_AREA)

        panoramas.append(pano)
        # Save a copy to the output directory
        cv2.imwrite(os.path.join(cubemap_dir, f"{name}.jpg"), pano)

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

def compare_occlusion_masks(image_a_path, image_b_path, sequence_id, save_dir, args, device, estimator_da3, estimator_mapanything, estimator_dfrm):
    print(f"\n{'='*50}\nComparing Occlusion Masks...\n{'='*50}")

    torch.set_float32_matmul_precision('highest')
    dfrm_trajectory, batch = estimator_dfrm.estimate_trajectory([image_a_path, image_b_path], sequence_id, return_batch_for_one_pair=True)
    
    Rt_1_to_2 = torch.inverse(dfrm_trajectory[1])
    Rt_2_to_1 = dfrm_trajectory[1] 

    estimator_dfrm.generate_occlusion_mask(image_a_path, image_b_path, sequence_id, save_dir, Rt_1_to_2, Rt_2_to_1, model_name="_dfrm", batch=batch)
    
    pano_A = cv2.imread(image_a_path)
    pano_B = cv2.imread(image_b_path)
    
    ma_traj = estimator_mapanything.run([pano_A, pano_B], sequence_id)
    estimator_dfrm.generate_occlusion_mask(image_a_path, image_b_path, sequence_id, save_dir, torch.inverse(ma_traj[1]), ma_traj[1], model_name="_mapanything", batch=batch)

    da3_traj = estimator_da3.run([pano_A, pano_B], sequence_id)
    estimator_dfrm.generate_occlusion_mask(image_a_path, image_b_path, sequence_id, save_dir, torch.inverse(da3_traj[1]), da3_traj[1], model_name="_depthanything3", batch=batch)


# compare depth
if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "config.yml")
    args = get_easy_dict_from_yaml_file(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_estimator = DepthEstimation(args)


    panorama_pairs = [
        ('01_prev_2.jpg', '02_prev_1.jpg', '03_center.jpg', '04_next_1.jpg', '05_next_2.jpg',),
        ('01_prev_2.jpg', '02_prev_1.jpg', '03_center.jpg', '04_next_1.jpg', '05_next_2.jpg',),
        ('01_prev_2.jpg', '02_prev_1.jpg', '03_center.jpg', '04_next_1.jpg', '05_next_2.jpg',),
        ('1.jpg', '2.jpg', '3.jpg', '4.jpg')

    ]

    sequence_ids = [
        "0b1aefa9-2a60-4ae7-a208-f6a934065086",
        "0b0838fe-7e19-4099-a77b-bc09fb406873",
        "0555c731-9dfb-4c23-8440-283d2fa20f69",
        'argentina_835-Calle-57-La-Plata_11-2024'
    ]

    for seq_id , panorama_names in zip(sequence_ids, panorama_pairs):
        print(f"\n\n{'#'*80}\nProcessing Sequence: {seq_id}, Panoramas: {panorama_names}\n{'#'*80}\n")
        panorama_paths = [os.path.join(args['data_path_ext'], seq_id, name) for name in panorama_names]

        depth_estimator.compare(panorama_paths[0], seq_id)  # Compare depth maps for the first panorama as an example
        






# # occlusion mask generation for a given panorama pair, not completely debugged
# if __name__ == "__main__":

#     camerapose_occlusionmsk = "pose"# "pose" or "occlusion_mask"
#     config_file = os.path.join(os.path.dirname(__file__), "config.yml")
#     args = get_easy_dict_from_yaml_file(config_file)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Setup configurations
#     estimator_dfrm = DFRMPoseEstimator(args, device=device)

#     cfg_mapanything = EasyDict(copy.deepcopy(args))
#     cfg_mapanything['use_mapanything'] = True
#     estimator_mapanything = CameraPoseEstimator(cfg_mapanything, device=device)

#     cfg_da3 = EasyDict(copy.deepcopy(args))
#     cfg_da3['use_mapanything'] = False
#     estimator_da3 = CameraPoseEstimator(cfg_da3, device=device)


#     # panorama_pairs = [
#     #     ('01_to_03_buildings_after.png', '01_to_03_buildings_before.png'),
#     #     ('01_to_02_buildings_after.png', '01_to_02_buildings_before.png'),
#     #      ('01_to_03_buildings_after.png', '01_to_03_buildings_before.png'),
#     # ]

#     # sequence_ids = [
#     #     "2a0047d3-8cb0-47e9-ab51-2b59af8b0b5a",
#     #    "e2851a09-cc55-488a-a230-648e37feaf8b",
#     #     "f7c00cfc-b1c2-4acc-9986-79f9bbc5941d", 
#     # ]

#     # panorama_pairs = [
#     #     # ('01_prev_2.jpg', '02_prev_1.jpg'),
#     #     # ('03_center.jpg', 'buffer_01.jpg'),
#     #     # ('03_center.jpg', '04_next_1.jpg', 'buffer_01.jpg'),
#     #     # ('03_center.jpg', '04_next_1.jpg', 'buffer_04.jpg'),
        
#     #     # ('03_center.jpg', '04_next_1.jpg', 'buffer_02.jpg'),
#     #     # ('03_center.jpg', 'buffer_01.jpg'),

#     #     # ('02_prev_1.jpg', '03_center.jpg', 'buffer_01.jpg'),
#     # ]

#     # sequence_ids = [
#     #     "0b1aefa9-2a60-4ae7-a208-f6a934065086",
#     #     "0b1aefa9-2a60-4ae7-a208-f6a934065086",
#     #     "0b1aefa9-2a60-4ae7-a208-f6a934065086",
#     #     "0b1aefa9-2a60-4ae7-a208-f6a934065086",


#     #     "0b0838fe-7e19-4099-a77b-bc09fb406873",
#     #     "0b0838fe-7e19-4099-a77b-bc09fb406873",

#     #     "0555c731-9dfb-4c23-8440-283d2fa20f69",
#     # ]


#     panorama_pairs = [
#         # ('01_prev_2.jpg', '02_prev_1.jpg', '03_center.jpg', '04_next_1.jpg', '05_next_2.jpg',),
#         # ('01_prev_2.jpg', '02_prev_1.jpg', '03_center.jpg', '04_next_1.jpg', '05_next_2.jpg',),
#         # ('01_prev_2.jpg', '02_prev_1.jpg', '03_center.jpg', '04_next_1.jpg', '05_next_2.jpg',),
#         ('1.jpg', '2.jpg', '3.jpg', '4.jpg')

#     ]

#     sequence_ids = [
#         # "0b1aefa9-2a60-4ae7-a208-f6a934065086",
#         # "0b0838fe-7e19-4099-a77b-bc09fb406873",
#         # "0555c731-9dfb-4c23-8440-283d2fa20f69",
#         'argentina_835-Calle-57-La-Plata_11-2024'
#     ]


#     # panorama_pairs = [
#     #     ('panorama3_original.png','panorama3_same street, heavy snow, _canny_depth_segmented_sd_strength0.8.png'),
#     #     ('panorama7_original.png','panorama7_same street, heavy snow, _canny_depth_segmented_sd_strength0.8.png'),
#     #     ('panorama8_original.png','panorama8_same street, heavy snow, _canny_depth_segmented_sd_strength0.8.png'),


#     # ]

#     # sequence_ids = [
#     #     "fake",
#     #     "fake",
#     #     "fake",

#     # ]


#     # # compare DFRM RT with CameraPoseEstimator RT on the same panorama pair
#     if camerapose_occlusionmsk == "pose":
#         for seq_id , panorama_names in zip(sequence_ids, panorama_pairs):
#             print(f"\n\n{'#'*80}\nProcessing Sequence: {seq_id}, Panoramas: {panorama_names}\n{'#'*80}\n")
#             panorama_paths = [os.path.join(args['data_path_ext'], seq_id, name) for name in panorama_names]
#             compare_RT_computed(panorama_paths, seq_id, args, estimator_da3, estimator_mapanything, estimator_dfrm)    
#             gc.collect()
#             torch.cuda.empty_cache()

#     # occlusion mask generation for a given panorama pair, not completely debugged
#     if camerapose_occlusionmsk == "occlusion_mask":
#         for seq_id , (panorama_a_name, panorama_b_name) in zip(sequence_ids, panorama_pairs):
#             print(f"\n\n{'#'*80}\nProcessing Sequence: {seq_id}, Panoramas: {panorama_a_name} -> {panorama_b_name}\n{'#'*80}\n")
#             panorama_a_path = os.path.join(args['data_path_ext'], seq_id, panorama_a_name)
#             panorama_b_path = os.path.join(args['data_path_ext'], seq_id, panorama_b_name)
#             save_dir = os.path.join(args['output_dir_ext'], seq_id, 'occlusion_masks')
#             compare_occlusion_masks(panorama_a_path, panorama_b_path, seq_id, save_dir, args, device, estimator_da3, estimator_mapanything, estimator_dfrm)
