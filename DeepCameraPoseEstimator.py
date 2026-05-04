import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import py360convert
from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs
from depth_anything_3.api import DepthAnything3
import matplotlib.pyplot as plt
import gc
from DFRMCameraPoseEstimator import DFRMPoseEstimator

class CameraPoseEstimator:
    def __init__(self, cfg, device):
        self.device = device
        self.cfg = cfg
        self.face_size = self.cfg['face_size']
        self.translation_tolerance = self.cfg['translation_tolerance']
        self.fov_degrees = self.cfg['fov_degrees']
        self.intrinsic_matrix = self.get_intrinsic_matrix(self.fov_degrees)
        if self.cfg['use_mapanything']:
            self.model = MapAnything.from_pretrained(self.cfg['mapanything_model_name']).to(self.device)
            self.model_name = "mapanything"
        else:
            self.model = DepthAnything3.from_pretrained(self.cfg['depthanything_model_name']).to(self.device)
            self.model_name = "depthanything3"
        print(f'[INFO] Initialized CameraPoseEstimator with model "{self.model_name}" on device "{self.device}".')

    def get_intrinsic_matrix(self, fov_degrees):
        '''
        Calculates the intrinsic matrix for a given Field of View.
        '''
        fov_rad = np.deg2rad(fov_degrees)
        
        # True focal length formula based on FOV and image width
        focal_length = (self.face_size / 2.0) / np.tan(fov_rad / 2.0)
        
        K = np.array([
            [focal_length, 0, self.face_size / 2.0],
            [0, focal_length, self.face_size / 2.0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K

    def get_inverse_local_offsets(self):
        """
        Dynamically generates the 4x4 inverse transformation matrices 
        based on the number of views in the configuration.
        """
        offsets = {}
        num_views = self.cfg.get('num_views', 8)
        step_angle = 360.0 / num_views

        for i in range(num_views):
            yaw = i * step_angle
            name = f'yaw_{int(yaw)}'
            
            # The inverse of a positive Yaw is a negative Yaw
            inverse_yaw = -yaw
            
            # Create a 3x3 rotation matrix for the inverse yaw rotation
            r = R.from_euler('y', inverse_yaw, degrees=True).as_matrix()

            # Create the 4x4 Homogeneous transformation matrix
            inverse_transform = np.eye(4)
            inverse_transform[:3, :3] = r
            offsets[name] = inverse_transform
            
        return offsets
    
    def extract_horizontal_cubemap_faces(self, panorama_image, sequence_id, output_dir=None, image_name=None, save_results=False):
        """
        Extract N overlapping horizontal perspective views from the panorama.
        """
        fov_degrees = self.cfg['fov_degrees']
        num_views = self.cfg.get('num_views', 8)
        step_angle = 360.0 / num_views
        
        faces_dict = {}
        
        for i in range(num_views):
            yaw = i * step_angle
            name = f'yaw_{int(yaw)}' # e.g., 'yaw_0', 'yaw_45', etc.
            
            face = py360convert.e2p(
                panorama_image, 
                fov_deg=(fov_degrees, fov_degrees), 
                u_deg=yaw, 
                v_deg=0.0, 
                out_hw=(self.face_size, self.face_size), 
                in_rot_deg=0, 
                mode='bilinear'
            )
            faces_dict[name] = face

        if save_results and output_dir and image_name:
            self.save_faces_as_images(faces_dict, output_dir=output_dir, image_name=image_name, sequence_id=sequence_id)

        return faces_dict

    def save_faces_as_images(self, faces_dict, output_dir, image_name, sequence_id):
        '''
        save the cubemap faces in one plot for visualization
        '''
        face_names = list(faces_dict.keys())
        face_images = list(faces_dict.values())
        border_size = 10
        num_images = len(face_names)
        grid_image = np.zeros((self.face_size, self.face_size * num_images + border_size * (num_images - 1), 3), dtype=np.uint8)
        for i, face in enumerate(face_names):
            start_x = i * (self.face_size + border_size)
            grid_image[:, start_x:start_x + self.face_size] = face_images[i]

        cv2.imwrite(os.path.join(output_dir, f'{image_name}_cubemap_faces.png'), grid_image)
        print(f'[INFO] Saved cubemap faces for {image_name} to {os.path.join(output_dir, f"{image_name}_cubemap_faces.png")}')

    def inference_mapanything(self, face_images):
        '''
        Run MapAnything inference on the input face image to get camera poses.
        '''
        # MapAnything expects a list of dictionaries for multi-modal inputs
        ma_views = []

        for img in face_images:
            ma_views.append({
                'img': img,
                'intrinsics': self.intrinsic_matrix,
                'data_norm_type': ['dinov2']
            })

        processed_views = preprocess_inputs(ma_views)
        ma_predictions = self.model.infer(processed_views)
        print(f'[INFO] MapAnything raw predictions len: {len(ma_predictions)} views, each with keys: {ma_predictions[0].keys()}')
        # Extract the 4x4 matrices
        ma_poses = []
        for pred in ma_predictions:
            pose = pred['camera_poses'].cpu().numpy().squeeze(0)
            scale = pred['metric_scaling_factor'].item()
            ma_poses.append(pose)
            
        print(f'[INFO] MapAnything predicted poses shape: {ma_poses[0].shape}')
        return ma_poses
    
    def inference_depthanything(self, face_images):
        '''
        Run DepthAnything inference on the input face images to get camera poses.
        '''
        
        da3_intrinsics = np.stack([self.intrinsic_matrix] * len(face_images))

        da_predictions = self.model.inference(face_images, intrinsics=da3_intrinsics, use_ray_pose=True)

        da_poses = da_predictions.extrinsics
        print(f'[INFO] DepthAnything3 predicted poses shape: {da_poses.shape}')
        

        # Pad the 3x4 matrices to 4x4
        N = da_poses.shape[0]
        da_poses_4x4 = np.zeros((N, 4, 4), dtype=np.float32)
        da_poses_4x4[:, :3, :] = da_poses   # Copy the 3x4 data into the top
        da_poses_4x4[:, 3, 3] = 1.0             # Add the '1' to the bottom right
        
        # Convert from World-to-Camera (Extrinsics) to Camera-to-World (Poses)
        c2w_poses = []
        for i in range(N):
            w2c_matrix = da_poses_4x4[i]
            c2w_matrix = np.linalg.inv(w2c_matrix)
            c2w_poses.append(c2w_matrix)
            
        da_poses_final = np.array(c2w_poses)
        print(f'[INFO] DA3 formatted 4x4 Poses shape: {da_poses_final.shape}')
        return da_poses_final
    
    def compute_panorama_pose(self, predicted_face_poses, translation_tolerance=0.5):
        """
        Takes the glocal poses of the cubemap faces, applies the inverse offsets,
        filters out bad matches, and return the true panorama center pose.

        Args:
            :param predicted_face_poses: Dictionary {'yaw_0': Rt_matrix, 'yaw_45': Rt_matrix, ...}
                                    where Rt_matrix is a 4x4 numpy array.
            :param translation_tolerance: Max allowed distance (in meters/units) from the median center to be considered a "good" match.
        """
        inverse_offsets = self.get_inverse_local_offsets()

        predicted_centers = []
        predicted_rotations = []
        valid_faces = []

        # Calculate the predicted panorama center from every face
        for face, RT_global in predicted_face_poses.items():
            if RT_global is None:
                continue
            
            RT_center = RT_global @ inverse_offsets[face]
            
            # extracrt the translation (center) and rotation
            translation = RT_center[:3, 3]
            rotation_matrix = RT_center[:3, :3]

            predicted_centers.append(translation)
            predicted_rotations.append(rotation_matrix)
            valid_faces.append(face)
        
        if len(predicted_centers) == 0:
            print('!! [ERROR] No valid face poses provided.')
            return None
        
        predicted_centers = np.array(predicted_centers)
        print(f'[INFO] Predicted centers from valid faces: {predicted_centers}')
        median_center = np.median(predicted_centers, axis=0)

        # Filter out bad matches based on translation distance from the median center
        filtered_centers = []
        filtered_rotations = []

        for center, rotation, face in zip(predicted_centers, predicted_rotations, valid_faces):
            distance = np.linalg.norm(center - median_center)
            if distance <= translation_tolerance:
                filtered_centers.append(center)
                filtered_rotations.append(rotation)
                print(f'[INFO] Face "{face}" is a good match (distance {distance:.2f} <= tolerance {translation_tolerance}).')
            else:
                print(f'!! [WARNING] Face "{face}" is a bad match (distance {distance:.2f} > tolerance {translation_tolerance}).')


        if len(filtered_centers) == 0:
            print('!! [ERROR] No valid face poses after filtering.')
            return None
        
        # Average the valid poses
        final_translation = np.mean(filtered_centers, axis=0)

        # average rotation (using Scipy for stable Quaternion averaging)
        rotations_scipy = R.from_matrix(filtered_rotations)
        final_rotation = rotations_scipy.mean().as_matrix()

        # Construct the final 4x4 transformation matrix
        final_RT = np.eye(4)
        final_RT[:3, :3] = final_rotation
        final_RT[:3, 3] = final_translation
        return final_RT
    
    def run(self, panoramas, sequence_id, save_results=False, output_dir=None, image_names=None):
        """
        Main function to estimate the transformation between two panoramas.
        """

        all_combined_faces = []
        face_keys_list = []
        face_counts = []

        # Step 1: Extract cubemap faces from All panoramas
        for i, pano in enumerate(panoramas):
            name = image_names[i] if image_names else None
            faces = self.extract_horizontal_cubemap_faces(pano, sequence_id, output_dir, name, save_results)
            
            all_combined_faces.extend(list(faces.values()))
            face_keys_list.append(list(faces.keys()))
            face_counts.append(len(faces))

        if self.cfg['use_mapanything']:
            predicted_poses = self.inference_mapanything(all_combined_faces)
        else:
            predicted_poses = self.inference_depthanything(all_combined_faces)
        
        # Dynamically split the predicted poses back into dictionaries
        pano_poses = []
        start_idx = 0
        for i, count in enumerate(face_counts):
            poses_dict = dict(zip(face_keys_list[i], predicted_poses[start_idx:start_idx + count]))
            # Compute global panorama pose for this specific panorama
            pano_pose = self.compute_panorama_pose(poses_dict, translation_tolerance=self.translation_tolerance)
            pano_poses.append(pano_pose)
            start_idx += count

        # Compute relative transformations mapping all poses to the FIRST panorama's coordinate space
        # Assuming panoramas[0] is our global origin.
        relative_tensors = []
        base_pose = pano_poses[0]

        if base_pose is None:
            print("!! [ERROR] Could not compute relative poses; base panorama (index 0) has missing valid centers.")
            return [None] * len(panoramas)

        base_pose_inv = np.linalg.inv(base_pose)

        for i, pose in enumerate(pano_poses):
            if pose is None:
                print(f"!! [WARNING] Could not compute pose for panorama index {i}.")
                relative_tensors.append(None)
                continue
            
            # Rt from Panorama 0 to Panorama i
            Rt_i_to_0 = base_pose_inv @ pose
            relative_tensors.append(torch.tensor(Rt_i_to_0, dtype=torch.float32))
            
        return relative_tensors
    
    def plot_camera_frustum(self, ax, Rt_matrix, name, scale=0.5):
        """
        Plots a 3D camera frustum, swapping CV coordinates to Matplotlib coordinates
        so the Z-axis correctly represents forward depth.
        """
        center = Rt_matrix[:3, 3]
        rotation = Rt_matrix[:3, :3]
        
        # Local CV axes (X=Right, Y=Down, Z=Forward)
        x_axis = rotation[:, 0] 
        y_axis = rotation[:, 1] 
        z_axis = rotation[:, 2] 
        
        ip_center = center + z_axis * scale
        hw = scale * 0.5 
        
        # Calculate 4 corners in CV space 
        # (Note: Because CV Y is Down, -Y moves the corner to the "Top" of the camera)
        tl = ip_center - hw * x_axis - hw * y_axis 
        tr = ip_center + hw * x_axis - hw * y_axis 
        br = ip_center + hw * x_axis + hw * y_axis 
        bl = ip_center - hw * x_axis + hw * y_axis 
        up_point = ip_center - y_axis * (scale * 0.5) 

        # --- THE AXIS SWAP ---
        # Helper to convert [X(Right), Y(Down), Z(Forward)] -> [X(Right), Y(Forward), Z(Up)]
        def to_mpl(pt):
            return np.array([pt[0], pt[2], -pt[1]])

        mpl_center = to_mpl(center)
        mpl_corners = [to_mpl(tl), to_mpl(tr), to_mpl(br), to_mpl(bl)]
        mpl_up = to_mpl(up_point)

        # 1. Plot the center point (Lens)
        ax.scatter(*mpl_center, color='black', s=30)
        ax.text(mpl_center[0], mpl_center[1], mpl_center[2] + 0.1, f' {name}', color='black', fontweight='bold')
        
        # 2. Plot the pyramid lines
        for corner in mpl_corners:
            ax.plot([mpl_center[0], corner[0]], [mpl_center[1], corner[1]], [mpl_center[2], corner[2]], color='blue', alpha=0.6)
            
        # 3. Plot the image plane boundary (Red screen)
        plane_x = [c[0] for c in mpl_corners] + [mpl_corners[0][0]]
        plane_y = [c[1] for c in mpl_corners] + [mpl_corners[0][1]]
        plane_z = [c[2] for c in mpl_corners] + [mpl_corners[0][2]]
        ax.plot(plane_x, plane_y, plane_z, color='red', linewidth=2)
        
        # 4. Plot green antenna showing "Up"
        mpl_ip_center = to_mpl(ip_center)
        ax.plot([mpl_ip_center[0], mpl_up[0]], [mpl_ip_center[1], mpl_up[1]], [mpl_ip_center[2], mpl_up[2]], color='green', linewidth=3)
    

    def verify_poses_visually(self, rt_matrices, names, sequence_id, image_name="trajectory", output_dir=None):
        """
        Creates an Isometric 3D plot of N Panoramas using frustums with dynamically scaling sizes.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        valid_matrices = []
        valid_names = []
        mpl_centers = []
        
        # ==========================================
        # PASS 1: Collect valid centers to find scale
        # ==========================================
        for rt, name in zip(rt_matrices, names):
            if rt is not None:
                # Basic sanity check to avoid Matplotlib stretching to infinity
                if np.linalg.norm(rt[:3, 3]) > 1000:
                    print(f"[WARNING] Skipping plot for {name}: Translation exceeds sensible limits.")
                    continue
                
                # # Check for mathematically distorted rotation matrices
                # if not np.isclose(np.linalg.det(rt[:3, :3]), 1.0, atol=0.15):
                #     print(f"[WARNING] Skipping plot for {name}: Distorted rotation matrix.")
                #     continue

                valid_matrices.append(rt)
                valid_names.append(name)
                # Store the Matplotlib-converted center [X, Z, -Y]
                mpl_centers.append(np.array([rt[0,3], rt[2,3], -rt[1,3]]))
        
        if not mpl_centers:
            print(f"[WARNING] No valid poses to plot for {image_name}.")
            plt.close(fig)
            return

        # ==========================================
        # DYNAMIC SCALE CALCULATION
        # ==========================================
        if len(mpl_centers) > 1:
            # Calculate distances between consecutive camera steps
            step_distances = [np.linalg.norm(mpl_centers[i] - mpl_centers[i-1]) for i in range(1, len(mpl_centers))]
            
            # Make the camera frustum roughly 40% of the median step size
            # Using median prevents a single massive outlier step from inflating the cameras
            dynamic_scale = np.median(step_distances) * 0.3
            
            # Set a sensible floor just in case the cameras didn't move at all
            dynamic_scale = max(dynamic_scale, 0.1) 
        else:
            dynamic_scale = 1.0  # Fallback if there is only 1 camera
            
        print(f"[INFO] Plotting {image_name} with dynamic camera scale: {dynamic_scale:.2f}")

        # ==========================================
        # PASS 2: Actually plot the data
        # ==========================================
        for rt, name in zip(valid_matrices, valid_names):
            self.plot_camera_frustum(ax, rt, name, scale=dynamic_scale)
        
        # Draw dotted lines connecting the trajectory sequentially
        for i in range(len(mpl_centers) - 1):
            pt1 = mpl_centers[i]
            pt2 = mpl_centers[i+1]
            ax.plot([pt1[0], pt2[0]], 
                    [pt1[1], pt2[1]], 
                    [pt1[2], pt2[2]], color='gray', linestyle='--', linewidth=2)
        
        # Labels and formatting
        ax.set_xlabel('X axis (Right)')
        ax.set_ylabel('Y axis (Forward / Depth)')
        ax.set_zlabel('Z axis (Up)')
        ax.set_title(f'Estimated Camera Trajectory: {image_name}')
        
        # Set boundaries based on the max extents
        max_dist = np.max(np.abs(np.array(mpl_centers)))
        limit = max(max_dist + dynamic_scale, 1.5) 
        
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])

        try:
            ax.set_box_aspect([1,1,1]) 
        except AttributeError:
            pass 

        ax.view_init(elev=20, azim=-45)

        if output_dir is None:
            output_dir = os.path.join(self.cfg['output_dir_ext'], sequence_id, "trajectory")

        save_path = os.path.join(output_dir, f'{image_name}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig) 
        print(f'[INFO] Saved 3D trajectory plot to {save_path}')


    def print_human_readable_pose(self, Rt_matrix, name="Relative Pose"):
        """
        Decomposes a 4x4 transformation matrix into human-readable 
        distance, directional shifts, and rotation angles (Yaw, Pitch, Roll).
        """
        if torch.is_tensor(Rt_matrix):
            Rt_matrix = Rt_matrix.cpu().numpy()

        # Extract Translation (3x1 vector) and Rotation (3x3 matrix)
        T = Rt_matrix[:3, 3]
        R_mat = Rt_matrix[:3, :3]
        
        # 1. Calculate Total Distance
        distance = np.linalg.norm(T)

        try:
            # Check if the matrix is biologically possible before passing to SciPy
            det = np.linalg.det(R_mat)
            if not np.isclose(det, 1.0, atol=0.1):
                print(f"\n!! [WARNING] The solver returned a heavily distorted matrix (determinant={det:.2f}).")
            
            # This will throw a ValueError if R_mat is mathematically invalid
            yaw, pitch, roll = R.from_matrix(R_mat).as_euler('yxz', degrees=True)
            
            print(f"\n==========================================")
            print(f" SUMMARY: {name}")
            print(f"==========================================")
            print(f" TRANSLATION")
            print(f"  Total Distance : {distance:.2f} units")
            print(f"  Right / Left   (X) : {T[0]:+8.2f} units")
            print(f"  Down / Up      (Y) : {T[1]:+8.2f} units")
            print(f"  Forward / Back (Z) : {T[2]:+8.2f} units")
            print(f"")
            print(f" ROTATION")
            print(f"  Yaw   (Turn L/R)   : {yaw:+8.2f}°")
            print(f"  Pitch (Look U/D)   : {pitch:+8.2f}°")
            print(f"  Roll  (Tilt Head)  : {roll:+8.2f}°")
            print(f"==========================================\n")
                
        except ValueError as e:
            # Catch the SciPy crash and print a clean error instead of stopping the script
            print(f"\n{'='*42}\n POSE ESTIMATION FAILED: {name}\n{'='*42}")
            print(f" !! [ERROR] The feature matcher failed to find good overlap.")
            print(f" !! SciPy rejected the geometry: {e}")
            print(f"{'='*42}\n")