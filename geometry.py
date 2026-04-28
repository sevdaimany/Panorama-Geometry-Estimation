import kornia as K
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch3d.transforms import matrix_to_quaternion
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as label_connected_components
from torchvision.ops import masks_to_boxes
import shapely.geometry
import os
import matplotlib.pyplot as plt
import math

def equirect_to_3d(image_coords, depth, W, H):
    # image_coords from get_index_grid are [0, 1]
    theta = (image_coords[..., 0] - 0.5) * 2.0 * math.pi   # Longitude: [-pi, pi]
    phi   = (0.5 - image_coords[..., 1]) * math.pi         # Latitude: [-pi/2, pi/2]
    
    cos_phi = torch.cos(phi)
    X = depth * cos_phi * torch.sin(theta)
    Y = depth * torch.sin(phi)
    Z = depth * cos_phi * torch.cos(theta)
    ones = torch.ones_like(X)
    return torch.stack([X, Y, Z, ones], dim=-1)  # [B, N, 4]


def project_3d_to_equirect(points_3d, W, H):
    X = points_3d[..., 0]
    Y = points_3d[..., 1]
    Z = points_3d[..., 2]
    
    # clamp to prevent division by zero
    depth = torch.sqrt(X**2 + Y**2 + Z**2).clamp(min=1e-6)
    
    theta = torch.atan2(X, Z)                          # [-pi, pi]
    phi   = torch.atan2(Y, torch.sqrt(X**2 + Z**2))    # [-pi/2, pi/2]
    
    # Map back to [0, 1] grid coords
    u_norm = (theta / (2.0 * math.pi)) + 0.5           # [0, 1]
    v_norm = 0.5 - (phi / math.pi)                     # [0, 1]
    
    # important: We MUST multiply by depth here. 
    # PyTorch3D's camera will divide by Z. If we don't multiply by depth now, 
    # the perspective division destroys the spherical coordinates
    u_depth = u_norm * depth
    v_depth = v_norm * depth
    
    return torch.stack([u_depth, v_depth, depth], dim=-1)  # [B, N, 3]


def get_index_grid(height, width, batch=None, type_as=None):
    y, x = torch.linspace(0, 1, height), torch.linspace(0, 1, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    index_grid = rearrange([yy, xx], "two y x -> y x two")
    index_grid[:, :, [0, 1]] = index_grid[:, :, [1, 0]]
    if batch is not None:
        index_grid = repeat(index_grid, "y x two -> b y x two", b=batch)
    if type_as is None:
        return index_grid
    return index_grid.type_as(type_as)


def bbox_iou_single_pair(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Args:
        bbox1 (Tensor): Bounding boxes, shape (4, ).
        bbox2 (Tensor): Bounding boxes, shape (4, ).
    Returns:
        iou (Tensor): IoU.
    """
    # top left
    tl = torch.max(bbox1[:2], bbox2[:2])
    # bottom right
    br = torch.min(bbox1[2:], bbox2[2:])

    area_i = torch.prod(br - tl, dim=0) * (tl < br).all().float()
    area_1 = torch.prod(bbox1[2:] - bbox1[:2], dim=0)
    area_2 = torch.prod(bbox2[2:] - bbox2[:2], dim=0)
    return area_i / (area_1 + area_2 - area_i + 1e-6)


def convert_to_pytorch3d_coordinate_system(points, image_hw=None):
    xy = points[:, :, :2]
    z = points[:, :, 2]
    xy = safe_division(xy, repeat(z, "... -> ... n", n=2))
    xy = 1 - (2 * xy)

    # fix for panorama: Stretch the Normalized Device Coordinates to fit non-square rectangular images (e.g. 1536x640)
    if image_hw is not None:
        h, w = image_hw
        s = min(h, w)
        scale_x = w / s
        scale_y = h / s
        xy[:, :, 0] = xy[:, :, 0] * scale_x
        xy[:, :, 1] = xy[:, :, 1] * scale_y

    xyz = torch.einsum("bni,bn->bni", F.pad(xy, (0, 1), value=1), z)
    return xyz


def convert_to_grid_sample_coordinate_system(points):
    return 2 * points - 1


def safe_division(numerator, denominator):
    sign = torch.sign(denominator)
    sign[sign == 0] = 1
    return numerator / (
        sign
        * torch.maximum(
            torch.abs(denominator),
            1e-5 * torch.ones(denominator.shape).type_as(denominator),
        )
    )


def transform_points(transformation_matrix, points, keep_depth=False):
    """
    Transforms points with a transformation matrix.
    """
    shape = points.shape
    if len(shape) == 2:
        transformation_matrix = transformation_matrix.unsqueeze(0)
        points = points.unsqueeze(0)
    points = F.pad(points, (0, 1), value=1)
    points = torch.einsum("bij,bnj->bni", transformation_matrix, points)
    if keep_depth:
        if len(shape) == 2:
            points = points.squeeze(0)
        return points
    points = safe_division(
        points[:, :, :-1],
        repeat(points[:, :, -1], "... -> ... n", n=points.shape[-1] - 1),
    )
    if len(shape) == 2:
        points = points.squeeze(0)
    return points


def convert_image_coordinates_to_world(image_coords, depth, K_inv, Rt):
    """
    Returns a point cloud of image coords projected into world.
    Note: image_coords must be of shape (b x n x 2), depth must be of shape (b x n)
    Output shape: (b x n x 3)
    """
    # convert to homogenous coordinates
    homogenous_coords = F.pad(image_coords, (0, 1), value=1)
    # multiply by the inverse of camera intrinsics
    camera_ref_coords = torch.einsum("bij,bnj->bni", K_inv, homogenous_coords)
    # introduce depth information
    camera_ref_coords_with_depth = torch.einsum("bni,bn->bni", camera_ref_coords, depth)
    # convert 3d coordinates to 4d homogenous coordinates
    points_in_4d = F.pad(camera_ref_coords_with_depth, (0, 1), value=1)
    # multiply by the inverse of camera extrinsics
    points_in_world = torch.einsum("bij,bnj->bni", Rt, points_in_4d)
    # convert 4d -> 3d
    points_in_world = safe_division(points_in_world[:, :, :3], repeat(points_in_world[:, :, 3], "... -> ... n", n=3))
    return points_in_world


def convert_world_to_image_coordinates(world_points, K_inv, Rt, keep_depth):
    """
    Given 3d world coordinates, convert them into image coordinates.
    Note: world_points must be of shape (b x n x 3)
    Output shape: (b x n x 2) if keep_depth=False, else (b x n x 3)
    """
    b = K_inv.shape[0]
    # compute camera projection matrix
    homogenous_intrinsics = torch.zeros((b, 3, 4)).type_as(K_inv)
    homogenous_intrinsics[:, :, :3] = torch.linalg.inv(K_inv)
    camera_projection_matrix = torch.einsum("bij,bjk->bik", homogenous_intrinsics, torch.linalg.inv(Rt))  # shape: bx3x4
    # project world points onto the image plane
    homogenous_coords = F.pad(world_points, (0, 1), value=1)
    camera_ref_coords_with_depth = torch.einsum("bij,bnj->bni", camera_projection_matrix, homogenous_coords)
    if keep_depth:
        return camera_ref_coords_with_depth
    # convert 3d -> 2d
    image_coords = safe_division(
        camera_ref_coords_with_depth[:, :, :2],
        repeat(camera_ref_coords_with_depth[:, :, 2], "... -> ... n", n=2),
    )
    return image_coords


def estimate_linear_warp(X, Y):
    """
    Given X, Y, estimate a warp (rotation, translation) from X to Y using least squares.
    Note: shape of X, Y must be (b x n x 3)

    Returns: R (shape: b x 3 x 3), T (shape: b x 3).
    For inference: torch.einsum("bij,bnj->bni", R, X) + T
    """
    M = []
    for i in range(len(X)):
        X_ = F.pad(X[i], (0, 1), value=1)
        Y_ = F.pad(Y[i], (0, 1), value=1)
        X_pinv = torch.linalg.pinv(X_)
        M.append(torch.einsum("ij,jk->ki", X_pinv, Y_))
    M = torch.stack(M)
    return M

def setup_canonical_cameras(batch_size, tensor_to_infer_type_from):
    b = batch_size
    K_inv = torch.eye(3).unsqueeze(0).repeat(b, 1, 1).type_as(tensor_to_infer_type_from)
    Rt = torch.eye(4).unsqueeze(0).repeat(b, 1, 1).type_as(tensor_to_infer_type_from)
    return K_inv, Rt

def construct_Rt_matrix(rotation, translation):
    Rt = torch.eye(4).type_as(rotation)
    Rt = Rt.unsqueeze(0).repeat(rotation.shape[0], 1, 1)
    Rt[:, :3, :3] = rotation
    Rt[:, :3, 3] = translation
    return Rt


def get_relative_pose(rotation_before, rotation_after, position_before, position_after, as_single_matrix=False):
    Rt_before = construct_Rt_matrix(rotation_before, position_before)
    Rt_after = construct_Rt_matrix(rotation_after, position_after)
    Rt_1_to_2 = torch.einsum("bij,bjk->bik", torch.linalg.inv(Rt_after), Rt_before)
    Rt_2_to_1 = torch.einsum("bij,bjk->bik", torch.linalg.inv(Rt_before), Rt_after)

    if as_single_matrix:
        return Rt_1_to_2, Rt_2_to_1

    rotation_from_1_to_2 = matrix_to_quaternion(Rt_1_to_2[:, :3, :3])
    rotation_from_2_to_1 = matrix_to_quaternion(Rt_2_to_1[:, :3, :3])
    translation_from_1_to_2 = Rt_1_to_2[:, :3, 3]
    translation_from_2_to_1 = Rt_2_to_1[:, :3, 3]
    return (
        rotation_from_1_to_2,
        rotation_from_2_to_1,
        translation_from_1_to_2,
        translation_from_2_to_1,
    )


def bboxes_to_masks(batch_of_boxes, image_hw):
    type_as = batch_of_boxes[0]
    masks = []
    for boxes in batch_of_boxes:
        boxes = np.array(boxes.cpu()).astype(int)
        mask = np.zeros(image_hw)
        for bbox in boxes:
            mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 1
        masks.append(mask)
    masks = rearrange(masks, "b h w -> b h w 1")
    return K.image_to_tensor(masks).squeeze().type_as(type_as).float()


def remove_invalid_bboxes(bboxes_as_tensor, image_side, add_dummuy_if_empty=True):
    bboxes_as_tensor = torch.clamp(bboxes_as_tensor, 0, image_side - 1)
    bboxes_as_tensor = bboxes_as_tensor[bboxes_as_tensor[:, 0] < bboxes_as_tensor[:, 2]]
    bboxes_as_tensor = bboxes_as_tensor[bboxes_as_tensor[:, 1] < bboxes_as_tensor[:, 3]]
    if len(bboxes_as_tensor) == 0 and add_dummuy_if_empty:
        bboxes_as_tensor = torch.tensor([[0, 0, 1, 1]]).type_as(bboxes_as_tensor)
    return bboxes_as_tensor


def suppress_overlapping_bboxes(bboxes, scores, iou_threshold=0.2):
    convert_to_np = False
    if isinstance(bboxes, np.ndarray):
        bboxes = torch.from_numpy(bboxes)
        scores = torch.from_numpy(scores)
        convert_to_np = True
    # sort bboxes by score
    sorted_indices = torch.argsort(scores, descending=True)
    bboxes = bboxes[sorted_indices]
    scores = scores[sorted_indices]
    # suppress overlapping bboxes
    suppressed_bboxes = []
    suppressed_scores = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        score = scores[i]
        should_suppress = False
        for suppressed_bbox in suppressed_bboxes:
            iou = bbox_iou_single_pair(bbox, suppressed_bbox)
            if iou > iou_threshold:
                should_suppress = True
                break
        if not should_suppress:
            suppressed_bboxes.append(bbox)
            suppressed_scores.append(score)
    bboxes, scores = torch.stack(suppressed_bboxes), torch.stack(suppressed_scores)
    if convert_to_np:
        bboxes, scores = bboxes.cpu().numpy(), scores.cpu().numpy()
    return bboxes, scores


def sample_depth_for_given_points(depth_map, points):
    depth_of_points = F.grid_sample(
        rearrange(depth_map, "b h w -> b 1 h w"),
        rearrange(
            convert_to_grid_sample_coordinate_system(points),
            "b n two -> b 1 n two",
        ),
    )
    return rearrange(depth_of_points, "b 1 1 n -> b n")


def remove_bboxes_with_area_less_than(bboxes_as_np_array, threshold):
    bboxes = []
    for bbox in bboxes_as_np_array:
        if shapely.geometry.box(*bbox[:4]).area < threshold:
            continue
        bboxes.append(bbox)
    bboxes = np.array(bboxes)
    return bboxes


def keep_matching_bboxes(batch, image_index, left_predictions, right_predictions, left_scores, right_scores, confidence_threshold=0.2):
    """
    The idea is that each change bbox must have a corresponding bbox in the other image.
    However, the model itself doesn't enforce this directly. In this post-processing function
    we can filter out bboxes that don't have a corresponding bbox in the other image.
    The process of finding "corresponding bounding boxes" is a bit naive for now.
    """
    left_high_confidence_bboxes = left_predictions[left_scores > confidence_threshold]
    right_high_confidence_bboxes = right_predictions[right_scores > confidence_threshold]
    left_centers = torch.tensor((left_high_confidence_bboxes[:, :2] + left_high_confidence_bboxes[:, 2:4]) / 2) / 224
    right_centers = torch.tensor((right_high_confidence_bboxes[:, :2] + right_high_confidence_bboxes[:, 2:4]) / 2) / 224

    left_centers_in_right = batch["transform_points_1_to_2"](left_centers, image_index) * 224
    right_centers_in_left = batch["transform_points_2_to_1"](right_centers, image_index) * 224
    
    left_bboxes_to_keep = []
    right_bboxes_to_keep = []
    for j, left_in_right in enumerate(left_centers_in_right):
        candidate_bbox = None
        minimum_dist = None
        for right_bbox in right_predictions[:,:4]:
            if shapely.geometry.box(*right_bbox).contains(shapely.geometry.Point(*left_in_right)):
                if candidate_bbox is None:
                    candidate_bbox = right_bbox
                    minimum_dist = torch.norm(left_in_right - torch.tensor((right_bbox[:2] + right_bbox[2:4]) / 2))
                    continue
                dist = torch.norm(left_in_right - torch.tensor((right_bbox[:2] + right_bbox[2:4]) / 2))
                if dist < minimum_dist:
                    candidate_bbox = right_bbox
                    minimum_dist = dist
        if candidate_bbox is not None:
            right_bboxes_to_keep.append(candidate_bbox[:4].tolist())
            left_bboxes_to_keep.append(left_high_confidence_bboxes[j,:4].tolist())

    for j, right_in_left in enumerate(right_centers_in_left):
        candidate_bbox = None
        minimum_dist = None
        for left_bbox in left_predictions[:,:4]:
            if shapely.geometry.box(*left_bbox).contains(shapely.geometry.Point(*right_in_left)):
                if candidate_bbox is None:
                    candidate_bbox = left_bbox
                    minimum_dist = torch.norm(right_in_left - torch.tensor((left_bbox[:2] + left_bbox[2:4]) / 2))
                    continue
                dist = torch.norm(right_in_left - torch.tensor((left_bbox[:2] + left_bbox[2:4]) / 2))
                if dist < minimum_dist:
                    candidate_bbox = left_bbox
                    minimum_dist = dist
                
        if candidate_bbox is not None:
            left_bboxes_to_keep.append(candidate_bbox[:4].tolist())
            right_bboxes_to_keep.append(right_high_confidence_bboxes[j,:4].tolist())

    return np.array(left_bboxes_to_keep), np.array(right_bboxes_to_keep)
from PIL import Image
from PIL.ExifTags import TAGS
from PIL import Image

def get_height_width(path):
    with Image.open(path) as img:
        width, height = img.size
    return height, width


def load_annotations_for_image(image_id: str, annotations_dir: str,scale_factor) -> np.ndarray:
    """
    Charge les annotations ground truth pour une image donnée
    
    Args:
        image_id: ID de l'image (ex: "id_image.png")
        annotations_dir: Dossier contenant les annotations
    
    Returns:
        Array numpy [N, 4] avec les bounding boxes [x1, y1, x2, y2]
    """
    # Le nom du fichier d'annotation
    annotation_file = f"{image_id}.txt"
    annotation_path = os.path.join(annotations_dir, annotation_file)
    
    if not os.path.exists(annotation_path):
        print(f"Annotation non trouvée: {annotation_path}")
        return np.empty((0, 4))
    
    bboxes = []
    try:
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                x1 = float(parts[0]) * scale_factor
                y1 = float(parts[1])* scale_factor
                x2 = float(parts[2])* scale_factor
                y2 = float(parts[3])* scale_factor
                bboxes.append([x1, y1, x2, y2])
    
    except Exception as e:
        print(f" Erreur lors de la lecture de {annotation_path}: {e}")
        return np.empty((0, 4))
    
    return np.array(bboxes) if bboxes else np.empty((0, 4))



def plot_mask_warping_comparison(mask1, mask2, mask1_warped, mask2_warped, 
                                im1_mask, im2_mask,image1_diff,image2_diff,
                                save_dir, batch_idx=0, pair_idx=0,feature_level=None):
    """
    Visualise et sauvegarde les masques avant et après warping

    Args:
        mask1, mask2: masques originaux [b, h, w] ou [b, 1, h, w]
        mask1_warped, mask2_warped: masques warpés [b, 1, h, w]
        visibility1, visibility2: cartes de visibilité [b, 1, h, w]
        im1_mask, im2_mask: masques finaux après traitement [b, 1, h, w]
        save_dir: répertoire de sauvegarde
        batch_idx: index du batch
        pair_idx: index de la paire d'images
    """
    if feature_level is None:
        h, w = mask1.shape[-2:]
        feature_level = f"{h}x{w}"

    # Créer une structure de dossiers organisée par niveau
    level_dir = os.path.join(save_dir, f"level_{feature_level}")
    os.makedirs(level_dir, exist_ok=True)
    # Créer le répertoire s'il n'existe pas

    # Prendre le premier élément du batch
    b_idx = batch_idx

    # Convertir en numpy et gérer les dimensions
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu()
            if tensor.dim() == 4:  # [b, c, h, w]
                tensor = tensor[b_idx, 0]  # Prendre le premier channel
            elif tensor.dim() == 3:  # [b, h, w]
                tensor = tensor[b_idx]
            return tensor.numpy()
        return tensor

    # Conversion des tenseurs
    mask1_np = to_numpy(mask1)
    mask2_np = to_numpy(mask2)
    mask1_warped_np = to_numpy(mask1_warped)
    mask2_warped_np = to_numpy(mask2_warped)
    im1_mask_np = to_numpy(im1_mask)
    im2_mask_np = to_numpy(im2_mask)
    image1_diff = to_numpy(image1_diff)
    image2_diff = to_numpy(image2_diff)

    # Créer la figure avec sous-graphiques
    fig, axes = plt.subplots(4, 2, figsize=(20, 15))
    fig.suptitle(f'Mask Warping Comparison - Batch {batch_idx}, Pair {pair_idx}', fontsize=16)

    # Ligne 1: Masques originaux et warpés
    axes[0, 0].imshow(mask1_np, cmap='gray')
    axes[0, 0].set_title('Mask 1 Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask2_np, cmap='gray')
    axes[0, 1].set_title('Mask 2 Original')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(mask1_warped_np, cmap='gray')
    axes[1, 0].set_title('Mask 1 Warped onto Mask 2')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask2_warped_np, cmap='gray')
    axes[1, 1].set_title('Mask 2 Warped onto Mask 1')
    axes[1, 1].axis('off')

    # Ligne 3: Masques finaux
    axes[2, 0].imshow(im1_mask_np, cmap='hot')
    axes[2, 0].set_title('Final IM1 Mask\n(visibility2 * (mask1 - mask2_warped))')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(im2_mask_np, cmap='hot')
    axes[2, 1].set_title('Final IM2 Mask\n(visibility1 * (mask2 - mask1_warped))')
    axes[2, 1].axis('off')

    axes[3, 0].imshow(image1_diff, cmap='hot')
    axes[3, 0].set_title('diff im2')
    axes[3, 0].axis('off')

    axes[3, 1].imshow(image2_diff, cmap='hot')
    axes[3, 1].set_title('diff im1')
    axes[3, 1].axis('off')

    plt.tight_layout()

    # Sauvegarder la figure
    save_path = os.path.join(level_dir, f'mask_warping_comparison_b{batch_idx}_p{pair_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Mask warping comparison saved to: {save_path}")

    return save_path


def get_exif_focal_length(img):
    try:
        exif_data = img.getexif()
        
        # Accéder aux données EXIF étendues
        if 34665 in exif_data:
            exif_ifd = exif_data.get_ifd(34665)
            
            # Chercher FocalLength directement
            if 37386 in exif_ifd:
                focal = exif_ifd[37386]
                print(f"Focale trouvée: {focal}")
                return float(focal)  # Simple et direct
            
            # Alternative: FocalLengthIn35mmFilm
            if 41989 in exif_ifd:
                focal = exif_ifd[41989]
                print(f"Focale 35mm trouvée: {focal}")
                return float(focal)
        
        return float(1)
        
    except Exception as e:
        print(f"Erreur: {e}")
        return None
