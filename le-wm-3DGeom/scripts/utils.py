import os
from threading import Thread
import h5py
import numpy as np

from depth_anything_3.specs import Prediction
from pycolmap import Image
import stable_worldmodel as swm
from hydra import initialize, compose
from omegaconf import OmegaConf
from PIL import Image as PILImage
import cv2

from natsort import natsorted

def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """
    Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix.
    """
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")

def load_ogbench_dataset(config_path="../models/le-wm/config/train"):
    """
    Loads the OGBench dataset using Hydra configuration and returns an HDF5Dataset instance.
    """
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
        
    with initialize(version_base=None, config_path=config_path):
        
        cfg = compose(config_name="lewm")

    return swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

def load_images_from_episode(epi_id):
    """
    Loads and permutes images of a specific episode from the OGBench dataset.
    Returns a numpy array of shape (T, H, W, C) with uint8 pixel values.
    """
    dataset = load_ogbench_dataset()

    images = dataset.load_episode(epi_id)["pixels"]  # (T, C, H, W)
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return images

def load_front_view_images_from_folder(folder_path):
    """
    Loads and sorts images from a specified folder path.
    Returns a numpy array of shape (T, H, W, C) with uint8 pixel values.
    """
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = natsorted(image_files)

    print(f"Found {len(image_files)} image files in {folder_path}.")  # Debug: Check number of images found

    images = []
    for img_file in image_files[::7]:
        img_path = os.path.join(folder_path, img_file)
        img = PILImage.open(img_path).convert('RGB')
        img = np.array(img)  # (H, W, C) uint8的 numpy 数组
        images.append(img)

    return np.stack(images, axis=0).astype(np.uint8)  # (T, H, W, C)

def load_multi_view_images_from_folder(folder_path):
    """
    Loads and sorts images from a specified folder path.
    Returns a numpy array of shape (T, H, W, C) with uint8 pixel values.
    """
    front_image_files = natsorted([f for f in os.listdir(folder_path) if 'front' in f])
    left_image_files = natsorted([f for f in os.listdir(folder_path) if 'left' in f])
    right_image_files = natsorted([f for f in os.listdir(folder_path) if 'right' in f])

    print(f"Found {len(front_image_files)} image files in {folder_path}.")  # Debug: Check number of images found

    front_images = []
    left_images = []
    right_images = []
    for front, left, right in zip(front_image_files, left_image_files, right_image_files):
        front_img_path = os.path.join(folder_path, front)
        left_img_path = os.path.join(folder_path, left)
        right_img_path = os.path.join(folder_path, right)

        front_img = PILImage.open(front_img_path).convert('RGB')
        left_img = PILImage.open(left_img_path).convert('RGB')
        right_img = PILImage.open(right_img_path).convert('RGB')

        front_img = np.array(front_img)  # (H, W, C) uint8的 numpy 数组
        left_img = np.array(left_img)  # (H, W, C) uint8的 numpy 数组
        right_img = np.array(right_img)  # (H, W, C) uint8的 numpy 数组

        front_images.append(front_img)
        left_images.append(left_img)
        right_images.append(right_img)

    return np.stack(front_images, axis=0).astype(np.uint8), np.stack(left_images, axis=0).astype(np.uint8), np.stack(right_images, axis=0).astype(np.uint8)  # (T, H, W, C)

def _depths_to_world_points_with_colors(
    depth: np.ndarray, # (N,H,W)
    K: np.ndarray, # (N,3,3)
    ext_w2c: np.ndarray, # (N,4,4)
    images_u8: np.ndarray, # (N,H,W,3) uint8
    conf: np.ndarray | None,
    conf_thr: float,
    world_pose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each frame, transform (u,v,1) through K^{-1} to get rays,
    multiply by depth to camera frame, then use (w2c)^{-1} to transform to world frame.
    Simultaneously extract colors.
    """
    
    N, H, W = depth.shape

    # print(f"Depth shape: {depth.shape}")
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W,3)

    pts_all, col_all = [], []

    for i in range(N):
        d = depth[i]  # (H,W)
        valid = np.isfinite(d) & (d > 0)
        if conf is not None:
            valid &= conf[i] >= conf_thr
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(K[i])  # (3,3)
        
        rays = K_inv @ pix[vidx].T  # (3,M)
        Xc = rays * d_flat[vidx][None, :]  # (3,M)

        if world_pose:
            c2w = np.linalg.inv(_as_homogeneous44(ext_w2c[i]))  # (4,4)

            Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))]) # (4,M)
            X = (c2w @ Xc_h)[:3].T.astype(np.float32)  # Xw (M,3)
        else:
            X = Xc.T.astype(np.float32)  # (M,3)

        cols = images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8)  # (M,3)
        
        # # recover shape (H,W,3)
        # H, W = depth[i].shape

        # point_map_hw = np.zeros((H, W, 3), dtype=np.float32)
        # point_map_hw[valid] = X

        # color_map_hw = np.zeros((H, W, 3), dtype=np.uint8)
        # color_map_hw[valid] = cols

        # attention_mask = valid

        pts_all.append(X)
        col_all.append(cols)

    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    # return pts_all, col_all
    return np.concatenate(pts_all, 0), np.concatenate(col_all, 0)

def async_call(fn):
    def wrapper(*args, **kwargs):
        Thread(target=fn, args=args, kwargs=kwargs).start()

    return wrapper

# @async_call
def export_to_npz(
    prediction: Prediction,
    export_dir: str,
):
    # output_file = os.path.join(export_dir, "exports", "npz", "results.npz")
    output_file = export_dir
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Use prediction.processed_images, which is already processed image data
    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required but not available")

    image = prediction.processed_images  # (N,H,W,3) uint8

    # Build save dict with only non-None values
    save_dict = {
        "image": image,
        "da3_depth": np.round(prediction.depth, 6),
        "pts3d": prediction.aux["pts3d"],
        "colors": prediction.aux["colors"],
    }

    if prediction.conf is not None:
        save_dict["conf"] = np.round(prediction.conf, 2)
    if prediction.extrinsics is not None:
        save_dict["extrinsics"] = prediction.extrinsics
    if prediction.intrinsics is not None:
        save_dict["intrinsics"] = prediction.intrinsics

    # aux = {k: np.round(v, 4) for k, v in prediction.aux.items()}
    np.savez_compressed(output_file, **save_dict)


def convert_npz_to_hdf5(npz_path, hdf5_path, compression="gzip"):
    print(f"Loading {npz_path}...")
    # Load the npz file (using mmap_mode allows handling files larger than RAM)
    npz_data = np.load(npz_path, mmap_mode="r")

    print(f"Creating HDF5 file at {hdf5_path}...")
    with h5py.File(hdf5_path, "w") as h5_file:
        for key in npz_data.files:
            data = npz_data[key]
            print(f" -> Writing dataset: '{key}' (shape: {data.shape}, dtype: {data.dtype})")

            # Create a dataset in the HDF5 file
            # Compression is optional but highly recommended for large RL datasets
            h5_file.create_dataset(
                name=key, data=data, compression=compression, chunks=True
            )

    print("Conversion complete!")

def batch_resize_area(data: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """
    Downsample a single-channel image or depth map with shape (N, H, W) 
    using cv2.INTER_AREA interpolation.
    
    :param data: Input numpy array with shape (N, H, W)
    :param target_size: Target size as a tuple (width, height)
    :return: Downsampled numpy array with shape (N, target_size[1], target_size[0])
    """
    # cv2.resize expects dsize as (Width, Height), which is (W, H)
    # Loop through the N dimension and resize each frame individually
    resized_list = [
        cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA) 
        for frame in data
    ]
    
    # Restack the list of arrays back into shape (N, H_new, W_new)
    return np.stack(resized_list, axis=0)