import os

import numpy as np
import glob

from utils import _depths_to_world_points_with_colors

def add_pts3d_to_npz(folder_path, world_pose=True):
    # Fetch and sort all npz files
    # Assumes filenames contain frame indices (e.g., frame_000.npz, frame_001.npz)
    files = sorted(glob.glob(os.path.join(folder_path, "*.npz")))
    if not files:
        print(f"No npz files found in {folder_path}")
        return

    for i, file_path in enumerate(files):
        # Load data from individual npz file
        with np.load(file_path, allow_pickle=True) as original_data:
            # Unpack all existing keys into a true dictionary so we can read and modify it safely
            data_dict = {key: original_data[key] for key in original_data.files}

        depth = data_dict['depth'] # Expected shape: (254016, 1) or (41, 254016, 1)
        images = data_dict['image']   # (H, W, C)
        intrinsics = data_dict['intrinsics']  # (3, 3) or (41, 3, 3)
        extrinsics = data_dict['extrinsics']  # (N, 3) or (41, N, 3)
        # pts3d = data_dict.get('pts3d', None)  # Check if pts3d already exists

        # print("depth shape:", depth.shape)
        # print("images shape:", images.shape)
        # print("intrinsics shape:", intrinsics.shape)
        # print("extrinsics shape:", extrinsics.shape)
        # print("pts3d shape:", pts3d.shape if pts3d is not None else "None")

        # break
        if depth.ndim == 3:
            depth_frames = depth
            images_frames = images 
            intrinsics_frames = intrinsics
            extrinsics_frames = extrinsics
        elif depth.ndim == 2:
            depth_frames = depth[np.newaxis, ...] # add a frame dimension to make it (1, 254016, 1)
            images_frames = images[np.newaxis, ...]
            intrinsics_frames = intrinsics[np.newaxis, ...]
            extrinsics_frames = extrinsics[np.newaxis, ...]

        print(f"Processing file {i+1}/{len(files)}: {file_path}")

        points, colors = _depths_to_world_points_with_colors(
            depth_frames,
            intrinsics_frames, # K
            extrinsics_frames,  # w2c
            images_frames,
            conf=None,  # prediction.conf,
            conf_thr=0.0,  # No confidence filtering for now
            world_pose=world_pose,
        )

        data_dict["pts3d"] = points
        data_dict["colors"] = colors
        data_dict["intrinsics"] = intrinsics_frames
        data_dict["extrinsics"] = extrinsics_frames
        data_dict["image"] = images_frames
        data_dict["depth"] = depth_frames


        np.savez_compressed(file_path, **data_dict)  # Overwrite the original npz file with the new data

        print(f"Updated {file_path} with pts3d and colors. ")

        # print(data_dict.keys())  # Debug: Print available keys in the npz file after modification



if __name__ == "__main__":

    input_folder = "/home/student/users/Yangjie_workspace/data_visualization/processed/da3_streaming_skip_1_chunk_5_overlap_1_conf_0_75/results_output"
    add_pts3d_to_npz(input_folder)