import os
from natsort import natsorted
import numpy as np
import cv2
import glob

def npz_to_depth_video(folder_path, output_path, fps=10, repeat_count=1, width=504, height=504):
    # 1. Fetch and sort all npz files
    # Assumes filenames contain frame indices (e.g., frame_1.npz, frame_10.npz)
    files = natsorted(glob.glob(os.path.join(folder_path, "*.npz")))
    if not files:
        print(f"No npz files found in {folder_path}")
        return

    # 2. Initialize Video Writer for a SINGLE window (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Starting depth video generation...")

    for i, file_path in enumerate(files):
        # Load data from individual npz file
        with np.load(file_path) as data:
            # print(data.files)  # Debug: Print available keys in the npz file
            # break
            depth = data['depth']  # Expected shape: (H, W) or (N, H, W)
            print("depth shape", f"{depth.shape}")

            # Robustly handle dimensions (1 frame vs multiple frames inside one npz)
            if depth.ndim == 3:
                depth_frames = depth
            elif depth.ndim == 2:
                depth_frames = np.expand_dims(depth, axis=0)
            elif depth.ndim == 4:
                depth_frames = depth

            # ######################################################
            # # For multiple views, take the front view depth for visualization
            # depth_frames = depth_frames[[1], ...]
            # depth_frames = depth_frames[[0], ...] # if depth: (N_view, H, W)
            # depth_frames = depth_frames[:, 0, ...] # if depth: (T, N_view, H, W)
            # #######################################################

            for raw_depth in depth_frames:
                # 3. ROBUST DEPTH NORMALIZATION & COLOR MAPPING
                # Convert depth data into a viewable 8-bit image safely
                
                # Handle possible NaN or Inf values from model predictions
                clean_depth = np.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)
                # clean_depth = np.clip(clean_depth, 0, 3) # for gt depth
                d_min, d_max = clean_depth.min(), clean_depth.max()
                
                if d_max - d_min > 1e-5:
                    # Normalize to [0, 255] range
                    depth_normalized = ((clean_depth - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
                else:
                    # Fallback for empty or flat depth fields
                    depth_normalized = np.zeros_like(clean_depth, dtype=np.uint8)

                # Apply a beautiful color map (JET is standard for depth viz, colormap outputs BGR)
                # If you prefer classic grayscale, simply use: cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

                # Ensure the frame matches the exact target video dimensions
                frame_final = cv2.resize(depth_colored, (width, height))

                # 4. Write frame with repeat_count for smooth playback / frame-rate padding
                for _ in range(repeat_count):
                    video.write(frame_final)

    # Clean up
    video.release()
    print(f"Rendering complete! Depth Video saved to: {output_path}")


if __name__ == "__main__":

    input_folder = "/home/student/users/Yangjie_workspace/data_visualization/processed/pcd_episode_00_gen_front_zoomed_504_with_cam_no_align_per_frame_DA3Nested-Giant-Large_0_0_worldTrue_rayFalse"
    output_video = "/home/student/users/Yangjie_workspace/data_visualization/processed/pcd_episode_00_gen_front_zoomed_504_with_cam_no_align_per_frame_DA3Nested-Giant-Large_0_0_worldTrue_rayFalse/depth_episode_0_gen_zoomed_front_with_cam_no_align_rotation.mp4"
    
    # os.makedirs(os.path.dirname("/home/student/users/Yangjie_workspace/data_visualization/processed/ground_truth/"), exist_ok=True)
    npz_to_depth_video(input_folder, output_video)