import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm

def render_npz_to_videos(npz_path, output_dir="output_videos", fps=10):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load npz and extract the first data array
    data_dict = np.load(npz_path)
    keys = list(data_dict.keys())
    if not keys:
        raise ValueError("The npz file is empty!")
    video_data = data_dict["image"]
    
    # Validate tensor shape (Frames, Views, Height, Width, Channels)
    if len(video_data.shape) != 5 or video_data.shape[1] != 6:
        raise ValueError(f"Expected shape (N, 6, H, W, 3), got {video_data.shape}")
        
    num_frames, num_views, height, width, _ = video_data.shape
    view_names = data_dict["camera_names"]
    
    # Render video for each view dimension
    for view_idx in range(num_views):
        view_name = view_names[view_idx]
        output_filename = os.path.join(output_dir, f"{view_name}.mp4")
        
        # Initialize OpenCV VideoWriter using MP4V codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        
        for frame_idx in tqdm(range(num_frames), desc=f"Rendering {view_name}"):
            img = video_data[frame_idx, view_idx]
            
            # Ensure proper data type for OpenCV
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
                
            # Convert color space since OpenCV VideoWriter expects BGR format
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(img_bgr)
            
        video_writer.release()
    print("\nAll videos rendered successfully!")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Render 6-view npz images into separate MP4 videos.")
    # parser.add_argument("--npz_path", type=str, required=True, help="Path to the input .npz file")
    # parser.add_argument("--output_dir", type=str, default="output_videos", help="Directory to save videos")
    # parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    
    # args = parser.parse_args()
    # render_npz_to_videos(args.npz_path, args.output_dir, args.fps)
    npz_path = "/home/student/users/Public_workspace/le-wm-3DGeom/models/ogbench/visualizations/outputs/multiview_data.npz"
    output_dir = "./data_visualization/processed/input_videos"
    render_npz_to_videos(npz_path, output_dir, fps=10)