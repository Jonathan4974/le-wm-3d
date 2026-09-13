import os
import glob
import numpy as np
import open3d as o3d
import imageio

def render_ply_sequence_to_mp4(ply_dir, output_mp4, width=1920, height=1080, fps=10):
    """
    Renders a sequence of sequential .ply files into a high-quality .mp4 video 
    using Open3D's OffscreenRenderer (ideal for headless servers).
    """
    # 1. Collect and sort all PLY files chronologically
    ply_files = sorted(glob.glob(os.path.join(ply_dir, "*.ply")))
    if not ply_files:
        print(f"Error: No .ply files found in directory '{ply_dir}'.")
        return

    # 2. Initialize the offscreen renderer
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    
    # 3. Configure the material record
    material = o3d.visualization.rendering.MaterialRecord()
    # Use 'defaultLit' if your point clouds contain RGB/normals, or 'defaultUnlit' for raw coordinates
    material.shader = "defaultLit" 

    # Initialize the video writer with the H.264 codec
    video_writer = imageio.get_writer(output_mp4, fps=fps, codec='libx264')
    
    # 4. Establish a unified camera view tracking anchor to prevent jittering
    # Read the first frame to lock the camera's focus on the center of the scene
    pcd_first = o3d.io.read_point_cloud(ply_files[0])
    center = pcd_first.get_center()
    
    print(f"Start rendering 4D video ({len(ply_files)} frames)...")
    
    for idx, ply_path in enumerate(ply_files):
        # Read the current point cloud frame
        pcd = o3d.io.read_point_cloud(ply_path)
        
        # Add the geometry object to the offscreen rendering scene
        render.scene.add_geometry(f"frame_{idx}", pcd, material)
        
        # Configure the camera matrix
        # Arguments: fov, center, eye (camera position), up_vector
        # Adjust center + [0, -3, 3] to tune your distance and viewpoint angle
        render.setup_camera(60.0, center, center + [0, -3, 3], [0, 0, -1])
        
        # Render the scene buffer to an image object and cast to a NumPy array
        img = render.render_to_image()
        img_np = np.asarray(img)
        
        # Append the rendered frame into the video stream
        video_writer.append_data(img_np)
        
        # Crucial step: Remove the geometry to prevent frames from overlapping spatially
        render.scene.remove_geometry(f"frame_{idx}")
        
    # Safely flush and close the file stream
    video_writer.close()
    print(f"Successfully saved 4D video to: {output_mp4}")

    
if __name__ == "__main__":
    # Example usage: Render .ply files from a specific directory into an .mp4 video
    ply_directory = "/home/student/users/Yangjie_workspace/data_visualization/processed/da3_streaming_skip_1_chunk_5/pcd"
    output_video_path = "/home/student/users/Yangjie_workspace/data_visualization/processed/da3_streaming_skip_1_chunk_5/pcd/pcd_episode_4_streaming_chunk_5_conf_1.mp4"
    
    render_ply_sequence_to_mp4(ply_directory, output_video_path, width=1920, height=1080, fps=10)