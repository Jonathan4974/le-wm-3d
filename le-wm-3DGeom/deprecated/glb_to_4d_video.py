import open3d as o3d
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

def create_4d_video_from_glb(input_dir, output_video_path, fps=10, width=1920, height=1080):
    # 1. sort .glb files path
    files = sorted(glob.glob(os.path.join(input_dir, "*.glb")))
    if not files:
        print(f"Do not find .glb files in {input_dir}")
        return

    # 2. setup Open3D offscreen renderer
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    
    # set background color 
    render.scene.set_background([0.9, 0.9, 0.9, 1.0]) 
    
    # set point cloud material
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.point_size = 2.0  # size of points in the point cloud

    # 3. setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Starting to render video with {len(files)} frames...")

    for i, file_path in enumerate(tqdm(files)):
        # read glb file as a triangle mesh, then convert to point cloud
        print(f"Processing file: {file_path}")
        # mesh = o3d.io.read_triangle_mesh(file_path)
        # print(f"Mesh vertices: {len(mesh.vertices)}, vertex colors: {len(mesh.vertex_colors)}")
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = mesh.vertices
        # pcd.colors = mesh.vertex_colors

        pcd = o3d.io.read_point_cloud(file_path, format='gltf')
        print(f"Point cloud points: {len(pcd.points)}, colors: {len(pcd.colors)}")

        # if the point cloud does not have colors, paint it with a uniform color
        if not pcd.has_colors():
            pcd.paint_uniform_color([0.5, 0.5, 0.5])

        # add point cloud to the scene
        obj_name = "pcd_frame"
        render.scene.add_geometry(obj_name, pcd, mat)
        
        # set camera for the first frame (you can customize this for better views)
        if i == 0:
            center = pcd.get_center()
            # set camera parameters: fov, center, eye, up
            render.setup_camera(60.0, center, center + [0, 0, 5], [0, -1, 0])
        
        # render the scene to an image
        img = render.render_to_image()
        
        # convert Open3D image to numpy array and then to BGR format for OpenCV
        img_np = np.asarray(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # write the frame to the video
        video_writer.write(img_bgr)
        
        # remove the point cloud from the scene before the next iteration
        render.scene.remove_geometry(obj_name)

    video_writer.release()
    print(f"Video saved to: {output_video_path}")

if __name__ == "__main__":
    INPUT_FOLDER = "/home/student/users/Yangjie_workspace/data_visualization/processed/pcd_frame_35_glb" 
    OUTPUT_NAME = "/home/student/users/Yangjie_workspace/data_visualization/processed/pcd_frame_35_glb/4d_pointcloud_output.mp4"
    
    create_4d_video_from_glb(INPUT_FOLDER, OUTPUT_NAME)
