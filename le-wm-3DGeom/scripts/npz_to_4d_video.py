import os
# MUST set this before importing open3d to force EGL headless rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"

from utils import _as_homogeneous44
import open3d as o3d
import numpy as np
import cv2
import glob
from natsort import natsorted

def npz_to_pcd_video(folder_path, output_path, fps=20, repeat_count=5, width=504, height=504):
    # 1. Fetch and sort all npz files
    # Assumes filenames contain frame indices (e.g., frame_1.npz, frame_10.npz)
    files = natsorted(glob.glob(os.path.join(folder_path, "*.npz")))
    if not files:
        print(f"No npz files found in {folder_path}")
        return

    # 2. Initialize the OffscreenRenderer
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    
    # 3. Setup Material
    # Using 'defaultUnlit' to show original colors without needing estimated normals/lighting
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 2.0  # Adjust based on the density of your 250k points

    # 4. Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height * 2))  # Adjust dimensions as needed

    print(f"Starting rendering in Headless (EGL) mode...")

    # Global rotation angle to maintain continuity between different point cloud frames
    current_angle = 0.0

    for i, file_path in enumerate(files):
        # Load data from individual npz file
        with np.load(file_path) as data:
            print("data keys:", data.files)  # Debug: Check available keys in the npz file
            pts3d = data['gt_points']   # Expected shape: (254016, 3) or (41, 254016, 3)
            colors = data['gt_colors'] # Expected shape: (254016, 3) or (41, 254016, 3)
            images = data['image']   # (H, W, C)
            depth = data['depth']
            intrinsics = data['intrinsics']  # (3, 3) or (41, 3, 3)
            extrinsics = data['extrinsics']  # (3, 4) or (41, 3, 4)
            print("pts3d shape:", pts3d.shape)
            print("colors shape:", colors.shape)
            print("image shape:", images.shape)
            print("depth shape:", depth.shape)
            print("intrinsics shape:", intrinsics.shape)
            print("extrinsics shape:", extrinsics.shape)

            if pts3d.ndim == 3:
                pts3d_frames = pts3d
                colors_frames = colors
                images_frames = images 
                intrinsics_frames = intrinsics
                extrinsics_frames = extrinsics
            elif pts3d.ndim == 2:
                pts3d_frames = np.expand_dims(pts3d, axis=0)  # Add a frame dimension to make it (1, 254016, 3)
                colors_frames = np.expand_dims(colors, axis=0)
                images_frames = np.expand_dims(images, axis=0)
                intrinsics_frames = np.expand_dims(intrinsics, axis=0)
                extrinsics_frames = np.expand_dims(extrinsics, axis=0)

            # ######################################################
            # # For multiple views, take the pts3d and colors from all views and concatenate them into a single point cloud for visualization
            # # take the front view image for visualization since the point cloud is already a fusion of all views
            # pts3d_frames = pts3d_frames.reshape(1, -1, 3)
            # colors_frames = colors_frames.reshape(1, -1, 3)
            # # # use the pts only from front view
            # # pts3d_frames = pts3d_frames[[1], ...]
            # # colors_frames = colors_frames[[1], ...]

            # images_frames = images_frames[[1], ...]  # take the front view image for visualization
            # intrinsics_frames = intrinsics_frames[[1], ...]
            # extrinsics_frames = extrinsics_frames[[1], ...]

            images_frames = images_frames[:, 0, :, :, :]  # take the front view image for visualization
            intrinsics_frames = intrinsics_frames[:, 0, :, :]
            extrinsics_frames = extrinsics_frames[:, 0, :, :]
            # #######################################################

            for pts, cols, raw_img, K, w2c in zip(pts3d_frames, colors_frames, images_frames, intrinsics_frames, extrinsics_frames):

        #         print(w2c.shape)
        #         print(K.shape)
        #         break
        # break
                # Build Open3D PointCloud object
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(cols / 255.0 if cols.max() > 1.1 else cols) # Color normalization (Open3D expects float values in [0, 1])
                center = pcd.get_center()
                # pcd.scale(4.0, center=pcd.get_center()) # Scale the point cloud to better fit the view (adjust factor as needed)

                # Add geometry to the scene
                scene = render.scene
                scene.add_geometry("pcd_frame", pcd, material)

                # --- PREPARE TOP-LEFT FRAME (Original Image) ---
                # Ensure the image is uint8 before converting color space
                if raw_img.dtype != np.uint8:
                    if raw_img.max() <= 1.0:
                        raw_img = (raw_img * 255.0).astype(np.uint8)
                    else:
                        raw_img = raw_img.astype(np.uint8)
                        
                # Convert RGB to BGR for OpenCV
                frame_tl = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
                # Resize to ensure it matches the exact sub-window size (width, height)
                frame_tl = cv2.resize(frame_tl, (width, height))


                # Extract the native translation (Camera Position) and Orientation from Extrinsics
                c2w = np.linalg.inv(_as_homogeneous44(w2c))

                # The 3rd column of c2w (indices [:3, 2]) represents the camera's optical axis (Forward vector)
                # The 1st column (indices [:3, 1]) represents the camera's Up vector
                native_eye = c2w[:3, 3]                    # Actual 3D position of the camera center
                native_forward = c2w[:3, 2]                # Vector pointing straight out of the lens
                native_up = -c2w[:3, 1]                     # Camera's raw 'Up' orientation vector Added a minus sign to flip the orientation right-side up

                # Calculate the exact point the camera was looking at along its optical axis
                # native_look_at = native_eye + native_forward * 2.0 # Look 2 meters ahead
                native_look_at = center

                # Reduce the scaling factor (e.g., from 1.0 to 0.6) to zoom in.
                zoom_factor = 1  # Lower value = closer distance = larger object in frame
                adjusted_eye = center + (native_eye - center) * zoom_factor

                # --- SUB-FRAME RENDERING (The Slow-Mo & Native View Part) ---
                for _ in range(repeat_count):
                    # Increment time/angle ONLY for the sweeping motion (Top-Right)
                    current_angle += 0.02  
                    
                    # 1. NATIVE-BASED RENDER FUNCTION
                    # Set the camera to the exact coordinates calculated during depth estimation
                    def render_native_view(eye_pos, look_target, up_vector, fov):
                        # Apply the calibrated camera position and orientation to Open3D
                        scene.camera.look_at(look_target, eye_pos, up_vector)

                        # reset the FOV for bigger view (smaller FOV = more zoomed in)
                        scene.camera.set_projection(fov, width / height, 0.1, 100.0, o3d.visualization.rendering.Camera.FovType.Vertical)

                        # Render and Capture the frame
                        rendered_img = render.render_to_image()
                        img_np = (np.asarray(rendered_img)).astype(np.uint8)
                        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    # 2. Bottom-Left: 100% Native Camera View (No hardcoded offset, no orbit)
                    # This yields the exact matching perspective as your raw input image
                    frame_bl = render_native_view(adjusted_eye, native_look_at, native_up, fov=50)  # Adjust FOV as needed

                    # 3. Bottom-Right: Fixed 45-degree Offset from the Native Position
                    # To shift 45 degrees while respecting the native setup, we rotate the eye around the object center
                    rot_matrix_45 = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, 1, 0]) * (-np.pi / 4))
                    br_eye = center + rot_matrix_45 @ (adjusted_eye - center)
                    frame_br = render_native_view(br_eye, center, native_up, fov=70)  # Adjust FOV as needed

                    # 4. Top-Right: Sweeping animation pivoting around the object center, 
                    # based out of the native camera elevation/distance setup
                    sweep_angle = np.sin(current_angle) * (-np.pi / 4)
                    rot_matrix_sweep = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, 1, 0]) * sweep_angle)
                    tr_eye = center + rot_matrix_sweep @ (adjusted_eye - center)
                    frame_tr = render_native_view(tr_eye, center, native_up, fov=70)  # Adjust FOV as needed

                # # --- SUB-FRAME RENDERING (The Slow-Mo & Orbit Part) ---
                # for _ in range(repeat_count):
                #     # Increment the angle for a smooth rotation
                #     current_angle += 0.02  # Adjust this to change rotation speed
                    
                #     # Helper function to render the point cloud from a specific angle
                #     def render_angle(angle_rad):
                #         cam_x = center[0] + radius * np.sin(angle_rad)
                #         cam_z = center[2] - radius * np.cos(angle_rad)
                #         cam_y = center[1] #- 0.5 # Slightly elevated or lowered perspective
                        
                #         eye = [cam_x, cam_y, cam_z]
                #         look_at = center
                #         up = [0, -1, 0] # Standard orientation for many CV datasets
                        
                #         scene.camera.look_at(look_at, eye, up)

                #         # Render and Capture
                #         rendered_img = render.render_to_image()
                #         img_np = (np.asarray(rendered_img)).astype(np.uint8)
                #         return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                #     # 1. Top-Right: Sweeping animation from Left 90 to Right 90 (-pi/2 to pi/2)
                #     # Using sin() naturally creates a smooth back-and-forth oscillation
                #     sweep_angle = np.sin(current_angle) * (np.pi / 2)
                #     frame_tr = render_angle(sweep_angle)

                #     # 2. Bottom-Left: Fixed original angle (0 degrees)
                #     frame_bl = render_angle(0.0)

                #     # 3. Bottom-Right: Fixed 45 degrees to the right (+pi/4)
                #     frame_br = render_angle(np.pi / 4)

                    # --- COMPOSE THE 2x2 GRID ---
                    # Stack Top-Left and Top-Right horizontally
                    top_row = np.hstack((frame_tl, frame_tr))
                    # Stack Bottom-Left and Bottom-Right horizontally
                    bottom_row = np.hstack((frame_bl, frame_br))
                    
                    # Stack the two rows vertically
                    final_frame = np.vstack((top_row, bottom_row))

                    video.write(final_frame)
                
                # IMPORTANT: Remove geometry after each frame to prevent accumulation/memory bloat
                scene.remove_geometry("pcd_frame")

    # Clean up
    video.release()
    print(f"Rendering complete! PCD Video saved to: {output_path}")

# Example usage:
# render_headless_pointclouds("./data_folder", "reconstruction_preview.mp4")

if __name__ == "__main__":
    # 加载 npz 文件
    data = np.load('/home/student/users/Public_workspace/le-wm-3DGeom/models/ogbench/visualizations/outputs/multiview_data.npz')

    # 查看文件中包含的所有数组名称（键值）
    print("包含的数组有:", data.files)

    print("数据类型:", type(data))
    print("image shape:", data['image'].shape)  # image shape: (201, 6, 224, 224, 3)
    print("image type:", data['image'].dtype)  # image type: uint8
    print("depth shape:", data['depth'].shape)  # depth shape: (201, 6, 224, 224)
    print("depth type:", data['depth'].dtype)  # depth type: float32
    # print("depth min/max:", data['depth'].min(), data['depth'].max())  # Depth value range
    print("extrinsics shape:", data['extrinsics'].shape)  # extrinsics shape: (201, 6, 4, 4)
    print("extrinsics type:", data['extrinsics'].dtype)  # extrinsics type: float32
    # print("extrinsics:", data['extrinsics'])  # Sample extrinsics matrix
    # print("intrinsics type:", type(data['intrinsics']))  # (3, 3)
    # print("extrinsics mean:", data['extrinsics'].mean(axis=0))  # Average extrinsics across frames
    print("intrinsics shape:", data['intrinsics'].shape)  # intrinsics shape: (201, 6, 3, 3)
    print("intrinsics type:", data['intrinsics'].dtype)  # intrinsics type: float32
    # print("intrinsics:", data['intrinsics'])  # Sample intrinsics matrix
    # print("mean_intrinsics:", data['intrinsics'].mean(axis=0))  # Average intrinsics across frames
    # print("conf shape:", data['conf'].shape)  # (H, W)
    print("pts3d shape:", data['gt_points'].shape)  # pts3d shape: (201, 301056, 3)
    print("colors shape:", data['gt_colors'].shape)  # colors shape: (201, 301056, 3)

    # input_folder = "/home/student/users/Public_workspace/le-wm-3DGeom/models/ogbench/visualizations/outputs/"
    # output_video = "/home/student/users/Yangjie_workspace/outputs/visualizations/multiview_pcd_episode_0_video.mp4"
    # npz_to_pcd_video(input_folder, output_video)