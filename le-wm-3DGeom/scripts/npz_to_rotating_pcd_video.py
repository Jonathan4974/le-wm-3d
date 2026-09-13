import os
# MUST set this before importing open3d to force EGL headless rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"

from utils import _as_homogeneous44
import open3d as o3d
import numpy as np
import cv2
import glob
from natsort import natsorted

def npz_to_rotating_pcd_video(folder_path, output_path, fps=10, repeat_count=1, width=504, height=504):
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
    # Modified: Set dimensions to (width, height) instead of (width * 2, height * 2)
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  

    print(f"Starting rendering in Headless (EGL) mode...")

    # Global rotation angle to maintain continuity between different point cloud frames
    current_angle = 0.0

    for i, file_path in enumerate(files):
        # Load data from individual npz file
        with np.load(file_path) as data:
            print("data keys:", data.files)  # Debug: Check available keys in the npz file
            # pts3d = data['gt_points']   # Expected shape: (254016, 3) or (41, 254016, 3)
            # colors = data['gt_colors'] # Expected shape: (254016, 3) or (41, 254016, 3)
            pts3d = data['pts3d']   # Expected shape: (254016, 3) or (41, 254016, 3)
            colors = data['colors'] # Expected shape: (254016, 3) or (41, 254016, 3)
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
            elif pts3d.ndim == 2:
                pts3d_frames = np.expand_dims(pts3d, axis=0)  # Add a frame dimension to make it (1, 254016, 3)
                colors_frames = np.expand_dims(colors, axis=0)

            if images.ndim == 4:
                images_frames = images 
                intrinsics_frames = intrinsics
                extrinsics_frames = extrinsics
            elif images.ndim == 3:
                images_frames = np.expand_dims(images, axis=0)
                intrinsics_frames = np.expand_dims(intrinsics, axis=0)
                extrinsics_frames = np.expand_dims(extrinsics, axis=0)
            elif images.ndim == 5:
                images_frames = images 
                intrinsics_frames = intrinsics
                extrinsics_frames = extrinsics
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

            # images_frames = images_frames[:, 0, :, :, :]  # take the front view image for visualization
            # intrinsics_frames = intrinsics_frames[:, 0, :, :]
            # extrinsics_frames = extrinsics_frames[:, 0, :, :]
            # #######################################################
            i = 0 
            for pts, cols, raw_img, K, w2c in zip(pts3d_frames, colors_frames, images_frames, intrinsics_frames, extrinsics_frames):
                # Build Open3D PointCloud object
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(cols / 255.0 if cols.max() > 1.1 else cols) # Color normalization (Open3D expects float values in [0, 1])
                center = pcd.get_center()
                # pcd.scale(4.0, center=pcd.get_center()) # Scale the point cloud to better fit the view (adjust factor as needed)

                # Add geometry to the scene
                scene = render.scene
                scene.add_geometry("pcd_frame", pcd, material)

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

                # --- SUB-FRAME RENDERING ---
                for _ in range(repeat_count):
                    # Increment time/angle ONLY for the sweeping motion (Top-Right)
                    current_angle += 0.03  
                    
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

                    # 2. Sweeping animation pivoting around the object center (Originally Top-Right view)
                    sweep_angle = np.sin(current_angle) * (-np.pi / 4)
                    # rot_matrix_sweep = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, 1, 0]) * sweep_angle) # for prediction without mujoco extrinsics
                    rot_matrix_sweep = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, 0, 1]) * sweep_angle) # for prediction with mujoco extrinsics
                    tr_eye = center + rot_matrix_sweep @ (adjusted_eye - center)
                    frame_tr = render_native_view(tr_eye, center, native_up, fov=70)  # Adjust FOV as needed

                    # Modified: Output only the top-right sweeping frame
                    video.write(frame_tr)
                
                # IMPORTANT: Remove geometry after each frame to prevent accumulation/memory bloat
                scene.remove_geometry("pcd_frame")

    # Clean up
    video.release()
    print(f"Rendering complete! PCD Video saved to: {output_path}")

if __name__ == "__main__":

    input_folder = "/home/student/users/Yangjie_workspace/data_visualization/processed/pcd_episode_00_gen_front_zoomed_504_with_cam_no_align_downsample224_per_frame_DA3Nested-Giant-Large_0_0_worldTrue_rayFalse"
    output_video = "/home/student/users/Yangjie_workspace/data_visualization/processed/pcd_episode_00_gen_front_zoomed_504_with_cam_no_align_downsample224_per_frame_DA3Nested-Giant-Large_0_0_worldTrue_rayFalse/pcd_episode_0_per_frame_DA3Nested-Giant-Large_0.0_worldTrue_rayFalse.mp4"
    
    # os.makedirs(os.path.dirname("/home/student/users/Yangjie_workspace/data_visualization/processed/ground_truth/"), exist_ok=True)
    npz_to_rotating_pcd_video(input_folder, output_video)