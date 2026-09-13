import h5py
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image
import cv2

SRC = "/home/student/data/ogbench/triple_views_with_cam_train_1_val_0_episodes.h5"
# MV = "/home/student/users/Public_workspace/data_link/ogbench/DA3_triple_depth_1_val_0_episodes.h5"
# MV = "/home/student/users/Public_workspace/data_link/ogbench/DA3_single_depth_10_val_1_episodes.h5"
# MV = "/home/student/users/Public_workspace/data_link/ogbench/cube_overfit_1_sources_1000_traj_1_actionBlocks_RGB.h5"
# MV = "/home/student/users/Public_workspace/data_link/ogbench/triple_views_with_cam_train_1000_val_100_episodes.h5"
# MV = "/home/student/users/Public_workspace/data_link/ogbench/triple_views_with_cam_train_1_val_0_episodes.h5"
# MV = "/home/student/users/Public_workspace/data_link/ogbench/triple_views_with_cam_train_10_val_1_episodes.h5"
# MV = "/home/student/users/Public_workspace/data_link/ogbench/front_pixels_depth_train_1000_val_100_episodes.h5"
# MV = "/home/student/users/Public_workspace/data_link/ogbench/gt_point_map_1000_val_100_episodes.h5"
MV = "/home/student/users/Public_workspace/data_link/ogbench/DA3_triple_depth_1000_val_100_episodes.h5"
# MV = "/home/student/users/Public_workspace/data_link/ogbench/DA3_single_depth_1000_val_100_episodes.h5"
# MV = "/home/student/users/Public_workspace/data_link/ogbench/front_pixels_NORMALS_train_1000_val_100_episodes.h5"
# MV = "/home/student/users/Public_workspace/data_link/ogbench/front_pixels_depth_train_1000_val_100_episodes.h5"


# --------------------------------------------------
# 1. Dataset structure
# --------------------------------------------------

def inspect_structure():

    with h5py.File(MV, "r") as f:

        print("\n=== DATASET STRUCTURE ===")

        for k in f.keys():
            print(f"{k}: {f[k].shape}")

        # print("\nep_idx:")
        # print(f["ep_idx"][:100])

        # print("\noriginal_episode_ids:")
        # print(f["original_episode_ids"][:100])

        # print("\nep_offset:")
        # print(f["ep_offset"][:100])

        # print("\nep_len:")
        # print(f["ep_len"][:100])

        # print("min:", np.min(f["pixels"])) 
        # print("max:", np.max(f["pixels"]))

        '''da3 triple:
                min: 0.249723
                max: 1.9426256
            da3 single:
                min: 0.33877194
                max: 4.2404356'''

        # print("\npixels_multiview:")
        # print(f["pixels_multiview"].shape)
        

# --------------------------------------------------
# 2. Episode metadata validation
# --------------------------------------------------

def validate_episode_metadata():

    with h5py.File(MV, "r") as f:

        print("\n=== EPISODE METADATA ===")

        ep_offset = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        ep_idx = f["ep_idx"][:]

        for ep in range(len(ep_offset)):

            start = ep_offset[ep]
            length = ep_len[ep]

            unique = set(
                ep_idx[start:start+length]
            )

            print(
                f"episode={ep} "
                f"start={start} "
                f"len={length} "
                f"ids={unique}"
            )


# --------------------------------------------------
# 3. Camera diversity check
# --------------------------------------------------

def validate_camera_diversity(frame_idx=100):

    with h5py.File(MV, "r") as f:

        print("\n=== CAMERA DIFFERENCES ===")

        views = f["pixels_multiview"][frame_idx]

        for i in range(len(views)):
            for j in range(i + 1, len(views)):

                diff = np.mean(
                    np.abs(
                        views[i].astype(np.float32)
                        - views[j].astype(np.float32)
                    )
                )

                print(
                    f"view {i} vs {j}: "
                    f"{diff:.3f}"
                )


# --------------------------------------------------
# 4. Original dataset consistency
# --------------------------------------------------

def validate_original_mapping():

    with h5py.File(SRC, "r") as src, \
         h5py.File(MV, "r") as mv:

        print("\n=== ORIGINAL EPISODE MAPPING ===")

        orig_ep = int(
            mv["original_episode_ids"][0]
        )

        src_start = int(
            src["ep_offset"][orig_ep]
        )

        qpos_close = np.allclose(
            src["qpos"][src_start],
            mv["qpos"][0]
        )

        qvel_close = np.allclose(
            src["qvel"][src_start],
            mv["qvel"][0]
        )

        print("original episode:", orig_ep)
        print("qpos close:", qpos_close)
        print("qvel close:", qvel_close)

        if not qpos_close:

            diff = np.abs(
                src["qpos"][src_start]
                - mv["qpos"][0]
            )

            print(
                "max qpos diff:",
                diff.max()
            )

            print(
                "mean qpos diff:",
                diff.mean()
            )


# --------------------------------------------------
# 5. Save visual sanity image of multiview
# --------------------------------------------------

def save_visualization():
    frame_num = 105
    with h5py.File(MV, "r") as f:

        views = f["pixels_multiview"][frame_num]
        fig, axes = plt.subplots(
            1,
            views.shape[0],
            figsize=(16, 3)
        )

        for i in range(views.shape[0]):

            axes[i].imshow(views[i])
            axes[i].axis("off")
            axes[i].set_title(f"View {i}")

        plt.tight_layout()
        plt.savefig(
            f"multiview_sanity_episode_id_{frame_num}_VAL.png",
            dpi=150
        )

        print(
            "\nSaved multiview_sanity.png"
        )

# --------------------------------------------------
# 5. Save visual sanity image of single view
# --------------------------------------------------

def save_single_rgb():
    frame_num = 105
    with h5py.File(MV, "r") as f:

        img_np = f["pixels"][frame_num]
        img = Image.fromarray(img_np.astype(np.uint8))
        img.save("normal_frame_105.png")
        print(
            "\nSaved rgb_frame_105.png"
        )
# --------------------------------------------------
# 6. Save DA3 depth sanity image
# --------------------------------------------------
def save_depth_with_colorbar():
    frame_num = 105
    with h5py.File(MV, "r") as f:
        depth_map = f["pixels"][frame_num]

        fig, ax = plt.subplots(figsize=(6, 5))  

        im = ax.imshow(
            depth_map,
            cmap="inferno",
            aspect="equal",
            vmin=depth_map.min(),
            vmax=depth_map.max(),
        )

        ax.axis("off")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Depth (meters)", rotation=-90, va="bottom")

        output_name = f"depth_gt_with_scale_frame_{frame_num}.png"
        plt.savefig(output_name, dpi=150, bbox_inches="tight")

        plt.close(fig)
        print(f"Saved with colorbar: {output_name}")

def save_septh_without_bar():
    frame_num = 105
    with h5py.File(MV, "r") as f:

        depth_map = f["pixels"][frame_num]
        depth_2d = np.squeeze(depth_map)

        d_min = depth_2d.min()
        d_max = depth_2d.max()

        depth_norm = ((depth_2d - d_min) / (d_max - d_min) * 255.0).clip(0, 255).astype(np.uint8)

        colored_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        cv2.imwrite(f"depth_DA3_triple_frame_{frame_num}.png", colored_depth)
        print(f"Saved depth map")
# --------------------------------------------------
# 7. Save point cloud
# --------------------------------------------------
def dequantize_pointcloud(q_points):
    """
    Args: q_points: (H, W, 3), uint16
    Returns: points: (H, W, 3), float32
    """

    # Global quantization parameters
    XYZ_MIN = np.array(
        [-1.023597, -0.84242713, 0.001695069],
        dtype=np.float32,
    )

    XYZ_MAX = np.array(
        [0.70306283, 0.8070305, 0.39999232],
        dtype=np.float32,
    )

    XYZ_SCALE = XYZ_MAX - XYZ_MIN

    points = (
        q_points.astype(np.float32) / 65535.0
    ) * XYZ_SCALE + XYZ_MIN

    return points

def save_point_cloud():
    frame_num = 105
    with h5py.File(MV, "r") as f:
        point_map_hw = f["pixels"][frame_num]
        image_hw = f["RGB"][frame_num]
        valid_mask = f["atten_mask"][frame_num]


        M_points = point_map_hw[valid_mask]
        if image_hw.dtype == np.uint8:
            M_colors = image_hw[valid_mask].astype(np.float32) / 255.0
        else:
            M_colors = image_hw[valid_mask]

        pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(dequantize_pointcloud(M_points))
        pcd.points = o3d.utility.Vector3dVector(M_points)

        if len(M_colors) > 0:
            pcd.colors = o3d.utility.Vector3dVector(M_colors)

        output_path = f"point_cloud_frame_{frame_num}.ply"
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved point cloud: {output_path}")


# --------------------------------------------------
# Run all checks
# --------------------------------------------------

inspect_structure()
# validate_episode_metadata()
# validate_camera_diversity()
# validate_original_mapping()
# save_visualization()
# save_depth_with_colorbar()
# save_point_cloud()
# save_single_rgb()
save_septh_without_bar()