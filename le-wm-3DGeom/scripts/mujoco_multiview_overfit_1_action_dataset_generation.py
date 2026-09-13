import numpy as np
from PIL import Image
import os
os.environ['MUJOCO_GL'] = 'egl'

import gymnasium
import ogbench
import mujoco
import h5py
import numpy as np

from utils import _depths_to_world_points_with_colors
from npz_to_4d_video import npz_to_pcd_video

NUM_SAMPLES = 1000
NUM_ACTIONS = 5
STEPS_PER_SAMPLE = NUM_ACTIONS + 1
TOTAL_STATES = NUM_SAMPLES * STEPS_PER_SAMPLE
# source_filename = '/home/student/users/Yangjie_workspace/data_link/ogbench/cube_single_expert.h5'
TGT_FILE = f'/home/student/users/Yangjie_workspace/data_link/ogbench/overfit_{NUM_ACTIONS}_action_{NUM_SAMPLES}_samples.h5'


# NUM_VIEWS = 6
# CAMERA_NAMES = ["front_zoomed", "front_pixels", "left", "right", "side", "top"]
NUM_VIEWS = 4
CAMERA_NAMES = ["front_zoomed", "front_pixels", "left", "right"]
IMG_H, IMG_W = 224, 224
ACTION_DIM = 5
# ============================================================
# Helper functions
# ============================================================

def get_camera_intrinsic(model, camera_name, width, height):
    """
    Return camera intrinsic matrix K (3, 3).
    """

    cam_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name,
    )

    fovy = model.cam_fovy[cam_id]

    fy = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    fx = fy

    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ],
    ], dtype=np.float32)

    return K

def get_camera_extrinsic(model, data, camera_name):
    """
    World -> Camera extrinsic matrix (4,4).
    """

    cam_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name,
    )

    pos = data.cam_xpos[cam_id].copy()

    # camera rotation matrix (camera -> world)
    R_c2w = data.cam_xmat[cam_id].reshape(3, 3).copy()

    # world -> camera
    # R_w2c = R_c2w.T
    R_cv_mj = np.diag([1, -1, -1])

    R_w2c = R_cv_mj @ R_c2w.T
    t_w2c = -R_w2c @ pos

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_w2c
    T[:3, 3] = t_w2c

    return T 

def get_body_position(model, data, body_name):
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )
    return data.xpos[body_id].copy()

def get_site_position(model, data, site_name):
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )
    return data.site_xpos[site_id].copy()

def extract_data(env, camera_name):
    # rgb
    rgb = env.unwrapped.render(
        camera=camera_name
    )

    # rgb_views.append(rgb)

    # H, W = rgb.shape[:2]

    # metric depth converted from z-buffer
    depth = env.unwrapped.render(
        camera=camera_name,
        depth=True,
    )

        # depth_views.append(depth)

        # # segmentation mask
        # # img: (H, W, 2), uint32
        # # 2-channel (object ID, object type) pair
        # # for the object ID, -1 is background, 0 is table, 62 is the cube,and positive integers are different objects
        # # for the object type, -1 is background, 5 is scene
        # seg = env.unwrapped.render(
        #     camera=camera_name,
        #     segmentation=True,
        # )

        # seg_views.append(seg)

        # K = get_camera_intrinsic(
        #     model,
        #     camera_name,
        #     W,
        #     H,
        # )

        # T = get_camera_extrinsic(
        #     model,
        #     data,
        #     camera_name,
        # )

        # camera_intrinsics.append(K)
        # camera_extrinsics.append(T)

    # rgb_views = np.stack(rgb_views)
    # depth_views = np.stack(depth_views)
    # seg_views = np.stack(seg_views)

    # camera_intrinsics = np.stack(camera_intrinsics)
    # camera_extrinsics = np.stack(camera_extrinsics)

    # # Back-project to world coordinates and get colors (world frame)
    # points, colors = _depths_to_world_points_with_colors(
    #     depth_views,
    #     camera_intrinsics,  # K
    #     camera_extrinsics,  # w2c
    #     rgb_views,
    #     conf=None,
    #     conf_thr=0.0,  # No confidence filtering for now
    #     world_pose=True,
    # )

    # pts3d_views = np.stack(points)
    # color_views = np.stack(colors)

    # print("rgb_views:", rgb_views.shape)
    # print("depth_views:", depth_views.shape)
    # print("seg_views:", seg_views.shape)
    # print("camera_intrinsics:", camera_intrinsics.shape)
    # print("camera_extrinsics:", camera_extrinsics.shape)
    # print("pts3d_views:", pts3d_views.shape)
    # print("color_views:", color_views.shape)
    return rgb, depth

# camera_extrinsics = np.zeros((NUM_VIEWS, 4, 4), dtype=np.float32)
# camera_intrinsics = np.zeros((NUM_VIEWS, 3, 3), dtype=np.float32)

# print(env.unwrapped.model.ncam)
# num_cams = 0
# for i in range(env.unwrapped.model.ncam):
#     print(i, env.unwrapped.model.cam(i).name)
#     num_cams += 1 

# print(f"num_cams: {num_cams}")
# if num_cams != NUM_VIEWS:
#     raise ValueError(f"Expected {NUM_VIEWS} cameras, but found {num_cams}")

# ============================================================
# Example qpos / qvel
# ============================================================
qpos_1 = [-1.89159375, -1.77377254, 1.8977722, -1.69479598, -1.57079633, 2.30020541,
          0., 0., 0., 0., 0., 0.,
          0., 0., 0.30315381, 0.05456338, 0.02, -0.77935584,
          0., 0., 0.62658158]

qvel_1 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

qpos_2 = [-1.85181565e+00, -1.60923590e+00, 2.08617517e+00, -2.06024515e+00,
          -1.57087074e+00, 2.63992110e+00, 1.18416218e-02, 1.40988445e-04,
          1.19935206e-02, -1.19351810e-02, 1.18486283e-02, 1.64493354e-04,
          1.21398305e-02, -1.19013717e-02, 4.58362109e-01, 5.16914823e-03,
          1.99607247e-02, -7.45318162e-01, -1.40061516e-17, 2.58257326e-17,
          6.66708960e-01]

qvel_2 = [-1.72652930e-01, 2.51166234e-01, 3.14683749e-01, -5.33587498e-01,
          -6.13332471e-05,  3.51802877e-01, -8.88692959e-02,  2.76072649e-04,
          -8.74059008e-02,  8.49779745e-02, -8.90307577e-02, -1.67066401e-04,
          -9.01642394e-02,  8.42026650e-02, -5.57628726e-19,  3.14418305e-18,
          5.19950003e-16, -9.97651036e-17,  1.74576422e-16, -3.24291343e-29]

qpos_init = np.array(qpos_1)
qvel_init = np.array(qvel_1)

env = gymnasium.make(
    "visual-cube-single-v0",
    terminate_at_goal=False,
    mode="data_collection",
)
env.reset()

env.unwrapped.set_state(qpos_init, qvel_init)
mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

# initial_physics_state = env.unwrapped.data.clone()

datasets = {}

with h5py.File(TGT_FILE, "w") as f:
    for name in CAMERA_NAMES:
        datasets[(name, "rgb")] = f.create_dataset(
            f"pixels_rgb_{name}", 
            shape=(TOTAL_STATES, IMG_H, IMG_W, 3), 
            dtype=np.uint8
        )

        datasets[(name, "depth")] = f.create_dataset(
            f"pixels_depth_{name}", 
            shape=(TOTAL_STATES, IMG_H, IMG_W), 
            dtype=np.float32
        )

    datasets["action"] = f.create_dataset(
        "action", 
        shape=(TOTAL_STATES, ACTION_DIM), 
        dtype=np.float32
    )

    # add ep_idx and ep_len so that we can use swm.data.HDF5Dataset to load this dataset
    # FIXME: the ep_idx and ep_len are not correct, but we don't use them in our code, so it's fine for now
    ep_lengths = np.full((NUM_SAMPLES,), fill_value=STEPS_PER_SAMPLE, dtype=np.int32)
    f.create_dataset("ep_len", data=ep_lengths)

    ep_offsets = np.arange(NUM_SAMPLES, dtype=np.int32) * STEPS_PER_SAMPLE
    f.create_dataset("ep_offset", data=ep_offsets)

    def save_current_timestep(env, idx):
        for camera in CAMERA_NAMES:
            rgb, depth = extract_data(env, camera)
            datasets[(camera, "rgb")][idx] = rgb
            datasets[(camera, "depth")][idx] = depth
        
        nxt_action = env.action_space.sample()
        datasets["action"][idx] = nxt_action
        return nxt_action
    
    global_idx = 0
    for i in range(NUM_SAMPLES):

        env.unwrapped.set_state(qpos_init, qvel_init)
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

        action = save_current_timestep(env, global_idx)
        global_idx += 1

        for step in range(NUM_ACTIONS):
            ob, reward, terminated, truncated, info = env.step(action)

            action = save_current_timestep(env, global_idx)
            global_idx += 1
print("datalength", global_idx)
print("Dataset is generated and saved in", TGT_FILE)
env.close()
# # ============================================================
# # save the data in npz file for testing
# # ============================================================
# save_dict = {
#         "image": rgb_views,
#         "depth": depth_views,
#         "extrinsics": camera_extrinsics,
#         "intrinsics": camera_intrinsics,
#         "pts3d": pts3d_views,
#         "colors": color_views,
#     }
# np.savez("/home/student/users/Yangjie_workspace/outputs/visualizations/multiview_data.npz", **save_dict)
# ============================================================
# save the points in ply file for testing
# ============================================================
# import open3d as o3d

# points = pts3d_views.reshape(-1, 3) 
# colors = color_views.reshape(-1, 3) / 255.0  # normalize to [0, 1] for Open3D

# pcd_main = o3d.geometry.PointCloud()
# pcd_main.points = o3d.utility.Vector3dVector(points)
# pcd_main.colors = o3d.utility.Vector3dVector(colors)

# # add marker points for cube position and tcp position
# cube_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
# cube_marker.compute_vertex_normals()
# cube_marker.paint_uniform_color([0.0, 0.0, 1.0]) # bright blue
# cube_marker.translate(cube_pos)                 

# tcp_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
# tcp_marker.compute_vertex_normals()
# tcp_marker.paint_uniform_color([1.0, 0.0, 0.0]) # bright red
# tcp_marker.translate(gripper_pos)                     

# pcd_marker_1 = cube_marker.sample_points_uniformly(number_of_points=500)
# pcd_marker_2 = tcp_marker.sample_points_uniformly(number_of_points=500)

# # 使用 "+" 号直接将它们融合成一个大点云
# final_pcd = pcd_main + pcd_marker_1 + pcd_marker_2

# output_path = "/home/student/users/Yangjie_workspace/outputs/visualizations/multiview_data_change_ex_in_with_marker.ply"
# o3d.io.write_point_cloud(output_path, final_pcd)

# print(f"point cloud saved to: {output_path}")
# # ============================================================
# # save the images for visualization
# # ============================================================
# save_dir_rgb = "/home/student/users/Public_workspace/rgb_views"
# os.makedirs(save_dir_rgb, exist_ok=True)

# for i, img in enumerate(rgb_views):
#     # img: (H, W, 3), uint8
#     Image.fromarray(img).save(f"{save_dir_rgb}/view_{i}.png")

# save_dir_depth = "/home/student/users/Public_workspace/depth_views"
# os.makedirs(save_dir_depth, exist_ok=True)
# for i, d in enumerate(depth_views):
#     # print(np.isnan(d).any())
#     # print(np.isinf(d).any())
#     # d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

#     # d_min, d_max = d.min(), d.max()

#     # print(f"Depth view {i}: min={d_min}, max={d_max}")
#     # if d_max > d_min:
#     #     d_norm = (d - d_min) / (d_max - d_min)
#     # else:
#     #     d_norm = np.zeros_like(d)

#     # d_img = (d_norm * 255).astype(np.uint8)

#     VIS_NEAR = 0.5
#     VIS_FAR  = 1.5

#     d_vis = np.clip(d, VIS_NEAR, VIS_FAR)
#     d_vis = (d_vis - VIS_NEAR) / (VIS_FAR - VIS_NEAR)

#     d_img = (255*d_vis).astype(np.uint8)

#     Image.fromarray(d_img).save(f"{save_dir_depth}/depth_{i}.png")

# save_dir_seg = "/home/student/users/Yangjie_workspace/seg_views"
# os.makedirs(save_dir_seg, exist_ok=True)

# for i, seg in enumerate(seg_views):
#     # img: (H, W, 2), uint32
#     print("segmentation shape:", seg.shape)
#     print("segmentation dtype:", seg.dtype)

#     # object id in seg[:, :, 0]
#     seg_id = seg[:, :, 0].copy()
#     unique_ids = np.unique(seg_id)
#     print(unique_ids)

#     colors = {}

#     for j in unique_ids:
#         if j == 62:
#             colors[j] = (255, 255, 255)  # id=62 white
#         elif j == -1:
#             colors[j] = (0, 0, 0)  # id=-1 black
#         else:
#             np.random.seed(int(j))
#             colors[j] = tuple(np.random.randint(0, 255, 3))

#     h, w = seg_id.shape
#     rgb = np.zeros((h, w, 3), dtype=np.uint8)

#     for k in unique_ids:
#         rgb[seg_id == k] = colors[k]

#     Image.fromarray(rgb).save(f"{save_dir_seg}/objid_view_62_white_test_{i}.png")

    # # object type in seg[:, :, 1]
    # seg_type = seg[:, :, 1].copy()
    # unique_types = np.unique(seg_type)
    # print(unique_types)

    # colors = {}

    # for j in unique_types:
    #     if j == 0:
    #         colors[j] = (255, 255, 255)  # type=0 white
    #     elif j == -1:
    #         colors[j] = (0, 0, 0)  # type=-1 black
    #     else:
    #         np.random.seed(int(j))
    #         colors[j] = tuple(np.random.randint(0, 255, 3))

    # h, w = seg_type.shape
    # rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # for k in unique_types:
    #     rgb[seg_type == k] = colors[k]
    # # Image.fromarray(rgb).save("seg.png")
    # Image.fromarray(rgb).save(f"{save_dir_seg}/objtype_view__{i}.png")





# with h5py.File(source_filename, 'r') as f_src:
#     with h5py.File(target_filename, 'w') as f_tgt:
        
#         # copy the datasets that we don't want to modify
#         for key in f_src.keys():
#             if key != 'pixels':  # to be replaced
#                 f_src.copy(key, f_tgt)
        
#         # define the shape and dtype for the new datasets based on existing ones
#         total_samples = f_src['qpos'].shape[0]  # number of samples, assuming all relevant keys have the same number of samples

#         # pixels
#         pixels_shape = (total_samples, 3, 224, 224, 3)  # shape for the new multiview pixels dataset
#         pixels_dtype = f_src['pixels'].dtype  # dtype for the new datasets (e.g., uint8)

#         # depth_gt
#         # depth_gt_shape = 
#         # depth_gt_dtype =/home/student/users/Public_workspace/le-wm-3DGeom

#         # cam extrinsics
#         # cam_ex_shape = 
#         # cam_ex_dtype =

#         # cam intrinsics
#         # cam_in_shape =
#         # cam_in_dtype =

#         # create the new datasets in the target file with appropriate shapes and dtypes
#         f_tgt.create_dataset('pixels', shape=pixels_shape, dtype=pixels_dtype, chunks=(1, 3, 224, 224, 3))
#         f_tgt.create_dataset('depth_gt', shape=depth_gt_shape, dtype=depth_gt_dtype, chunks=True)
#         f_tgt.create_dataset('cam_extrinsics', shape=cam_ex_shape, dtype=cam_ex_dtype, chunks=True)
#         f_tgt.create_dataset('cam_intrinsics', shape=cam_in_shape, dtype=cam_in_dtype, chunks=True)
        
#         # add new datasets for depth_gt, cam_extrinsics, cam_intrinsics as needed
#         for i in range(total_samples):
#             qpos = f_src['qpos'][i]
#             qvel = f_src['qvel'][i]
            
#             # generate multiview pixels with ogbench based on qpos and qvel
#             # generation logic here, e.g., using ogbench to render images from multiple views based on qpos and qvel
#             three_views = np.stack([img1, img2, img3], axis=0)
#             f_tgt['pixels'][i] = three_views
            
#             # extract or compute depth_gt, cam_extrinsics, cam_intrinsics based on qpos and qvel
#             # depth_gt = ...
#             # cam_extrinsics = ...
#             # cam_intrinsics = ...
#             f_tgt['depth_gt'][i] = depth_gt
#             f_tgt['cam_extrinsics'][i] = cam_extrinsics
#             f_tgt['cam_intrinsics'][i] = cam_intrinsics
            
#             if (i + 1) % 1000 == 0 or (i + 1) == total_samples:
#                 print(f"read and generated {i + 1}/{total_samples} data...")

# print(f"finished writing to {target_filename}")

# with h5py.File(source_filename, 'r') as f:

#     print("keys:", list(f.keys()))
    
#     qpos = f['qpos']
#     qvel = f['qvel']
#     pixels = f['pixels']
    
#     # 3. 将数据集转换为 NumPy 数组（真正将数据读入内存）
#     # data_array = qpos[:] 
    
#     print("qpos_shape:", qpos.shape)
#     print("qpos_type:", type(qpos))
#     print("qpos_dtype:", qpos.dtype)
#     print("qpos_sample:", qpos[100])  # 打印第一条数据样本
#     print("qvel_shape:", qvel.shape)
#     print("qvel_type:", type(qvel))
#     print("qvel_dtype:", qvel.dtype)
#     print("qvel_sample:", qvel[100])  # 打印第一条数据样本
#     print("pixels_shape:", pixels.shape)
#     print("pixels_type:", type(pixels))
#     print("pixels_dtype:", pixels.dtype)