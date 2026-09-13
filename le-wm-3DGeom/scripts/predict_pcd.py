import gc
import glob
import os
# import time
# from threading import Thread
# from omegaconf import OmegaConf
# from hydra import initialize, compose
import torch
import numpy as np
import open3d as o3d
from einops import rearrange
# import stable_worldmodel as swm
from depth_anything_3.api import DepthAnything3 # Ensure 'pip install -e .' was successful
# from depth_anything_3.specs import Prediction
from npz_to_4d_video import npz_to_pcd_video
from npz_to_depth_video import npz_to_depth_video
from npz_to_rotating_pcd_video import npz_to_rotating_pcd_video
from utils import (
    load_images_from_episode, 
    _depths_to_world_points_with_colors, 
    export_to_npz, 
    load_front_view_images_from_folder, 
    load_multi_view_images_from_folder,
    batch_resize_area,
)
from PIL import Image


def build_global_pcd(
        # config_path, 
        output_path, 
        DA_model="DA3-Large", # DA3Nested-Giant-Large can not run with frameskip 5, use DA3-Large is doable
        # model=None,
        conf_thresh=0.0, 
        ray_pose=False,
        world_pose=True,
        epi_id = 4, 
        per_frame=False,
        ):
    """
    Extracts an episode from OGBench, processes it with DA3, 
    and fuses frames into a global point cloud using Open3D.
    """
    # 1. Initialize DA3 Model 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnything3.from_pretrained(f"depth-anything/{DA_model}").to(device)

    # # 2. Load OGBench Data
    # with initialize(version_base=None, config_path=config_path):
        
    #     cfg = compose(config_name="lewm")

    # dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    # print("Dataset 'action' shape:", dataset[0]['action'].shape)
    # print("Dataset 'observation' shape:", dataset[0]['observation'].shape)
    # print("Dataset 'proprio' shape:", dataset[0]['proprio'].shape)

    # # single frame
    # images = dataset[35]['pixels'][3]  # (C, H, W)
    # images = images.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    # img_list = [images]
    # print("Loaded images shape:", images.shape)
    # pcd_path = os.path.join(output_path, "pcd_frame_354")

    # # frames from one clip
    # images = dataset[4]['pixels']  # (T, C, H, W)
    # images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    # img_list = [img for img in images]
    # print("Loaded images shape:", images.shape)
    # print("img_list length:", len(img_list))
    # pcd_path = os.path.join(output_path, "pcd_clip_4")

    #####################################################
    # inference on all frames from one episode and export point clouds as npz files
    #####################################################

    # # frames from one episode
    # images = dataset.load_episode(epi_id)["pixels"]  # (T, C, H, W)
    # print("Loaded images shape:", images.shape)
    # images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)


    # images = load_images_from_episode(epi_id)
    # print("Loaded images shape:", images.shape)

    # conf_str = str(conf_thresh).replace('.', '_')
    # pcd_folder_path = os.path.join(output_path, f"pcd_episode_{epi_id}_{'per_frame' if per_frame else 'per_episode'}_{DA_model}_{conf_str}_world{world_pose}_ray{ray_pose}")

    ######################################
    # # load single view images from folder
    # images = load_front_view_images_from_folder("/home/student/users/Yangjie_workspace/data_link/ogbench_views/front_pixels_dataset/episode_00")
    # print("Loaded images shape:", images.shape)
    # pcd_folder_path = os.path.join(output_path, f"pcd_front_pixels_dataset_{DA_model}_{conf_thresh}_world{world_pose}_ray{ray_pose}")
    #######################################

    #######################################
    # # load multi-view images from folder
    # multiview_folder = "/home/student/users/Yangjie_workspace/data_link/ogbench_views/multiview_dataset/episode_00"
    # pcd_folder_path = os.path.join(output_path, f"pcd_multiview_dataset_{DA_model}_{conf_thresh}_world{world_pose}_ray{ray_pose}")
    # front_images, left_images, right_images = load_multi_view_images_from_folder(multiview_folder)
    # images = [[l, f, r] for f, l, r in zip(front_images, left_images, right_images)]
    # print("lodaded frames number:", len(images))
    #######################################
    # load multi-view images with camera parameters from generated .npz dataset
    conf_str = str(conf_thresh).replace('.', '_')
    pcd_folder_path = os.path.join(output_path, f"pcd_episode_00_gen_front_zoomed_224_with_cam_no_align_{'per_frame' if per_frame else 'per_episode'}_{DA_model}_{conf_str}_world{world_pose}_ray{ray_pose}")
    dataset = np.load("/home/student/users/Public_workspace/le-wm-3DGeom/models/ogbench/visualizations/outputs/multiview_data.npz")
    multiview_images = dataset['image']  # (T, N_views, H, W, C)
    multiview_intrinsics = dataset['intrinsics']  # (T, N_views, 3, 3)
    multiview_extrinsics = dataset['extrinsics']  # (T, N_views, 4, 4)

    # # for generated multiview input
    # multiview_idx = [0, 2, 3, 4, 5]
    # images = multiview_images[:, multiview_idx, ...] # list of frames, each frame is a list of multi-view images
    # intrinsics = multiview_intrinsics[:, multiview_idx, ...]
    # extrinsics = multiview_extrinsics[:, multiview_idx, ...]

    # # for lewm front_pixels input
    # images = multiview_images[:, [1], ...] # take the lewm front view images for inference (T, 1, H, W, C)
    # intrinsics = multiview_intrinsics[:, [1], ...] # (T, 1, 3, 3)
    # extrinsics = multiview_extrinsics[:, [1], ...] # (T, 1, 4, 4)

    # for generated front_zoomed input
    images = multiview_images[:, [0], ...] # take the generated zoomed front view images for inference
    intrinsics = multiview_intrinsics[:, [0], ...]
    extrinsics = multiview_extrinsics[:, [0], ...]

    # contronl the size of DA3 input resolution
    target_res = 224

    if per_frame:
        # Process each frame individually and save separate point clouds
        for i, (img_array, intr_array, ext_array) in enumerate(zip(images, intrinsics, extrinsics)):
            # img_list = [images[i]] # list of 1 view for the same frame
            # img_list = images[i] # list of 3 views for the same frame

            img_list = list(img_array) # list of 3 views for the same frame, each view is (H, W, C)

            pcd_path = os.path.join(pcd_folder_path, f"frame_{i}.npz") # the file contain all views of the same frame

            # predict_and_export_depth_and_pcd_to_npz(
            #     img_list=img_list,
            #     export_path=pcd_path,
            #     DA_model=model,
            #     conf_thresh=conf_thresh,
            #     ray_pose=ray_pose,
            #     world_pose=world_pose,
            #     ref_view_strategy="first",
            # )

            prediction = model.inference(
                image=img_list,
                extrinsics=ext_array,             # # for single view per frame inference,we can't provide extr and intr by default,
                intrinsics=intr_array,
                align_to_input_ext_scale = True, # i.e. we cannot compute the scale, return only relative depth or not aligned metric depth (can be close to the metric depth if we use DA3-Nested)
                use_ray_pose=ray_pose,
                process_res=target_res,
                # export_dir=export_path, # comment out export_dir to avoid redundant saving since we will call export_to_npz separately after processing the depth and point cloud
                export_format="npz",
                conf_thresh_percentile=conf_thresh,
                num_max_points=500_000,
                show_cameras=True,
                ref_view_strategy = "first",  # "first", "middle", "saddle_balanced", "saddle_sim_range"
            )
            
            # ==============================
            # downsample the depth from 504 to 224 and use the input 224 images for reconstruction
            # ==============================
            # depth_224 = batch_resize_area(prediction.depth, target_size=(224, 224))
            # prediction.processed_images = img_array
            # ==============================

            images_u8 = prediction.processed_images  # (N,H,W,3) uint8

            # replace the predicted cam with the input the cam to compare the result
            prediction.extrinsics = extrinsics[:, 0, ...]
            prediction.intrinsics = intrinsics[:, 0, ...]

            # print(prediction.intrinsics.shape, prediction.extrinsics.shape)

            # Back-project to world coordinates and get colors (world frame)
            points, colors = _depths_to_world_points_with_colors(
                depth=prediction.depth, # prediction.depth for 504, depth_224 for 224
                K=prediction.intrinsics,
                ext_w2c=prediction.extrinsics,  # w2c
                images_u8=images_u8, 
                conf=None,  # prediction.conf,
                conf_thr=0.0,  # No confidence filtering for now
                world_pose=world_pose,
            )

            prediction.aux["pts3d"] = points
            prediction.aux["colors"] = colors

            # ==================================================
            # Just save the first frame's point cloud as PLY/PCD for quick visualization sanity check and metric computing
            # ==================================================
            os.makedirs(os.path.dirname(pcd_path), exist_ok=True)

            if i == 0:  
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)

                o3d.io.write_point_cloud(
                    f"{pcd_folder_path}/mujoco_pcd_frame_0.ply",
                    pcd
                )

                o3d.io.write_point_cloud(
                    f"{pcd_folder_path}/mujoco_pcd_frame_0.pcd",
                    pcd
                )

                print("point cloud of the first frame is saved under", f"{pcd_folder_path}")
            # ===================================================

            # export_to_npz(prediction, export_dir=pcd_path)

            # ===================================================
            # modified export to npz
            # ===================================================

            image = prediction.processed_images  # (N,H,W,3) uint8

            # Build save dict with only non-None values
            save_dict = {
                "image": image,
                "depth": np.round(prediction.depth, 6),
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
            np.savez_compressed(pcd_path, **save_dict)
            # =====================================================
    else:
        # Process all frames together and save separate point cloud
        img_list = [img for img in images]
        # print("image shape:", img_list[0].shape)
        # for idx, img in enumerate(img_list):
        #     img_pil = Image.fromarray(img)
        #     img_pil.save(os.path.join(pcd_folder_path, f"frame_{idx}.png"))
        pcd_path = os.path.join(pcd_folder_path, f"episode_{epi_id}.npz")
        
        predict_and_export_depth_and_pcd_to_npz(
            img_list=img_list,
            export_path=pcd_path,
            DA_model=model,
            conf_thresh=conf_thresh,
            ray_pose=ray_pose,
            world_pose=world_pose,
        )

    print(f"Point cloud prediction and export complete! NPZ saved under: {pcd_folder_path}")

    output_pcd_video = os.path.join(pcd_folder_path, f"pcd_episode_{epi_id}_{'per_frame' if per_frame else 'per_episode'}_{DA_model}_{conf_thresh}_world{world_pose}_ray{ray_pose}.mp4")
    output_depth_video = os.path.join(pcd_folder_path, f"depth_episode_{epi_id}_{'per_frame' if per_frame else 'per_episode'}_{DA_model}_{conf_thresh}_world{world_pose}_ray{ray_pose}.mp4")
    return pcd_folder_path, output_pcd_video, output_depth_video

# =========================
# utilities
# =========================

def predict_and_export_depth_and_pcd_to_npz(
    img_list: list[np.ndarray],
    export_path: str,
    DA_model,
    conf_thresh: float = 0.0,
    ray_pose: bool = False,
    world_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
):
    prediction = DA_model.inference(
            image=img_list,
            use_ray_pose=ray_pose,
            # export_dir=export_path,
            export_format="npz",
            conf_thresh_percentile=conf_thresh,
            num_max_points=500_000,
            show_cameras=True,
            ref_view_strategy = ref_view_strategy,  # "first", "middle", "saddle_balanced", "saddle_sim_range"
        )

    images_u8 = prediction.processed_images  # (N,H,W,3) uint8

    print(prediction.intrinsics.shape, prediction.extrinsics.shape)

    # Back-project to world coordinates and get colors (world frame)
    points, colors = _depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,  # w2c
        images_u8,
        conf=None,  # prediction.conf,
        conf_thr=0.0,  # No confidence filtering for now
        world_pose=world_pose,
    )

    prediction.aux["pts3d"] = points
    prediction.aux["colors"] = colors

    export_to_npz(prediction, export_dir=export_path)
    

if __name__ == "__main__":

    # cfg_path = "../models/le-wm/config/train"
    OUTPUT = "./data_visualization/processed/" 
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    # DA_model="DA3-Large"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = DepthAnything3.from_pretrained(f"depth-anything/{DA_model}").to(device)

    # for ray_pose in [False, True]:
    #     for world_pose in [False, True]:
    #         for per_frame in [False, True]:
    #             for conf_thresh in [0.0, 0.75]:
    #                 pcd_folder_path, output_pcd_video, output_depth_video = build_global_pcd(
    #                     OUTPUT, 
    #                     model=model, 
    #                     # DA_model="DA3-Large",
    #                     conf_thresh=conf_thresh, 
    #                     ray_pose=ray_pose, 
    #                     world_pose=world_pose, 
    #                     epi_id=4, 
    #                     per_frame=per_frame,
    #                 )
    #                 npz_to_pcd_video(pcd_folder_path, output_pcd_video)
    #                 npz_to_depth_video(pcd_folder_path, output_depth_video)

    #                 gc.collect()           
    #                 torch.cuda.empty_cache() 
    # # 1. Trigger the asynchronous point cloud generation pipeline "DA3Nested-Giant-Large"
    pcd_folder_path, output_pcd_video, output_depth_video = build_global_pcd(OUTPUT, DA_model="DA3Nested-Giant-Large", conf_thresh=0.0, ray_pose=False, world_pose=True, epi_id=0, per_frame=True)
    # # 2. File I/O Guard for Async Export
    # # Since export_to_npz is decorated with @async_call, we must block the main thread 
    # # and wait until the background thread completes the disk write operation.
    # print("Detected async file export. Waiting for .npz files to complete writing...")
    # timeout = 30  # Maximum wait time in seconds (npz compression can take time)
    # start_time = time.time()
    
    # while True:
    #     if os.path.exists(pcd_folder_path):
    #         # Gather all npz files inside the target folder
    #         files = [os.path.join(pcd_folder_path, f) for f in os.listdir(pcd_folder_path) if f.endswith('.npz')]
            
    #         if len(files) > 0:
    #             # Target the first file to verify if the file stream is completely closed
    #             size_before = os.path.getsize(files[0])
    #             time.sleep(0.3)  # Wait briefly to let the file buffer flush
    #             size_after = os.path.getsize(files[0])
                
    #             # If the file size remains identical between intervals and is greater than 0,
    #             # it safely indicates that the background thread has finished writing.
    #             if size_before == size_after and size_after > 0:
    #                 print(f"Success! {len(files)} .npz file(s) fully flushed and written to disk.")
    #                 break
                    
    #     # Raise an exception if the file isn't created within the safety window
    #     if time.time() - start_time > timeout:
    #         raise TimeoutError(f"Async saving timeout! Files were not ready in: {pcd_folder_path}")
            
    #     time.sleep(0.5)  # Polling interval to reduce CPU overhead

    # # 3. Proceed to the next dependent task once files are fully ready
    # print(f"Starting 4D video generation from predicted point clouds...")

    ############################################################################################################
    video_fps = 10
    frame_padding = 1

    npz_to_rotating_pcd_video(pcd_folder_path, output_pcd_video, fps=video_fps, repeat_count=frame_padding, width=504, height=504)
    # npz_to_pcd_video(pcd_folder_path, output_pcd_video)
    npz_to_depth_video(pcd_folder_path, output_depth_video, fps=video_fps, repeat_count=frame_padding, width=504, height=504)