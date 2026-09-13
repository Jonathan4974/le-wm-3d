information
h5 file:
    Keys: ['action', 'control', 'ep_idx', 'ep_len', 'ep_offset', 'id', 'observation', 'pixels', 'prev_qpos', 'prev_qvel', 'privileged_block_0_pos', 'privileged_block_0_quat', 'privileged_block_0_yaw', 'privileged_target_block', 'privileged_target_block_pos', 'privileged_target_block_yaw', 'privileged_target_task', 'proprio_effector_pos', 'proprio_effector_yaw', 'proprio_gripper_contact', 'proprio_gripper_opening', 'proprio_gripper_vel', 'proprio_joint_pos', 'proprio_joint_vel', 'qpos', 'qvel', 'render_time', 'reward', 'step_idx', 'success', 'target', 'terminated', 'time', 'truncated']
    Action length: (2010000, 5)
    Action sample: [ 0.05934808 -0.0612269  -0.14093517 -0.07329748 -0.01652568]
    control shape: (2010000, 7)
    control sample: [0. 0. 0. 0. 0. 0. 0.]
    Pixels shape: (2010000, 224, 224, 3)

DA3 npz output:
    data keys: ['image', 'depth', 'pts3d', 'colors', 'conf', 'extrinsics', 'intrinsics']
    pts3d shape: (41, 254016, 3)
    colors shape: (41, 254016, 3)
    image shape: (41, 504, 504, 3)
    depth shape: (41, 504, 504)
    intrinsics shape: (41, 3, 3)
    extrinsics shape: (41, 3, 4)

DA3 prediction pipeline：
    224x224 input image/intrinsic/extrinsic -> 504x504 depth map/image/intrinsic/extrinsic ->replace 504x504 outputimage with 224x224 original input image / downsample 504x504 depthmap with

DA3 inference ablation:
    1. 224 input, target size 504 without camera parameter -> 504 output depth & image, reconstruction on this 504 output
    2. 224 input, target size 504 with camera parameter -> 504 output depth & image, reconstruction on this 504 output
    3. 224 input, target size 504 with camera parameter -> 504 output depth & image, downsample depth to 224,replace 504 output image with 224 original, reconstruction on this 224 output
    4. 224 input, target size 224 with camera parameter -> 224 output depth & image, reconstruction on this 224 output

    compute icp metric on each of the output

Videos for presentation:
<!-- 1.1. DA3 inference on lewm signle view input without cam. -> input front view video;rotation pcd video;depth map video -->
1.2. DA3 inference on lewm signle view input with cam. -> input front view video;rotation pcd video;depth map video
    ## Findout: for static camera sigle view inference, we can not use Umeyama Sim(3),which needs at leat 3 points (camera pose) and RANSAC which needs at leat 3 frames to Align depth map to input extrinsics.
    but we can still provide extr and intr by:
        1.2.1 comment out the _align_to_input_extrinsics_intrinsics(), and use the predicted extr & intr to reconstruct
        1.2.2 comment out the _align_to_input_extrinsics_intrinsics(), and use the input extr & intr to replace the predicted one (prediction.intrinsics = intrinsics.numpy()) to reconstruct

<!-- 2. generated zoomed front view video -->
<!-- 3.1. DA3 inference on generated multiview input without cam -> rotation pcd video;depth map video of front view -->
<!-- 3.2. DA3 inference on generated multiview input with cam -> rotation pcd video;depth map video of front view -->
<!-- 4. DA3 inference on generated zoomed front input, target size 504,without camera parameter -> rotation pcd video;depth map video -->
5. DA3 inference on generated zoomed front input, target size 504,with camera parameter -> rotation pcd video;depth map video
    <!-- 5.1. comment out the _align_to_input_extrinsics_intrinsics(), and use the predicted extr & intr to reconstruct -->
    <!-- 5.2. comment out the _align_to_input_extrinsics_intrinsics(), and use the input extr & intr to replace the predicted one (prediction.intrinsics = intrinsics.numpy()) to reconstruct -->
<!-- 6. DA3 inference on generated zoomed front input, target size 504,with camera parameter -> downsample depth to 224,replace 504 output image with 224 original, reconstruction on this 224 output -> rotation pcd video;depth map video -->
<!-- 7. DA3 inference on generated zoomed front input, target size 224,with camera parameter -> rotation pcd video;depth map video -->
<!-- 8. rotating gt pcd video from generated ogbench data (adjust the view angle in the video) -->

<!-- 9. segmentation image -->
<!-- 10. gt pcd with marker image -->
<!-- 11. gt depth image -->

<!-- 12. !!!output pcd file for each inference. -->