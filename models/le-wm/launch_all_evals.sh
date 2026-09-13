

# # 1C min-max
# python eval.py --config-name=cube_DEPTH_OGB.yaml single_channel=True dynamic_depth=False preprocessing=minmax policy=/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/results_16_07/front_pixels_depth_train_1000_val_100_episodes/20260730_111834/lewm_epoch_10

# # 1C std-mean 
# python eval.py --config-name=cube_DEPTH_OGB.yaml single_channel=True dynamic_depth=False preprocessing=stdmean policy=/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/results_16_07/front_pixels_depth_train_1000_val_100_episodes/20260729_231956/lewm_epoch_10

# # 3C min-max 
# python eval.py --config-name=cube_DEPTH_OGB.yaml single_channel=False dynamic_depth=False preprocessing=minmax policy=/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/results_16_07/front_pixels_depth_train_1000_val_100_episodes/20260720_034623/lewm_epoch_10

# # 3C std-mean
# python eval.py --config-name=cube_DEPTH_OGB.yaml single_channel=False dynamic_depth=False preprocessing=stdmean policy=/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/results_16_07/front_pixels_depth_train_1000_val_100_episodes/20260722_163545/lewm_epoch_10

# # 3C Dynamic
# python eval.py --config-name=cube_DEPTH_OGB.yaml single_channel=False dynamic_depth=True preprocessing=minmax policy=/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/results_16_07/front_pixels_depth_train_1000_val_100_episodes/20260730_030255/lewm_epoch_10

# # NORMALS horizon5 
# python eval.py --config-name=cube_NRM_OGB.yaml plan_config.horizon=5 policy=/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/results_16_07/front_pixels_NORMALS_train_1000_val_100_episodes/20260718_172939/lewm_epoch_10

# NORMALS horizon10
# python eval.py --config-name=cube_NRM_OGB.yaml plan_config.horizon=10 policy=/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/results_16_07/front_pixels_NORMALS_train_1000_val_100_episodes/20260718_172939/lewm_epoch_10

# # DINO RGB horizon5 
# python eval.py --config-name=cube_RGB.yaml eval.num_total_eval=50 eval.num_eval=5 plan_config.horizon=5 policy=/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/dinoresults_15_07/cube_single_expert/20260724_060038/prejepa_ogcube_of_1_dinov2_small_psmall_epoch_10

# DINO RGB horizon10
python eval.py --config-name=cube_RGB.yaml eval.num_total_eval=50 eval.num_eval=4 plan_config.horizon=10 policy=/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/dinoresults_15_07/cube_single_expert/20260724_060038/prejepa_ogcube_of_1_dinov2_small_psmall_epoch_10
