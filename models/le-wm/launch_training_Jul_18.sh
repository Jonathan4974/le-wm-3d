#

# DONE (NORMALS): 
#python train.py data=ogb  data.dataset.name=/home/student/data/ogbench/front_pixels_NORMALS_train_1000_val_100_episodes  trainer.max_epochs=10 wandb.config.name=final_NORMS_OGB_1000_eps_10_epochs  wm.history_size=3 wm.num_preds=2 train_num_episodes=1000 val_num_episodes=100+

# OGB Point Maps:
python train.py data=ogb data.dataset.name=/home/student/data/ogbench/gt_point_map_1000_val_100_episodes trainer.max_epochs=10 wandb.config.name=final_PTMAP_OGB_1000_eps_10_epochs wm.history_size=3 wm.num_preds=2 train_num_episodes=9000 val_num_episodes=1000

# OGB Depth Maps: 
python train.py data=ogb data.dataset.name=/home/student/data/ogbench/front_pixels_depth_train_1000_val_100_episodes trainer.max_epochs=10 wandb.config.name=final_DEPTH_OGB_1000_eps_10_epochs wm.history_size=3 wm.num_preds=2 train_num_episodes=1000 val_num_episodes=100

# RGB+D
python train.py data=ogb data.dataset.name=/home/student/data/ogbench/front_pixels_RGB_DEPTH_train_100_val_10_episodes trainer.max_epochs=10 wandb.config.name=final_RGBD_OGB_1000_eps_10_epochs wm.history_size=3 wm.num_preds=2 train_num_episodes=1000 val_num_episodes=100 concat_RGBD=True 


# OGB DEPTH + 10 epochs + 10k dataset
python train.py data=ogb data.dataset.name=/home/student/data/ogbench/front_pixels_DEPTH_ONLY_train_9000_val_1000_episodes trainer.max_epochs=10 wandb.config.name=final_DEPTH_OGB_10k_eps_10_epochs wm.history_size=3 wm.num_preds=2 train_num_episodes=9000 val_num_episodes=1000
