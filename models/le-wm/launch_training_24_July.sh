

python train.py data=ogb data.dataset.name=/home/student/data/ogbench/front_pixels_RGB_DEPTH_train_1000_val_100_episodes trainer.max_epochs=10 concat_RGBD=True wandb.config.name=RGBD_std_normal_1k_10_epochs wm.history_size=3 wm.num_preds=2 train_num_episodes=1000 val_num_episodes=100
python train_dinowm.py data=ogb trainer.max_epochs=10 wandb.config.name=DINO_1k_10_epochs wm.history_size=3 wm.num_preds=2