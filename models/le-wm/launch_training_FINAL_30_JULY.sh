


### SINGLE CHANNEL
python train.py data=ogb data.dataset.name=/home/student/data/ogbench/front_pixels_depth_train_1000_val_100_episodes trainer.max_epochs=10 wandb.config.name=FINAL_SINGLE_channel_DEPTH wm.history_size=3 wm.num_preds=2 train_num_episodes=1000 val_num_episodes=100 single_channel_depth=True

### DYNAMIC CHANNELS 
python train.py data=ogb data.dataset.name=/home/student/data/ogbench/front_pixels_depth_train_1000_val_100_episodes trainer.max_epochs=10 wandb.config.name=FINAL_DYNAMIC_CHANNELS_DEPTH wm.history_size=3 wm.num_preds=2 train_num_episodes=1000 val_num_episodes=100 dynamic_channel_depth=True