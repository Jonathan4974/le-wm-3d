


python train.py data=ogb data.dataset.name=/home/student/data/ogbench/DA3_single_depth_1000_val_100_episodes trainer.max_epochs=10 wandb.config.name=DA3_1k_10_epochs wm.history_size=3 wm.num_preds=2 train_num_episodes=1000 val_num_episodes=100
python train.py data=ogb trainer.max_epochs=10 wandb.config.name=RGB_1k_10_epochs wm.history_size=3 wm.num_preds=2