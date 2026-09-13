import os
from functools import partial
from pathlib import Path
import numpy as np

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datetime import datetime
import torch
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from omegaconf import OmegaConf, open_dict
from scipy.stats import pearsonr


from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, instantiate_world_model, \
            sanity_check_on_loaded_batch, collect_all_callbacks
from utils import ModelObjectCallBack, LinearProbeCallback

from preprocessing import depth_img_preprocessor, normal_img_preprocessor, rgbd_img_preprocessor, \
                    da3_img_preprocessor, point_map_preprocessor, depth_da3_single_img_preprocessor, \
                    depth_da3_triple_img_preprocessor

from torch.utils.data import Subset
import random

import stable_worldmodel.data

def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)           # [B,T,D]                           

    emb = output["emb"]  # (B, T, D)    -> T = number of sampled frames in one batch sequence, eg [f0, f4, f8, f12] with T=4 
    act_emb = output["act_emb"]

    # print("Encoded embeddings shape: ", emb.shape)
    # print("Encoded action embeddings shape: ", act_emb.shape)

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:].contiguous()

    # print("Context embeddings shape: ", ctx_emb.shape)
    # print("Context action embeddings shape: ", ctx_act.shape)

    tgt_emb = emb[:, n_preds:] # label

    # print("Target embeddings shape: ", tgt_emb.shape)
    # print("ctx_len: ", ctx_len, "n_preds: ", n_preds)

    # print("Running prediction")
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred
    # print("Prediction complete. pred_emb shape: ", pred_emb.shape)

    # LeWM loss
    # print("Target embedding shape: ", tgt_emb.shape)
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  
    
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, on_epoch=True, sync_dist=False)

    # print("*****************************************")
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_name_without_full_path = cfg.data.dataset.name.split("/")[-1]
    SAVE_CKPT_PATH = f"/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/checkpoints/{dataset_name_without_full_path}/"
    os.makedirs(SAVE_CKPT_PATH, exist_ok=True)
    RUN_OUTPUT_DIR=f"/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/results_16_07/{dataset_name_without_full_path}/{run_timestamp}"
    os.makedirs(RUN_OUTPUT_DIR,exist_ok=True )


    RGB_DEPTH = "rgb_depth" in cfg.data.dataset.name.lower()

    if not RGB_DEPTH:
        print("Using the wrong dataset!")
        exit(1)

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

    transforms = [rgbd_img_preprocessor(img_size=cfg.img_size,)]       # RGB+D
    
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    print("Experimenting logger")

    print(dataset.clip_indices[:10])
    print(dataset.clip_indices[-10:])

    print("\n\nDataset initialized with the following parameters:")
    print(OmegaConf.to_yaml(cfg))
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:

            if col in ["qpos", "qvel"]:
                continue

            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    print("dataset=", cfg.data.dataset)
    print("dataset.column_names=", dataset.column_names)
    print("dataset.frameskip=", dataset.frameskip)
    print("dataset.span=", dataset.span)
    print("dataset.num_steps=", dataset.num_steps)
    print("total clip_indices=", len(dataset.clip_indices))
    print("dataset samples= ", len(dataset))

    ##############################
    ##       DATALOADERS      ##
    ##############################
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = torch.utils.data.DataLoader(dataset, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val_loader = train_loader
    print("number of batches in data loader=", len(train_loader))

    all_episode_ids = torch.load("episode_order.pt")
    print(all_episode_ids[:10])

    train_episode_count = cfg.train_num_episodes
    print("\n\ntrain_episode_count=", train_episode_count)

    # Fixed validation split:
    selected_train_episodes = set(all_episode_ids[0:train_episode_count])
    selected_val_episodes   = set(all_episode_ids[train_episode_count:train_episode_count+cfg.val_num_episodes])

    print(dataset.episode_ids[:20])

    train_indices = [
        idx
        for idx, (local_ep, _) in enumerate(dataset.clip_indices)
        if dataset.episode_ids[local_ep] in selected_train_episodes
    ]

    val_indices = [
        idx
        for idx, (local_ep, _) in enumerate(dataset.clip_indices)
        if dataset.episode_ids[local_ep] in selected_val_episodes
    ]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    print(
        f"Dataset split | "
        f"train episodes={len(selected_train_episodes)} "
        f"train samples={len(train_set)} | "
        f"val episodes={len(selected_val_episodes)} "
        f"val samples={len(val_set)}"
    )
    print("train_indices[:20]=", train_indices[:20])
    print("train_indices[-20:]=",train_indices[-20:])
    print("clip_indices[20:]=", dataset.clip_indices[:20])

    train_loader = torch.utils.data.DataLoader(train_set,**cfg.loader,shuffle=True,drop_last=True,generator=rnd_gen,)
    val_loader = torch.utils.data.DataLoader(val_set,**cfg.loader,shuffle=False,drop_last=False,)

    print()
    print("len(train_loader)=", len(train_loader))
    print("len(val_loader)=", len(val_loader))

    ##############################
    ##       MODEL / OPTIM      ##
    ##############################

    world_model = instantiate_world_model(cfg)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    print(type(world_model))
    print(world_model.__class__)
    print(world_model.__class__.__module__)
    print(world_model.__class__.__mro__)


    ##########################
    ##       TRAINING       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    csv_logger = CSVLogger(save_dir="logs", name="csv")

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    all_callbacks = collect_all_callbacks(cfg, RUN_OUTPUT_DIR)

    run_baseline_ckpt = cfg.get("run_baseline_ckpt", False)
    if run_baseline_ckpt:       # Evaluate the model on the official lewm weights

        ### Run with 
        '''
        python train.py data=ogb data.dataset.name=/home/student/data/ogbench/cube_single_expert wandb.config.name=CKPT_exp wm.history_size=3 wm.num_preds=2 train_num_episodes=100 val_num_episodes=100 run_baseline_ckpt=True

        plus change configs: 
        (1): official RGB lewm checkpoint: 
            - checkpoint path: "/home/student/data/baseline_ckpt/hf_cube/weights.pt"
            - wm.history_size=3 + wm.num_preds=2
            - data.dataset.name = cube_single_expert
        (2): RGB trained on 100 episodes, 100 epochs: 
            - checkpoint path: /home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/checkpoints/cube_single_expert/weights_20260708_000215.pt
            - wm.history_size=1 + wm.num_preds=1
            - data.dataset.name = cube_single_expert
        (3): DEPTH trained on 100 episodes, 100 epochs: 
            - checkpoint path: /home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/checkpoints/front_pixels_depth_train_1000_val_100_episodes//weights_20260707_215039.pt
            - wm.history_size=1 + wm.num_preds=1
            - data.dataset.name = "/home/student/data/ogbench/front_pixels_depth_train_1000_val_100_episodes"
        (4): POINT MAP trained on 100 episodes, for 5 epochs
            - /home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/checkpoints/gt_point_map_100_val_10_episodes//weights_20260716_114513.pt
            - wm.history_size=1 + wm.num_preds=1
            - data.dataset.name = gt_point_map_100_val_10_episodes
        '''

        
        checkpoint_path = "/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/checkpoints/cube_single_expert//weights_20260723_052148.pt"
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        world_model.model.load_state_dict(state_dict, strict=True)

        print(f"\n\nLoaded pretrained weights into the model from {checkpoint_path}.")
        print(f"\n\nCheckpoint found at {checkpoint_path}. Validating model with this checkpoint.")

        latent_plots_dir = "latent_plots/"
        

        trainer = pl.Trainer(
            **cfg.trainer,
            callbacks=all_callbacks,
            num_sanity_val_steps=0,
            logger=logger,
            enable_checkpointing=False,
        )

        probe_callback = LinearProbeCallback()
        probe_callback.run_probe_call(train_loader=train_loader, val_loader=val_loader, trainer=trainer, pl_module=world_model)

        print("\n\nValidation on baseline checkpoint complete. ")
        
        return
    else: 
        print("\n\nNo baseline checkpoint loaded. Starting training from scratch.\n\n")
        trainer = pl.Trainer(
            **cfg.trainer,
            callbacks=all_callbacks,
            num_sanity_val_steps=0,
            logger=logger,
            enable_checkpointing=False,
        )

        print(trainer.log_every_n_steps)
        print(trainer.accumulate_grad_batches)
        print(trainer.global_step)

        manager = spt.Manager(
            trainer=trainer,
            module=world_model,
            data=data_module,
            ckpt_path=None  # run_dir / f"{cfg.output_model_name}_weights.ckpt",
        )

        print("train batches =", len(train_loader))
        print("val batches   =", len(val_loader))

        trainer = manager.trainer
        print("trainer.limit_train_batches =", trainer.limit_train_batches)
        print("trainer.limit_val_batches   =", trainer.limit_val_batches)
        print("trainer.fast_dev_run        =", trainer.fast_dev_run)
        print("trainer.overfit_batches     =", trainer.overfit_batches)

        manager()

        print("\n\n -- DONE WITH TRAINING -- \n\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(world_model.model.state_dict(),os.path.join(SAVE_CKPT_PATH, f"weights_{timestamp}.pt"))
        print(f"Saved checkpoint: {SAVE_CKPT_PATH}/weights_{timestamp}.pt")
    
        return

if __name__ == "__main__":
    run()
