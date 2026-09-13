import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch import nn
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
from omegaconf import OmegaConf, open_dict

from PIL import Image
import numpy as np

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack, get_episode_subset, save_episode_as_video
from custom_callbacks import ModelDiagnosticsCallback, LeWMPredictorIGCallback





def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    # ###################################
    # print(f"\n\nTesting dataset loading")
    # print(f"Dataset length: {len(dataset)}")
    # print(dataset.offsets[-1])

    # print(dataset.column_names)

    # print("\n\nTrying to index into the pixels column")
    # ####################################

    print("\n\nData preprocessing")
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            
            print("Processing column:", col)
            
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    # # To check the structure of the dataset, we can print out a sample and its keys/shapes:
    # sample = dataset[0]
    # if isinstance(sample, dict):
    #     for k, v in sample.items():
    #         print(f"{k}: {v.shape}")
    # else:
    #     print(f"Shape: {sample.shape}")

    # Get subset of the dataset corresponding to episode 0 (first episode)
    # dataset = get_episode_subset(dataset, 0)
    print(f"\n\nDataset size: {len(dataset)}")

    ### Check if the episode subset is correct by saving it as a video
    # save_episode_as_video(dataset, output_path="episode_0_test_sample.mp4")

    # rnd_gen = torch.Generator().manual_seed(cfg.seed)
    # train_set, val_set = spt.data.random_split(
    #     dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    # )

    # train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    # val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    #################################################################################################
    ##       overfitting on one batch (batch_size = 2) with consecutive frames （frameskip = 1）    ##
    #################################################################################################
    # batch_size = cfg.loader.batch_size

    # # two different clips in the overfitting set
    # overfit_subset = torch.utils.data.Subset(train_set, range(batch_size))

    train_dataset = get_episode_subset(dataset, cfg.train_episode_idx)

    val_dataset = get_episode_subset(dataset, cfg.val_episode_idx)

    # two same clips in the overfitting set
    overfit_train_subset = torch.utils.data.Subset(train_dataset, [1])

    overfit_val_subset = torch.utils.data.Subset(val_dataset, [101])

    train = torch.utils.data.DataLoader(overfit_train_subset, **cfg.loader,shuffle=False, drop_last=False)
    val = torch.utils.data.DataLoader(overfit_val_subset, **cfg.loader, shuffle=False, drop_last=False)

    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])

    # print("Inspect all batches")
    # for batch in train:
    #     print(batch["pixels"].shape)

    #     for idx, batch_item_pixels in enumerate(batch["pixels"]):
    #         for i, frame in enumerate(batch_item_pixels):
    #             arr = frame.numpy()
    #             arr = np.transpose(arr, (1,2,0))
    #             arr_vis = arr * std + mean
    #             arr_vis = np.clip(arr_vis, 0, 1)
    #             arr_vis = (255 * arr_vis).astype(np.uint8)
    #             im = Image.fromarray(arr_vis)
    #             im.save(f'rgb_batch_{idx}_frame_{i}.png')
    #         batch_item_pixels =  batch["pixels"]

    #     print(batch["action"])
        
    #####################################################################
    
    # print("\n\nFinal Dataloader being used for training")
    # print(len(train))
    # print(len(val))

    print("\n=== Dataloader Inspection ===")

    print("Train batches:", len(train))
    print("Val batches:", len(val))

    # Inspect first batch
    train_batch = next(iter(train))
    print("\nTrain Batch keys:")
    print(train_batch.keys())
    print("pixels shape:", train_batch["pixels"].shape)
    print("action shape:", train_batch["action"].shape)

    val_batch = next(iter(val))
    print("\nVal Batch keys:")
    print(val_batch.keys())
    print("pixels shape:", val_batch["pixels"].shape)
    print("action shape:", val_batch["action"].shape)

    

    # frame0 = batch["pixels"][0, 0]
    # frame1 = batch["pixels"][0, 1]

    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # print("Inspect all batches")
    # output_dir = 'outputs/overfit'
    # os.makedirs(output_dir, exist_ok=True)
    # for batch in train:
    #     print(batch["pixels"].shape)

    #     for idx, (batch_item_pixels, batch_item_action) in enumerate(zip(batch["pixels"], batch["action"])):
    #         for i, (frame, action) in enumerate(zip(batch_item_pixels, batch_item_action)):
    #             arr = frame.numpy()
    #             arr = np.transpose(arr, (1,2,0))
    #             arr_vis = arr * std + mean
    #             arr_vis = np.clip(arr_vis, 0, 1)
    #             arr_vis = (255 * arr_vis).astype(np.uint8)
    #             im = Image.fromarray(arr_vis)
    #             im.save(f'{output_dir}/sanity_check_new_train_rgb_batch_{idx}_frame_{i}.png')
    #             print(f"batch {idx} frame {i} has action {action}")
    #         batch_item_pixels =  batch["pixels"]

    # for batch in val:
    #     print(batch["pixels"].shape)

    #     for idx, (batch_item_pixels, batch_item_action) in enumerate(zip(batch["pixels"], batch["action"])):
    #         for i, (frame, action) in enumerate(zip(batch_item_pixels, batch_item_action)):
    #             arr = frame.numpy()
    #             arr = np.transpose(arr, (1,2,0))
    #             arr_vis = arr * std + mean
    #             arr_vis = np.clip(arr_vis, 0, 1)
    #             arr_vis = (255 * arr_vis).astype(np.uint8)
    #             im = Image.fromarray(arr_vis)
    #             im.save(f'{output_dir}/sanity_check_new_val_rgb_batch_{idx}_frame_{i}.png')
    #             print(f"batch {idx} frame {i} has action {action}")
    #         batch_item_pixels =  batch["pixels"]

    # print("frame0 mean:", frame0.mean().item())
    # print("frame1 mean:", frame1.mean().item())
    # batch1 = next(iter(train))
    # batch2 = next(iter(train))
    # print("batch1 and batch2 close:",
    #     torch.allclose(
    #         batch1["pixels"],
    #         batch2["pixels"]
    #     )
    # )

    # for key, value in batch.items():
    #     if torch.is_tensor(value):
    #         print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    #     else:
    #         print(f"{key}: type={type(value)}")

    # for i, batch in enumerate(train):
    #     print(f"\nBatch {i}")

    #     for key, value in batch.items():
    #         if torch.is_tensor(value):
    #             print(f"{key}: {value.shape}")

    #     if i == 2:
    #         break

    # # See what is inside a single training example: 

    # batch = next(iter(train))
    # print("\n\nBatch keys:", batch.keys())
    # print("Batch 'pixels' shape:", batch['pixels'].shape)
    # print("Batch 'action' shape:", batch['action'].shape)
    # print("Batch 'observation' shape:", batch['observation'].shape)
    # print("Batch 'proprio' shape:", batch['proprio'].shape)

    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.LayerNorm, # originally torch.nn.BatchNorm1d
        # norm_fn=torch.nn.BatchNorm1d,
    )




    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.LayerNorm, # originally torch.nn.BatchNorm1d
        # norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    # ##################################################
    # # check the structure of the model FIXME need debugging to make this work
    # dummy_img = torch.randn(1, 3, 224, 224)
    # dummy_act = torch.randn(1, 5)

    # torch.onnx.export(
    #     world_model,
    #     (dummy_img, dummy_act),
    #     "outputs/lewm.onnx",
    #     opset_version=17,
    #     input_names=["image", "action"],
    #     output_names=["output"]
    # )
    # ###################################################

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )
    diagnostics_callback = ModelDiagnosticsCallback()
    predictor_callback = LeWMPredictorIGCallback(every_n_epochs=1, n_steps=30, ctx_len = cfg.wm.history_size)
    early_stop_callback = EarlyStopping(
        monitor="validate/loss",
        patience=cfg.early_stop_patience,
        mode="min",
        min_delta=cfg.early_stop_min_improv,
        verbose=True,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, diagnostics_callback, predictor_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
