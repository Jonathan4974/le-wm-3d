import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

import cv2
from torch.utils.data import Subset

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")


def get_episode_subset(dataset, k):
    """
    Retrieves all slice indices belonging to the k-th episode and returns a Subset.
    """
    # Iterate through clip_indices to find all global indices 'i' where the episode index matches k
    indices = [i for i, (ep, start) in enumerate(dataset.clip_indices) if ep == k]
    
    if not indices:
        print(f"Warning: Episode {k} is too short to generate any slices (span={dataset.span})")
        
    return Subset(dataset, indices)


def get_episode_subset_fast(dataset, k):
    """
    Calculates the index range for the k-th episode mathematically and returns a Subset.
    """
    # 1. Define slice count logic per episode
    # This matches the base class logic: (length - span + 1) if length >= span
    def get_num_slices(length, span):
        return (length - span + 1) if length >= span else 0

    # 2. Sum the slice counts of all episodes prior to k to find the starting global index
    start_idx = sum(get_num_slices(dataset.lengths[i], dataset.span) for i in range(k))
    
    # 3. Determine the number of slices contributed by the target episode k
    num_slices = get_num_slices(dataset.lengths[k], dataset.span)
    
    # 4. Construct the range and wrap it in a Subset
    return Subset(dataset, range(int(start_idx), int(start_idx + num_slices)))


def save_episode_as_video(dataset, output_path="episode_check.mp4", fps=24):
    """
    Iterates over dataset samples and saves them as an MP4 video
    to verify episode continuity.
    """

    print(f"\n\nTesting episode correctness")

    # Get first frame to infer shape
    sample = dataset[0]
    frame = sample["pixels"]

    print(f"Sample keys: {sample.keys()}")
    print(f"dataset[0] shape: {frame.shape}, dtype: {frame.dtype}")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (224, 224)
    )


    for i in range(len(dataset)):           # iterate over all temporal windows
        sample = dataset[i]
        cur_temporal_window = sample["pixels"]

        if hasattr(cur_temporal_window, "cpu"):
            cur_temporal_window = cur_temporal_window.cpu().numpy()

        # for frame_index in range(cur_temporal_window.shape[0]):
        #     frame = cur_temporal_window[frame_index]               # [C,H,W]
        #     if frame.shape[0] in [1, 3]:                          # (C,H,W) -> (H,W,C)
        #         frame = np.transpose(frame, (1, 2, 0))  

        #     # float32 [0,1] -> uint8 [0,255]
        #     frame = (frame * 255).clip(0, 255).astype(np.uint8)

        #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        #     writer.write(frame)

        frame = cur_temporal_window[0]               # [C,H,W], only take the first frame of the window for visualization
        if frame.shape[0] in [1, 3]:                          # (C,H,W) -> (H,W,C)
            frame = np.transpose(frame, (1, 2, 0))  

        # float32 [0,1] -> uint8 [0,255]
        frame = (frame * 255).clip(0, 255).astype(np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        writer.write(frame)


    writer.release()
    print(f"Saved video to: {output_path}")


class EmbImageDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)

        self.emb = data["cls_emb"].astype(np.float32)
        self.pixels = data["pixels"].astype(np.float32)

        if self.pixels.max() > 1:
            self.pixels /= 255.0

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.emb[idx]),
            torch.from_numpy(self.pixels[idx]),
        )