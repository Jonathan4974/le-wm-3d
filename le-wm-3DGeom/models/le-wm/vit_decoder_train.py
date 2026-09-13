import os
import torch
from torch.utils.data import DataLoader, random_split, DataLoader
from tqdm import tqdm
import hydra

from vit_decoder_module import VisualizationDecoder
from utils import EmbImageDataset


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):

        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss
            self.counter = 0

            self.best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

            return False  # not stop

        else:
            self.counter += 1

            if self.counter >= self.patience:
                return True  # stop

        return False

# @hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(
    npz_path="train.npz",
    save_path="./checkpoints/decoder.pth",
    epochs=50,
    batch_size=64,
    val_ratio=0.1,
    num_workers=4,
):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #########################
    ##       dataset       ##
    #########################
    dataset = EmbImageDataset(npz_path)

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    print("val_size:", val_size)
    
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    ##############################
    ##       model / optim      ##
    ##############################
    model = VisualizationDecoder(
        emb_dim=192,
        decoder_dim=512,
        depth=6,
        num_heads=8,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05,
    )

    loss_fn = torch.nn.MSELoss()

    early_stopper = EarlyStopping(patience=10)
    ##########################
    ##       training       ##
    ##########################
    for epoch in range(epochs):

        # ======================
        # TRAIN
        # ======================
        model.train()

        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]")

        for emb, img in pbar:

            emb = emb.to(device)
            img = img.to(device)

            pred = model(emb)

            loss = loss_fn(pred, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        # ======================
        # VALIDATION
        # ======================
        model.eval()

        val_loss = 0

        with torch.no_grad():

            for emb, img in val_loader:

                emb = emb.to(device)
                img = img.to(device)

                pred = model(emb)

                loss = loss_fn(pred, img)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        # ======================
        # EARLY STOP
        # ======================
        stop = early_stopper.step(val_loss, model)

        if stop:
            print("Early stopping triggered.")

            # restore best model
            model.load_state_dict(early_stopper.best_state)
            break

    # save the model
    torch.save({
        "model": model.state_dict(),
        "config": {
            "emb_dim": 192,
            "decoder_dim": 512,
            "depth": 6
        }
    }, save_path)
    print(f"Model saved to: {save_path}")

    return save_path



# @hydra.main(version_base=None, config_path="./config/train", config_name="lewm")


if __name__ == "__main__":

    model_path = run(
        npz_path="/home/student/users/Aaron_workspace/le-wm-3DGeom/models/le-wm/outputs/emb_latents/lewm_final_trained_100_eps.pt/_val_100_latents_with_pixels/merged_data/merged_decoder_train_data.npz",
        save_path="./decoder_ckpt/decoder_1.pth",
        epochs=100,
        batch_size=64,
        val_ratio=0.1,
        num_workers=4,
    )
