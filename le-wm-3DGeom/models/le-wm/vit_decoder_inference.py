import os
import torch
import glob
import numpy as np
from PIL import Image
from torchvision.utils import save_image

from vit_decoder_module import VisualizationDecoder

def evaluate(
    model_path,
    embedding,
    image,
    idx,
    save_dir="./reconstructions",
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset = EmbImageDataset(npz_path)

    ckpt = torch.load(model_path, map_location=device)
    model = VisualizationDecoder(**ckpt["config"])
    model.load_state_dict(ckpt["model"])

    model = model.to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        cls_emb = torch.from_numpy(embedding)

        if cls_emb.dim() == 1:
            cls_emb = cls_emb.unsqueeze(0)

        cls_emb = cls_emb.to(device)

        pred = model(cls_emb)

        save_image(
            pred.cpu(),
            f"{save_dir}/decoded_img_{idx}.png",
        )

    Image.fromarray(np.transpose(image, (1, 2, 0)).astype(np.uint8)).save(f"{save_dir}/origin_img_{idx}.png")

    print(f"Saved image to {save_dir}")

if __name__ == "__main__":

    LATENT_DIR="/home/student/users/Aaron_workspace/le-wm-3DGeom/models/le-wm/outputs/emb_latents/lewm_final_trained_100_eps.pt/_val_100_latents_with_pixels"

    files = sorted(glob.glob(os.path.join(LATENT_DIR, "*.npz")))

    # read one sample to set the shape and dtype
    sample = np.load(files[0], mmap_mode="r")

    # print(type(sample["pixels"]))
    # print(sample["pixels"].shape)
    # print(sample["pixels"].ndim)

    for i, (cls, img) in enumerate(zip(sample["cls_emb"], sample["pixels"].squeeze(0))):

        print(type(img))
        print(img.shape)
        print(img.ndim)
        print(img.dtype)
        print(img.min(), img.max())
        print(type(cls))
        print(cls.shape)
        print(cls.ndim)

        break
        # evaluate(
        #     model_path="./decoder_ckpt/decoder_1.pth",
        #     embedding=cls,
        #     save_dir="./decoded_img",
        #     image = img,
        #     idx = i
        # )