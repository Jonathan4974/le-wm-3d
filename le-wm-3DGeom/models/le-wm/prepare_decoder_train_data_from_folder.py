import os
import glob
import numpy as np

LATENT_DIR="/home/student/users/Aaron_workspace/le-wm-3DGeom/models/le-wm/outputs/emb_latents/lewm_final_trained_100_eps.pt/_val_100_latents_with_pixels"
OUTPUT_FILE = "/home/student/users/Aaron_workspace/le-wm-3DGeom/models/le-wm/outputs/emb_latents/lewm_final_trained_100_eps.pt/_val_100_latents_with_pixels/merged_data/merged_decoder_train_data.npz"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

files = sorted(glob.glob(os.path.join(LATENT_DIR, "*.npz")))

# read one sample to set the shape and dtype
sample = np.load(files[0], mmap_mode="r")

cls_shape = (len(files),) + sample["cls_emb"].shape
img_shape = (len(files),) + sample["pixels"].squeeze(0).shape

cls_dtype = sample["cls_emb"].dtype
img_dtype = sample["pixels"].dtype

# print("cls_shape", cls_shape)
# print("cls_dtype", cls_dtype)
# print("img_shape", img_shape)
# print("img_dtype", img_dtype)

# reserve memory for output
cls_arr = np.empty(cls_shape, dtype=cls_dtype)
img_arr = np.empty(img_shape, dtype=img_dtype)

idx = 0
for f in files:
    data = np.load(f, mmap_mode="r")

    if "cls_emb" not in data or "pixels" not in data:
        print(f"Skip {f}, missing keys")
        continue

    cls_arr[idx] = data["cls_emb"]
    img_arr[idx] = data["pixels"].squeeze(0)

    idx += 1

cls_arr = cls_arr.reshape(-1, 192)
img_arr = img_arr.reshape(-1, 3, 224, 224)

np.savez(OUTPUT_FILE, **{
    "cls_emb": cls_arr,
    "pixels": img_arr
})

print("Done")
print("cls_emb", cls_arr.shape)
print("pixels", img_arr.shape)