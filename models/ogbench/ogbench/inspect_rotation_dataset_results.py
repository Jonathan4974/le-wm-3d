import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

DATASET = "/home/student/data/ogbench/datasets/ogbench/cube_single_expert.h5"
CSV = "rotation_subset_10_0.5.csv"

WINDOW = 32
N_SAMPLES = 25

OUT_DIR = "rotation_subset_visualization"
os.makedirs(OUT_DIR, exist_ok=True)

# show these frames from each clip
VIS_FRAMES = [0, 8, 16, 24, 31]

# -------------------------------------------------------
# Load subset
# -------------------------------------------------------

df = pd.read_csv(CSV)

# Random sample
df = df.sample(
    min(N_SAMPLES, len(df)),
    random_state=0,
)

# -------------------------------------------------------
# Open HDF5
# -------------------------------------------------------

with h5py.File(DATASET, "r") as f:

    pixels = f["pixels"]
    obs = f["observation"]

    print(f["pixels"].shape)
    dset = f["pixels"]

    print("Compression:", dset.compression)
    print("Compression opts:", dset.compression_opts)

    print("Filters:")
    print(dset.id.get_create_plist())

    for i, row in df.iterrows():

        start = int(row["start"])

        clip = pixels[start:start + WINDOW]

        block_pos = obs[start:start + WINDOW, 19:22]

        fig, axes = plt.subplots(
            1,
            len(VIS_FRAMES),
            figsize=(15,3)
        )

        for ax, frame_idx in zip(axes, VIS_FRAMES):

            img = clip[frame_idx]

            # HDF5 may store uint8 already
            if img.dtype != np.uint8:
                img = (255 * img).astype(np.uint8)

            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"t={frame_idx}")

        fig.suptitle(
            f"Episode {int(row['episode'])}\n"
            f"Start {start}\n"
            f"Rotation {row['block_rotation_deg']:.1f}°   "
            f"Translation {row['block_translation']:.3f} m"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUT_DIR,
                f"{i:03d}.png"
            ),
            dpi=200,
        )

        plt.close(fig)

print("Done.")