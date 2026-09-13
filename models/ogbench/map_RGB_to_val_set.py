import h5py
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RGB_DATASET = "/home/student/data/ogbench/datasets/ogbench/cube_single_expert.h5"
EPISODE_ORDER = "/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/episode_order.pt"
OUTPUT_DATASET = "/home/student/data/ogbench/datasets/ogbench/cube_single_expert_VAL_100.h5"

import h5py
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Load validation episode ids
# ---------------------------------------------------------------------

episode_order = torch.load(EPISODE_ORDER)
val_episode_ids = np.asarray(episode_order[1000:1100], dtype=np.int32)

print("Validation episodes:")
print(val_episode_ids)

# ---------------------------------------------------------------------
# Open datasets
# ---------------------------------------------------------------------

with h5py.File(RGB_DATASET, "r") as src, \
     h5py.File(OUTPUT_DATASET, "w") as dst:

    ep_offset = src["ep_offset"][:]
    ep_len = src["ep_len"][:]

    # ---------------------------------------------------------------
    # Collect frame indices
    # ---------------------------------------------------------------

    selected_indices = []

    new_ep_len = []
    new_ep_offset = []

    running_offset = 0

    for original_ep in tqdm(val_episode_ids, desc="Collecting episodes"):

        start = ep_offset[original_ep]
        length = ep_len[original_ep]

        selected_indices.extend(range(start, start + length))

        new_ep_len.append(length)
        new_ep_offset.append(running_offset)

        running_offset += length

    selected_indices = np.asarray(selected_indices, dtype=np.int64)

    print(f"Selected {len(selected_indices)} frames.")

    # ---------------------------------------------------------------
    # New episode indices
    # ---------------------------------------------------------------

    new_ep_idx = np.repeat(
        np.arange(len(val_episode_ids), dtype=np.int32),
        new_ep_len,
    )

    # ---------------------------------------------------------------
    # Copy every dataset
    # ---------------------------------------------------------------

    for key in tqdm(src.keys(), desc="Copying datasets"):

        if key == "ep_idx":
            dst.create_dataset(key, data=new_ep_idx, compression="gzip")

        elif key == "ep_len":
            dst.create_dataset(key, data=np.asarray(new_ep_len, dtype=src[key].dtype), compression="gzip")

        elif key == "ep_offset":
            dst.create_dataset(key, data=np.asarray(new_ep_offset, dtype=src[key].dtype), compression="gzip")

        else:

            data = src[key]

            # Per-frame dataset
            if data.shape[0] == len(src["ep_idx"]):

                out = dst.create_dataset(
                    key,
                    shape=(len(selected_indices),) + data.shape[1:],
                    dtype=data.dtype,
                    compression="gzip",
                )

                write_ptr = 0

                for original_ep in val_episode_ids:

                    start = ep_offset[original_ep]
                    length = ep_len[original_ep]

                    # Copy one contiguous block
                    out[write_ptr:write_ptr+length] = data[start:start+length]

                    write_ptr += length

            # Per-episode dataset
            elif data.shape[0] == len(src["ep_len"]):

                out = dst.create_dataset(
                    key,
                    data=data[val_episode_ids],
                    compression="gzip",
                )

            else:

                out = dst.create_dataset(
                    key,
                    data=data[:],
                    compression="gzip",
                )


print("Done.")


with h5py.File(OUTPUT_DATASET) as f:
    print("Episodes:", len(f["ep_len"]))
    print("Frames:", len(f["ep_idx"]))

    print("Unique ep_idx:", np.unique(f["ep_idx"]))

    print("ep_len:", np.unique(f["ep_len"]))

    print("First offsets:", f["ep_offset"][:5])

    print("Last offset:", f["ep_offset"][-1])