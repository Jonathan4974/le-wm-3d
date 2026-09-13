
import hdf5plugin 
import h5py
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio

DATASET_PATH = Path("data_link/ogbench/cube_single_expert.h5")
OUTPUT_DIR = Path("./outputs/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EPISODE_ID = 1


with h5py.File(DATASET_PATH, "r") as f:

    print("Keys:", list(f.keys()))
    print("Action length:", f["action"].shape)
    print("Action sample:", f["action"][0])
    print("control shape:", f["control"].shape)
    print("control sample:", f["control"][0])
    print("Pixels shape:", f["pixels"].shape)

    # loaded_offsets = f["ep_offset"]
    # offset_distances = []

    # for i in range(10000):
    #     offset = loaded_offsets[i] 
    #     next_offset = loaded_offsets[i + 1] if i + 1 < len(loaded_offsets) else len(f["pixels"])
    #     length = next_offset - offset
    #     offset_distances.append(length)

    # print(np.mean(offset_distances), np.min(offset_distances), np.max(offset_distances))
    # print(f["pixels"][0])  # just try one frame
    # pixels = f["pixels"][0]
    # print(pixels.shape)
    # img = pixels  # already f["pixels"][0]

    # plt.imshow(img)
    # plt.title("First frame")
    # plt.axis("off")
    # plt.savefig(OUTPUT_DIR / f"episode_{EPISODE_ID}_frame_0.png")
    # print(f"Saved frame_0.png for episode {EPISODE_ID}")

    # ep_id = EPISODE_ID

    # offset = f["ep_offset"][ep_id]
    # length = f["ep_len"][ep_id]

    # pixels = f["pixels"][offset:offset+length]      # for the full sequence in episode[0]

    # print("Episode pixels shape:", pixels.shape)

    # frames = []

    # for i in range(len(pixels)):
    #     img = pixels[i]

    #     # ensure uint8 (should already be, but safe)
    #     if img.dtype != np.uint8:
    #         img = (img * 255).astype(np.uint8)
    #     frames.append(img)

    # output_path = OUTPUT_DIR / f"episode_{EPISODE_ID}.gif"
    # imageio.mimsave(output_path, frames, fps=20)

    # print(f"Saved GIF to {output_path}")

# print("--------------------------------------------------------------------------------------------------------------")

# print("=" * 60)
# print(f"开始分析 HDF5 文件: {DATASET_PATH}")
# print("=" * 60)

# # 用于分类的列表
# meaningful_keys = []
# dead_keys = []

# with h5py.File(DATASET_PATH, 'r') as f:
#     for key in f.keys():
#         data = f[key]
        
#         # 1. 检查是否是空数据集
#         if data.shape is None or data.size == 0:
#             dead_keys.append((key, "Shape is None or Empty", None))
#             continue
            
#         shape = data.shape
        
#         # 2. 如果数据量太大，我们只抽样前 10000 条，或者如果总数小于 10000 就用全部
#         # 这样可以极大地提高运行速度，且不影响判断
#         sample_size = min(10000, shape[0])
#         sample_data = data[:sample_size]
        
#         # 3. 核心判断逻辑
#         try:
#             # 计算标准差，如果标准差为 0，说明所有数字都一模一样（比如全是 0，或者全是 1）
#             std_dev = np.std(sample_data)
            
#             if std_dev == 0:
#                 # 获取它唯一的那个值是什么
#                 first_val = np.reshape(sample_data, -1)[0]
#                 dead_keys.append((key, f"All elements are identical (Value: {first_val})", shape))
#             else:
#                 meaningful_keys.append((key, shape))
                
#         except TypeError:
#             # 如果是字符串或者离散的类别标签（无法计算标准差），我们看 unique 的数量
#             unique_vals = np.unique(sample_data)
#             if len(unique_vals) <= 1:
#                 dead_keys.append((key, f"All text/labels are identical (Value: {unique_vals})", shape))
#             else:
#                 meaningful_keys.append((key, f"{shape} (Categorical/Text, {len(unique_vals)} unique values)"))

# # --- 打印报告 ---
# print("\n🟢 【有实质内容的数据 Key】(数据在不断变化，含有真实信息):")
# print("-" * 60)
# for key, info in meaningful_keys:
#     print(f" 📂 {key:<30} | Shape/Info: {info}")

# print("\n🔴 【无意义/死数据 Key】(全为0、全为固定值或空值):")
# print("-" * 60)
# for key, reason, shape in dead_keys:
#     shape_str = str(shape) if shape else "N/A"
#     print(f" ❌ {key:<30} | Shape: {shape_str:<15} | 原因: {reason}")
    
# print("=" * 60)

# with h5py.File(DATASET_PATH, "r") as f:
#     print("Dataset keys:", list(f.keys()))
#     for key in f.keys():
#         print(f"\nKey: {key}")
#         print(f"Shape: {f[key].shape}")
#         try:
#             print(f"Sample data:\n{f[key][0]}")
#         except Exception as e:
#             print(f"Could not read sample data for key '{key}': {e}")
#             continue

#     # ---- Episode metadata ----
#     ep_id = 0
#     offset = f["ep_offset"][ep_id]
#     length = f["ep_len"][ep_id]

#     print(f"Episode {ep_id}: offset={offset}, length={length}")

#     # ---- Extract sequence ----
#     pixels = f["pixels"][offset:offset+length]  # (T, H, W, C)
#     actions = f["action"][offset:offset+length]

#     print("Pixels shape:", pixels.shape)
#     print("Actions shape:", actions.shape)




    # pixels = f["pixels"]
    # ep_offset = f["ep_offset"][:]
    # ep_len = f["ep_len"][:]

    # episode = 0
    # start = int(ep_offset[episode])
    # length = int(ep_len[episode])

    # print("Episode", episode, "start", start, "length", length)

    # # Save the first frame and a small sequence of frames.
    # for i in range(min(5, length)):
    #     frame = pixels[start + i].astype("uint8")
    #     frame_path = OUTPUT_DIR / f"episode_{episode:03d}_frame_{i}.png"
    #     plt.imsave(frame_path, frame)
    #     print(f"Saved frame {i} to {frame_path}")

    # print(f"Saved {min(5, length)} frames to {OUTPUT_DIR}")