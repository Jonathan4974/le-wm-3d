import h5py
from scipy.spatial.transform import Rotation
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


DS_PATH = "/home/student/data/ogbench/datasets/ogbench/cube_single_expert.h5"

WINDOW = 32

results = []

with h5py.File(DS_PATH, "r") as f:


    print("Keys: %s" % f.keys())

    obs = f["observation"]
    episode_ids = f["ep_idx"][:]   # shape (N,)

    print(np.unique(episode_ids[:30]))

    N = len(obs)

    sanity_checks = 0

    for start in range(N - WINDOW + 1):

        # Don't allow windows crossing episode boundaries
        if episode_ids[start] != episode_ids[start + WINDOW - 1]:
            continue

        clip = obs[start:start + WINDOW]

        # -----------------------------
        # Extract quantities
        # -----------------------------

        ee_position = clip[:, 12:15]
        ee_yaw = clip[:, 15:17]

        block_position = clip[:, 19:22]
        block_quaternion = clip[:, 22:26]
        block_yaw = clip[:, 26:28]

        block_translation = np.linalg.norm(
            np.diff(block_position, axis=0),
            axis=1,
        ).sum()

        ee_translation = np.linalg.norm(
            np.diff(ee_position, axis=0),
            axis=1,
        ).sum()

        cube_angles = []

        for q1, q2 in zip(block_quaternion[:-1], block_quaternion[1:]):

            r1 = Rotation.from_quat(q1)
            r2 = Rotation.from_quat(q2)

            relative = r2 * r1.inv()

            cube_angles.append(relative.magnitude())

        block_rotation = np.sum(cube_angles)

        ee_angles = np.arctan2(ee_yaw[:, 1], ee_yaw[:, 0])

        delta = np.diff(ee_angles)

        # unwrap around ±π
        delta = (delta + np.pi) % (2 * np.pi) - np.pi

        ee_rotation = np.abs(delta).sum()

        block_score = (
            np.rad2deg(block_rotation)
            /
            (100 * block_translation + 1e-6)
        )

        ee_score = (
            np.rad2deg(ee_rotation)
            /
            (100 * ee_translation + 1e-6)
        )


        score = {
            "start": start,
            "episode": episode_ids[start],

            "block_translation": block_translation,
            "block_rotation": block_rotation,

            "ee_translation": ee_translation,
            "ee_rotation": ee_rotation,

            "block_score": block_score,
            "ee_score": ee_score,
        }
        results.append(score)

        if block_rotation > np.deg2rad(5) and sanity_checks < 10:
            print("=" * 60)
            print(f"Episode: {episode_ids[start]}")
            print(f"Start:   {start}")

            print(f"Block translation : {block_translation:.5f} m")
            print(f"Block rotation    : {np.rad2deg(block_rotation):.2f} deg")

            print(f"EE translation    : {ee_translation:.5f} m")
            print(f"EE rotation       : {np.rad2deg(ee_rotation):.2f} deg")

            print(f"Block score       : {score['block_score']:.2f}")
            print(f"EE score          : {score['ee_score']:.2f}")

            print("Block positions:")
            print(block_position)

            print("EE positions:")
            print(ee_position)

            print("Block quaternions:")
            print(block_quaternion)

            print("Block quaternions norm:")
            print(np.linalg.norm(block_quaternion, axis=1))

            print("Incremental cube rotations:")
            print(np.rad2deg(cube_angles))

            print("EE yaw angles:")
            print(np.rad2deg(ee_angles))


            sanity_checks += 1 
            




    df = pd.DataFrame(results)

    print(df.describe())

    print(df.nlargest(
        10,
        "block_score"
    )[[
        "episode",
        "start",
        "block_translation",
        "block_rotation",
        "block_score"
    ]])


    plt.hist(df["block_score"], bins=100)
    plt.xlabel("Rotation / Translation")
    plt.ylabel("Count")

    plt.title("Block rotation / translation ratio")
    plt.savefig("block_rotation_translation_ratio.png", dpi=300)

    plt.scatter(
    df["block_translation"],
    np.rad2deg(df["block_rotation"]),
        s=2,
        alpha=0.3,
    )
    plt.xlabel("Translation (m)")
    plt.ylabel("Rotation (deg)")
    plt.title("Block rotation vs translation")
    plt.savefig("block_rotation_vs_translation.png", dpi=300)


    df["block_rotation_deg"] = np.rad2deg(df["block_rotation"])
    df["ee_rotation_deg"] = np.rad2deg(df["ee_rotation"])

    print("\nRotation statistics")
    print(df["block_rotation_deg"].describe())

    thresholds = [1, 5, 10, 20, 30, 45, 60, 90]

    print("\nNumber of windows above rotation thresholds")
    for t in thresholds:
        n = (df["block_rotation_deg"] >= t).sum()
        print(f">{t:2d}° : {n:8d} ({100*n/len(df):5.2f}%)")


    df["rot_per_meter"] = (
        df["block_rotation_deg"] /
        (df["block_translation"] + 1e-6)
    )

    rotation_dataset = df[
        (df.block_rotation_deg > 20) &
        (df.block_translation < 0.5)
    ]

    print(rotation_dataset.describe())
    print(len(rotation_dataset))

    print(df[["block_translation", "block_rotation_deg"]].corr())

    print(
        df[["block_translation", "block_rotation_deg"]]
        .corr(method="spearman")
    )

    rotation_thresholds = [5,10,20,30,45]
    translation_thresholds = [0.25,0.5,1.0,2.0]

    for r in rotation_thresholds:
        print(f"\nRotation > {r}°")

        for t in translation_thresholds:

            n = (
                (df.block_rotation_deg > r) &
                (df.block_translation < t)
            ).sum()

            print(f"translation < {t:4.2f} m : {n:8d}")

    rotation_thresholds = [10, 20, 30, 45]
    translation_thresholds = [0.25, 0.5, 1.0]

    for r in rotation_thresholds:
        print(f"\nRotation > {r}°")

        for t in translation_thresholds:

            subset = df[
                (df.block_rotation_deg > r) &
                (df.block_translation < t)
            ]

            print(
                f"translation < {t:4.2f} m : "
                f"{len(subset):6d} windows "
                f"({subset['episode'].nunique():4d} episodes)"
            )
            print(
                subset["block_rotation_deg"].mean(),
                subset["block_translation"].mean()
            )

            subset.to_csv(
                f"rotation_subset_{r}_{t}.csv",
                index=False,
            )