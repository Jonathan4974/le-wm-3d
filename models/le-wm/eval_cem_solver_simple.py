import os
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from utils import instantiate_world_model

os.environ["MUJOCO_GL"] = "egl"

SRC_QPOS_0 = np.array([
    -1.8570e+00,
    -1.5349e+00,
    2.1462e+00,
    -2.2110e+00,
    -1.5708e+00,
    2.7199e+00,
    4.4934e-01,
    5.8575e-04,
    4.4669e-01,
    -4.4292e-01,
    4.4929e-01,
    2.9956e-04,
    4.4632e-01,
    -4.3992e-01,
    4.7053e-01,
    4.2916e-03,
    1.0841e-01,
    -7.5304e-01,
    -1.4355e-02,
    1.7852e-02,
    6.5758e-01,
], dtype=np.float32)

SRC_QVEL_0 = np.array([
    7.3992e-02,
    4.4032e-01,
    4.0368e-01,
    -6.6603e-01,
    1.0991e-03,
    1.5137e-01,
    -4.6606e-04,
    1.4755e-04,
    6.1730e-04,
    -1.4372e-03,
    -7.2293e-04,
    -7.0954e-04,
    -2.5147e-03,
    6.0789e-03,
    -5.2453e-02,
    4.6213e-02,
    -2.9620e-01,
    -1.9350e-01,
    7.8925e-02,
    -7.7470e-02,
], dtype=np.float32)


def make_ogbench_env(depth: bool = False):
    return gym.make(
        "visual-cube-single-v0",
        terminate_at_goal=False,
        mode="data_collection",
        width=224,
        height=224,
        pixel_transparent_arm=False,
    )


def render_frame(env, depth: bool = False):
    img = env.unwrapped.render(camera="front_pixels", depth=depth)
    if img.dtype in [np.float32, np.float64]:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    return img


def preprocess_frame(img, depth: bool = False):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = np.clip(img, 0.0, 1.0).astype(np.float32)

    if not depth:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)

    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)


def pixel_mse(img_a, img_b):
    a = np.asarray(img_a).astype(np.float32) / 255.0
    b = np.asarray(img_b).astype(np.float32) / 255.0
    return np.mean((a - b) ** 2)


def build_goal_image(action_block: int = 5, depth: bool = False):
    goal_env = make_ogbench_env(depth=depth)
    goal_env.reset()
    goal_env.unwrapped.set_state(SRC_QPOS_0, SRC_QVEL_0)

    action_sequence = np.stack(
        [goal_env.action_space.sample().astype(np.float32) for _ in range(action_block)],
        axis=0,
    )

    for action in action_sequence:
        goal_env.step(action)

    goal_img = render_frame(goal_env, depth=depth)
    goal_env.close()
    return goal_img, action_sequence


def actions_to_tensor(actions: np.ndarray):
    flat = actions.reshape(1, 1, -1)
    return torch.from_numpy(flat.astype(np.float32))


def simple_cem_plan(
    model,
    current_pixels,
    goal_pixels,
    action_space,
    action_block: int,
    num_samples: int = 128,
    topk: int = 16,
    num_iters: int = 5,
    device: str = "cuda",
    cfg=None,
):
    current_pixels = current_pixels.to(device)
    goal_pixels = goal_pixels.to(device)

    action_dim = int(np.prod(action_space.shape))
    total_dim = action_dim * action_block
    low = torch.from_numpy(np.repeat(action_space.low.astype(np.float32), action_block)).to(device)
    high = torch.from_numpy(np.repeat(action_space.high.astype(np.float32), action_block)).to(device)

    mean = torch.zeros(total_dim, device=device)
    std = torch.ones(total_dim, device=device) * 0.5
    best_cost = float("inf")
    best_action = mean.clone()

    history_size = int(getattr(cfg.wm, "history_size", 1)) if cfg is not None else 1
    num_preds = int(getattr(cfg.wm, "num_preds", 1)) if cfg is not None else 1

    for _ in range(num_iters):
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(num_samples, total_dim, device=device)
        samples = torch.max(torch.min(samples, high.unsqueeze(0)), low.unsqueeze(0))
        sample_actions = samples.view(num_samples, 1, total_dim)

        info = {
            "pixels": current_pixels.repeat(num_samples, 1, 1, 1, 1),
            "goal": goal_pixels.repeat(num_samples, 1, 1, 1, 1),
            "action": sample_actions,
        }

        with torch.no_grad():
            output = model.encode(info)
            ctx_emb = output["emb"][:, :history_size]
            ctx_act = output["act_emb"][:, :history_size]
            tgt_emb = output["emb"][:, num_preds:]
            pred_emb = model.predict(ctx_emb, ctx_act)
            cost = ((pred_emb - tgt_emb).pow(2).flatten(1).mean(dim=1)).cpu()

        best_idx = cost.argmin().item()
        if cost[best_idx].item() < best_cost:
            best_cost = cost[best_idx].item()
            best_action = samples[best_idx].detach().clone()

        elite_idx = cost.argsort()[:topk]
        elite = samples[elite_idx]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0).clamp(min=1e-3)

    return best_action.view(1, 1, total_dim), best_cost


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    ckpt_path = cfg.get("ckpt_path") or os.getenv(
        "EVAL_CEM_CKPT_PATH",
        "/home/student/users/Aaron_workspace/le-wm-3DGeom/models/le-wm/outputs/checkpoints/ogbench/ogbench/cube_overfit_1_sources_1000_traj_1_actionBlocks_RGB/weights_20260701_123855.pt",
    )

    output_dir = Path.cwd() / "eval_cem_simple" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    depth = False
    env = make_ogbench_env(depth=depth)
    env.reset()
    env.unwrapped.set_state(SRC_QPOS_0, SRC_QVEL_0)

    current_img = render_frame(env, depth=depth)
    Image.fromarray(current_img).save(output_dir / "current_start.png")

    goal_img, goal_actions = build_goal_image(action_block=5, depth=depth)
    Image.fromarray(goal_img).save(output_dir / "goal.png")
    np.save(output_dir / "goal_actions.npy", goal_actions)

    print("Loading world model from", ckpt_path)
    world_model = instantiate_world_model(cfg)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    world_model.load_state_dict(state_dict, strict=True)
    world_model.eval().to("cuda")

    current_pixels = preprocess_frame(current_img, depth=depth)
    goal_pixels = preprocess_frame(goal_img, depth=depth)

    action_block = 5
    max_cycles = int(cfg.get("max_cycles", 10))
    goal_threshold = float(cfg.get("goal_threshold", 1e-3))

    total_action_steps = 0
    goal_reached = False
    plan_count = 0

    while plan_count < max_cycles and not goal_reached:
        plan_count += 1
        plan, cost = simple_cem_plan(
            world_model,
            current_pixels,
            goal_pixels,
            env.action_space,
            action_block=action_block,
            num_samples=int(cfg.get("cem_num_samples", 128)),
            topk=int(cfg.get("cem_topk", 16)),
            num_iters=int(cfg.get("cem_iters", 5)),
            device="cuda",
            cfg=cfg,
        )

        action_seq = plan.view(action_block, -1).cpu().numpy()
        print(f"Cycle {plan_count}: sampled best cost {cost:.6f}, executing {action_block} actions")

        for step_idx, action in enumerate(action_seq, start=1):
            _, _, terminated, truncated, _ = env.step(action)
            total_action_steps += 1
            if terminated or truncated:
                break

        current_img = render_frame(env, depth=depth)
        Image.fromarray(current_img).save(output_dir / f"cycle_{plan_count}_post.png")
        current_pixels = preprocess_frame(current_img, depth=depth)

        mse = pixel_mse(current_img, goal_img)
        print(f"Cycle {plan_count}: pixel mse to goal = {mse:.6f}")
        if mse < goal_threshold:
            goal_reached = True
            print("Goal reached at cycle", plan_count)
            break

    print("Finished planning.")
    print("plans executed:", plan_count)
    print("total real environment action steps:", total_action_steps)
    print("goal reached:", goal_reached)
    print("Saved outputs to", output_dir)


if __name__ == "__main__":
    run()
