import re
import time
from collections.abc import Callable, Iterable
from typing import Any
from collections.abc import Sequence

import gymnasium as gym
import mujoco
import numpy as np
from scipy.ndimage import zoom

from stable_worldmodel.utils import get_in


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
        """
        Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix.
        """
        if ext.shape == (4, 4):
            return ext
        if ext.shape == (3, 4):
            H = np.eye(4, dtype=ext.dtype)
            H[:3, :4] = ext
            return H
        raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")

def _depths_to_world_points_with_colors(
        depth: np.ndarray,
        K: np.ndarray,
        ext_w2c: np.ndarray,
        images_u8: np.ndarray | None = None,
        conf: np.ndarray | None = None,
        conf_thr: float = 0.0,
        pose: str = "GLB",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        For each frame, transform (u,v,1) through K^{-1} to get rays,
        multiply by depth to camera frame, then use (w2c)^{-1} to transform to world frame.
        Simultaneously extract colors.
        """
        H, W = depth.shape
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        ones = np.ones_like(us)
        pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W,3)
        
        depth = depth  # (H,W)
        valid = np.isfinite(depth) & (depth > 0)
        if conf is not None:
            valid &= conf >= conf_thr
        if not np.any(valid):
            return np.zeros((H, W, 3), dtype=np.float32)

        d_flat = depth.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(K)  # (3,3)
        c2w = np.linalg.inv(_as_homogeneous44(ext_w2c))  # (4,4)

        rays = K_inv @ pix[vidx].T  # (3,M)
        Xc = rays * d_flat[vidx][None, :]  # (3,M)
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        
        ### SWITCH: 
        if pose == "GLB":
            Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  ### GLOBAL SPACE            !!! temporarily disable global shift
        elif pose == "CAM":
            Xw = Xc.T.astype(np.float32)                ### CAM SPACE
        else:
            raise ValueError(f"Unknown pose type: {pose}")

        # cols = images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8)  # (M,3)

        # recover shape (H,W,3)
        point_map_hw = np.zeros((H, W, 3), dtype=np.float32)
        point_map_hw[valid] = Xw

        # color_map_hw = np.zeros((H, W, 3), dtype=np.uint8)
        # color_map_hw[valid] = cols

        # attention_mask = valid # (H,W)

        # return point_map_hw, color_map_hw, attention_mask
        return point_map_hw
    
def _depth_to_normals_with_intrinsics(depth, K):
        """
        depth : (H,W)
        K     : (3,3)

        returns
            normals : (H,W,3)
        """

        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]

        H, W = depth.shape

        u, v = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
            indexing="xy",
        )

        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth

        V = np.stack((X, Y, Z), axis=-1)

        normals = np.zeros_like(V)

        dx = V[:,2:] - V[:,:-2]
        dy = V[2:,:] - V[:-2,:]

        n = np.cross(dx[1:-1], dy[:,1:-1])

        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8

        normals[1:-1,1:-1] = n

        return normals
    
def _get_camera_intrinsic(model, camera_name, width, height):
        """
        Return camera intrinsic matrix K (3, 3).
        """

        cam_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )

        fovy = model.cam_fovy[cam_id]

        fy = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
        fx = fy

        cx = width / 2.0
        cy = height / 2.0

        K = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1 ],
        ], dtype=np.float32)

        return K
    
def _get_camera_extrinsic(model, data, camera_name):
        """
        World -> Camera extrinsic matrix (4,4).
        """

        cam_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )

        pos = data.cam_xpos[cam_id].copy()

        # camera rotation matrix (camera -> world)
        R_c2w = data.cam_xmat[cam_id].reshape(3, 3).copy()

        # world -> camera
        R_cv_mj = np.diag([1, -1, -1])

        R_w2c = R_cv_mj @ R_c2w.T
        t_w2c = -R_w2c @ pos

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R_w2c
        T[:3, 3] = t_w2c

        return T 
    
def _extract_data(env, camera_name):
        # rgb
        rgb = env.unwrapped.render(
            camera=camera_name
        )

        H, W = rgb.shape[:2]

        # camera intrinsics
        K = _get_camera_intrinsic(
            env.unwrapped.model,
            camera_name,
            W,
            H,
        )

        # camera extrinsics
        T = _get_camera_extrinsic(
            env.unwrapped.model,
            env.unwrapped.data,
            camera_name,
        )

        depth = env.unwrapped.render(
            camera=camera_name,
            depth=True,
        )
        return rgb, K, T, depth

def _resize_depth_map_scipy(depth_map: np.ndarray, target_shape: tuple) -> np.ndarray:
    zoom_factors = (target_shape[0] / depth_map.shape[0], target_shape[1] / depth_map.shape[1])
    return zoom(depth_map, zoom_factors, order=1)


class EnsureInfoKeysWrapper(gym.Wrapper):
    """Validates that required keys are present in the info dict."""

    def __init__(self, env: gym.Env, required_keys: Iterable[str]):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            required_keys: Iterable of regex patterns that must match keys in info.
        """
        super().__init__(env)
        self._patterns: list[re.Pattern] = []
        for k in required_keys:
            self._patterns.append(re.compile(k))

    def _check(self, info: dict, where: str) -> None:
        """Check if all required patterns have at least one match in info.

        Args:
            info: The info dictionary to check.
            where: String indicating where the check is performed (e.g., "reset").

        Raises:
            RuntimeError: If any required pattern is missing from info.
        """
        keys = list(info.keys())
        missing = [
            p.pattern
            for p in self._patterns
            if not any(p.fullmatch(k) for k in keys)
        ]
        if missing:
            raise RuntimeError(
                f'{where}: required info keys missing (patterns with no match): {missing}. Present keys: {keys}'
            )

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform environment step and validate info keys.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._check(info, 'step()')
        return obs, reward, terminated, truncated, info

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and validate info keys.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        obs, info = self.env.reset(*args, **kwargs)
        self._check(info, 'reset()')
        return obs, info


class EnsureImageShape(gym.Wrapper):
    """Validates that an image in the info dict has the expected spatial dimensions."""

    def __init__(
        self, env: gym.Env, image_key: str, image_shape: tuple[int, int]
    ):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            image_key: Key in info dict containing the image.
            image_shape: Expected (height, width) of the image.
        """
        super().__init__(env)
        self.image_key = image_key
        self.image_shape = image_shape  # (height, width)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and validate image shape.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.

        Raises:
            RuntimeError: If image shape does not match expected shape.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info[self.image_key].shape[:-1] != self.image_shape:
            raise RuntimeError(
                f'Image shape {info[self.image_key].shape} should be {self.image_shape}'
            )
        return obs, reward, terminated, truncated, info

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset and validate image shape.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.

        Raises:
            RuntimeError: If image shape does not match expected shape.
        """
        obs, info = self.env.reset(*args, **kwargs)
        if info[self.image_key].shape[:-1] != self.image_shape:
            raise RuntimeError(
                f'Image shape {info[self.image_key].shape} should be {self.image_shape}'
            )
        return obs, info


class EnsureGoalInfoWrapper(gym.Wrapper):
    """Validates that 'goal' key is present in info dict."""

    def __init__(
        self, env: gym.Env, check_reset: bool, check_step: bool = False
    ):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            check_reset: Whether to check 'goal' presence on reset.
            check_step: Whether to check 'goal' presence on each step.
        """
        super().__init__(env)
        self.check_reset = check_reset
        self.check_step = check_step

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset and validate goal presence.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.

        Raises:
            RuntimeError: If 'goal' is missing and check_reset is True.
        """
        obs, info = self.env.reset(*args, **kwargs)
        if self.check_reset and 'goal' not in info:
            raise RuntimeError(
                "The info dict returned by reset() must contain the key 'goal'."
            )
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and validate goal presence.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.

        Raises:
            RuntimeError: If 'goal' is missing and check_step is True.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.check_step and 'goal' not in info:
            raise RuntimeError(
                "The info dict returned by step() must contain the key 'goal'."
            )
        return obs, reward, terminated, truncated, info


class EverythingToInfoWrapper(gym.Wrapper):
    """Moves all transition information into the info dict."""

    def __init__(self, env: gym.Env):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        self._variations_watch: Sequence[str] = []
        self._step_counter = 0
        self._id = 0

    def _gen_id(self) -> int:
        """Generate a random unique identifier for the current episode.

        Returns:
            A random 64-bit integer.
        """
        max_int = np.iinfo(np.int64).max
        rng = self.env.unwrapped.np_random
        return int(
            rng.integers(0, max_int)
            if hasattr(rng, 'integers')
            else rng.randint(0, max_int)
        )

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and move all data to info.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        self._step_counter = 0
        obs, info = self.env.reset(*args, **kwargs)
        if not isinstance(obs, dict):
            _obs = {'observation': obs}
        else:
            _obs = obs

        for key, val in _obs.items():
            assert key not in info
            info[key] = val

        assert 'reward' not in info
        info['reward'] = np.nan
        assert 'terminated' not in info
        info['terminated'] = False
        assert 'truncated' not in info
        info['truncated'] = False
        assert 'action' not in info
        info['action'] = self.env.action_space.sample()
        assert 'step_idx' not in info
        info['step_idx'] = self._step_counter
        assert 'id' not in info
        self._id = self._gen_id()
        info['id'] = self._id

        # add all variations to info if needed
        options = kwargs.get('options') or {}

        if 'variation' in options:
            var_opt = options['variation']
            assert isinstance(var_opt, list | tuple), (
                'variation option must be a list or tuple containing variation names to sample, found: '
                f'{type(var_opt)}'
            )
            if len(var_opt) == 1 and var_opt[0] == 'all':
                self._variations_watch = (
                    self.env.unwrapped.variation_space.names()
                )
            else:
                self._variations_watch = var_opt

        for key in self._variations_watch:
            var_key = f'variation.{key}'
            assert var_key not in info
            subvar_space = get_in(
                self.env.unwrapped.variation_space, key.split('.')
            )
            info[var_key] = subvar_space.value

        if isinstance(info['action'], dict):
            raise NotImplementedError
        else:
            info['action'] = np.full_like(info['action'], np.nan)
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and move all data to info.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_counter += 1
        if not isinstance(obs, dict):
            _obs = {'observation': obs}
        else:
            _obs = obs
        for key, val in _obs.items():
            assert key not in info
            info[key] = val
        assert 'reward' not in info
        info['reward'] = reward
        assert 'terminated' not in info
        info['terminated'] = bool(terminated)
        assert 'truncated' not in info
        info['truncated'] = bool(truncated)
        assert 'action' not in info
        info['action'] = action
        assert 'step_idx' not in info
        info['step_idx'] = self._step_counter
        assert 'id' not in info
        info['id'] = self._id

        for key in self._variations_watch:
            var_key = f'variation.{key}'
            assert var_key not in info
            subvar_space = get_in(
                self.env.unwrapped.variation_space, key.split('.')
            )
            info[var_key] = subvar_space.value

        return obs, reward, terminated, truncated, info


class MapKeysWrapper(gym.Wrapper):
    """Renames keys in the info dict according to a mapping.

    Useful when an env's observation is lifted into info under a name that
    downstream code does not expect. For example, with ``add_pixels=False``
    a raw pixel observation lands in ``info['observation']``; mapping
    ``{'observation': 'pixels'}`` makes it visible to the rest of the
    pipeline (eval video logging, goal handling, ...).
    """

    def __init__(self, env: gym.Env, key_map: dict[str, str]):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            key_map: Mapping from source key to destination key. Each source
                key is removed from info and re-added under its destination
                name.
        """
        super().__init__(env)
        self.key_map = dict(key_map)

    def _remap(self, info: dict) -> dict:
        """Rename mapped keys in info.

        Args:
            info: The info dictionary to modify.

        Returns:
            The same info dictionary with keys renamed.

        Raises:
            KeyError: If a source key is absent from info.
        """
        for src, dst in self.key_map.items():
            if src not in info:
                raise KeyError(
                    f'MapKeysWrapper: key {src!r} not found in info; '
                    f'present keys: {list(info.keys())}'
                )
            info[dst] = info.pop(src)
        return info

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and remap info keys.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        obs, info = self.env.reset(*args, **kwargs)
        return obs, self._remap(info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and remap info keys.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._remap(info)


class AddPixelsWrapper(gym.Wrapper):
    """Adds rendered environment pixels_RGB to info dict for visualization."""

    def __init__(
        self,
        env: gym.Env,
        pixels_shape: tuple[int, int] = (84, 84),  # (height, width)
        torchvision_transform: Callable[[Any], Any] | None = None,
        resample: int | None = None,
        camera_name: list[str] | None = None,
    ):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            pixels_shape: Target (height, width) for rendered pixels.
            torchvision_transform: Optional transform to apply to the pixels.
            resample: PIL resample filter (e.g. ``Image.BILINEAR``,
                ``Image.NEAREST``). Defaults to BILINEAR.
        """
        super().__init__(env)
        self.pixels_shape = pixels_shape
        self.torchvision_transform = torchvision_transform
        self.cam_name = camera_name
        # For resizing, use PIL (required for torchvision transforms)
        from PIL import Image

        self.Image = Image
        self.resample = resample if resample is not None else Image.BILINEAR

    def _get_pixels(self) -> tuple[dict[str, np.ndarray], float]:
        """Render environment and process pixels.

        Returns:
            A tuple of (pixels dictionary, render time).
        """
        # Render the environment as an RGB array
        render = getattr(self.env.unwrapped, 'render_multiview', None)
        render_fn = render if callable(render) else self.env.render

        t0 = time.time()
        if self.cam_name != None:
            img = {f"{cam}": render_fn(camera=cam) for cam in self.cam_name}
        else:
            img = render_fn()
        t1 = time.time()

        def _process_img(img_array: np.ndarray) -> np.ndarray:
            # Convert to PIL Image for resizing
            pil_img = self.Image.fromarray(img_array)
            height, width = self.pixels_shape
            pil_img = pil_img.resize((width, height), self.resample)
            # Optionally apply torchvision transform
            if self.torchvision_transform is not None:
                pixels = self.torchvision_transform(pil_img)
            else:
                pixels = np.array(pil_img)
            return pixels

        if isinstance(img, dict):
            pixels = {f'pixels_rgb.{k}': _process_img(v) for k, v in img.items()}
        elif isinstance(img, (list | tuple)):
            pixels = {
                f'pixels_rgb.{i}': _process_img(v) for i, v in enumerate(img)
            }
        else:
            pixels = {'pixels_rgb': _process_img(img)}

        return pixels, t1 - t0


class AddRGBWrapper(gym.Wrapper):
    """Adds rendered environment pixels to info dict."""

    def __init__(
        self,
        env: gym.Env,
        pixels_shape: tuple[int, int] = (84, 84),  # (height, width)
        torchvision_transform: Callable[[Any], Any] | None = None,
        resample: int | None = None,
        camera_name: list[str] | None = None,
    ):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            pixels_shape: Target (height, width) for rendered pixels.
            torchvision_transform: Optional transform to apply to the pixels.
            resample: PIL resample filter (e.g. ``Image.BILINEAR``,
                ``Image.NEAREST``). Defaults to BILINEAR.
        """
        super().__init__(env)
        self.pixels_shape = pixels_shape
        self.torchvision_transform = torchvision_transform
        self.cam_name = camera_name
        # For resizing, use PIL (required for torchvision transforms)
        from PIL import Image

        self.Image = Image
        self.resample = resample if resample is not None else Image.BILINEAR

    def _get_pixels(self) -> tuple[dict[str, np.ndarray], float]:
        """Render environment and process pixels.

        Returns:
            A tuple of (pixels dictionary, render time).
        """
        # Render the environment as an RGB array
        render = getattr(self.env.unwrapped, 'render_multiview', None)
        render_fn = render if callable(render) else self.env.render

        t0 = time.time()
        if self.cam_name != None:
            img = {f"{cam}": render_fn(camera=cam) for cam in self.cam_name}
        else:
            img = render_fn()
        t1 = time.time()

        def _process_img(img_array: np.ndarray) -> np.ndarray:
            # Convert to PIL Image for resizing
            pil_img = self.Image.fromarray(img_array)
            height, width = self.pixels_shape
            pil_img = pil_img.resize((width, height), self.resample)
            # Optionally apply torchvision transform
            if self.torchvision_transform is not None:
                pixels = self.torchvision_transform(pil_img)
            else:
                pixels = np.array(pil_img)
            return pixels

        if isinstance(img, dict):
            pixels = {f'pixels.{k}': _process_img(v) for k, v in img.items()}
        elif isinstance(img, (list | tuple)):
            pixels = {
                f'pixels.{i}': _process_img(v) for i, v in enumerate(img)
            }
        else:
            pixels = {'pixels': _process_img(img)}

        return pixels, t1 - t0

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and add pixels to info.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        obs, info = self.env.reset(*args, **kwargs)
        pixels, info['render_time'] = self._get_pixels()
        info.update(pixels)
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and add pixels to info.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        pixels, info['render_time'] = self._get_pixels()
        info.update(pixels)
        return obs, reward, terminated, truncated, info


class AddNormalWrapper(gym.Wrapper):
    """Adds rendered environment Normal map to info dict as 'pixels'."""

    def __init__(
        self,
        env: gym.Env,
        pixels_shape: tuple[int, int] = (84, 84),  # (height, width)
        torchvision_transform: Callable[[Any], Any] | None = None,
        resample: int | None = None,
        camera_name: list[str] | None = None,
    ):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            pixels_shape: Target (height, width) for rendered pixels.
            torchvision_transform: Optional transform to apply to the pixels.
            resample: PIL resample filter (e.g. ``Image.BILINEAR``,
                ``Image.NEAREST``). Defaults to BILINEAR.
        """
        super().__init__(env)
        self.pixels_shape = pixels_shape
        self.torchvision_transform = torchvision_transform
        self.cam_name = camera_name
        # For resizing, use PIL (required for torchvision transforms)
        from PIL import Image

        self.Image = Image
        self.resample = resample if resample is not None else Image.BILINEAR

    def _get_pixels(self) -> tuple[dict[str, np.ndarray], float]:
        """Render environment and process normal map

        Returns:
            A tuple of (pixels dictionary, render time).
        """
        # Render the environment        
        t0 = time.time()
        if self.cam_name != None:
            K = {}
            depth = {}
            for cam in self.cam_name:
                _, K[cam], _, dep = _extract_data(self.env, cam)
                depth[cam] = _resize_depth_map_scipy(np.clip(dep, 0.5, 3.0), self.pixels_shape)
        else:
            _, K, _, depth = _extract_data(self.env, "front_pixels")
            depth = _resize_depth_map_scipy(np.clip(depth, 0.5, 3.0), self.pixels_shape)
        # img = render_fn()
        t1 = time.time()

        if isinstance(depth, dict):
            pixels = {
                f'pixels.{k}': _depth_to_normals_with_intrinsics(depth[k], K[k]) 
                for k in depth.keys()
            }
        elif isinstance(depth, (list | tuple)):
            pixels = {
                f'pixels.{i}': _depth_to_normals_with_intrinsics(dep, intr) 
                for i, (dep, intr) in enumerate(zip(depth, K))
            }
        else:
            pixels = {'pixels': _depth_to_normals_with_intrinsics(depth, K)}

        return pixels, t1 - t0

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and add pixels to info.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        obs, info = self.env.reset(*args, **kwargs)
        pixels, info['render_time'] = self._get_pixels()
        info.update(pixels)
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and add pixels to info.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        pixels, info['render_time'] = self._get_pixels()
        info.update(pixels)
        return obs, reward, terminated, truncated, info


class AddDepthWrapper(gym.Wrapper):
    """Adds rendered Depth-only observations to info["pixels"]."""

    def __init__(
        self,
        env: gym.Env,
        pixels_shape: tuple[int, int] = (84, 84),
        torchvision_transform=None,
        resample=None,
        camera_name=None,   # kept for API compatibility
    ):
        super().__init__(env)

        self.pixels_shape = pixels_shape
        self.torchvision_transform = torchvision_transform

        from PIL import Image
        self.Image = Image
        self.resample = (
            resample if resample is not None else Image.BILINEAR
        )

    def _get_pixels(self) -> tuple[dict[str, np.ndarray], float]:
        """Render depth and return a 4-channel RGBD image."""

        t0 = time.time()

        _ , _, _, depth = _extract_data(self.env, "front_pixels")

        # Resize + clip depth
        depth = _resize_depth_map_scipy(np.clip(depth, 0.5, 3.0),self.pixels_shape,)

        depth = depth[..., None]            # add channel dimension (will get expanded later)

        # Apply preprocessing (RGB + depth handled jointly)
        if self.torchvision_transform is not None:
            depth = self.torchvision_transform(depth)

        if not hasattr(self, "_printed"):
            self._printed = True
            print("\n\n === GET PIXELS DEBUG === ")
            print("Raw depth:")
            print("shape:", depth.shape)
            print("dtype:", depth.dtype)
            print("min/max:", np.nanmin(depth), np.nanmax(depth))

        t1 = time.time()

        return {"pixels": depth}, t1-t0

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        pixels, info["render_time"] = self._get_pixels()
        info.update(pixels)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pixels, info["render_time"] = self._get_pixels()
        info.update(pixels)
        return obs, reward, terminated, truncated, info


class AddRGBDWrapper(gym.Wrapper):
    """Adds rendered RGB+D observations to info["pixels"]."""

    def __init__(
        self,
        env: gym.Env,
        pixels_shape: tuple[int, int] = (84, 84),
        torchvision_transform=None,
        resample=None,
        camera_name=None,   # kept for API compatibility
    ):
        super().__init__(env)

        self.pixels_shape = pixels_shape
        self.torchvision_transform = torchvision_transform

        from PIL import Image
        self.Image = Image
        self.resample = (resample if resample is not None else Image.BILINEAR)

    def _get_pixels(self) -> tuple[dict[str, np.ndarray], float]:
        """Render RGB + depth and return a 4-channel RGBD image."""

        t0 = time.time()

        rgb, K, T, depth = _extract_data(self.env, "front_pixels")

        t1 = time.time()

        # Resize RGB
        h, w = self.pixels_shape
        rgb = np.asarray(self.Image.fromarray(rgb).resize((w, h), self.resample))

        # Resize + clip depth
        depth = _resize_depth_map_scipy(
            np.clip(depth, 0.5, 3.0),
            self.pixels_shape,
        )

        # Construct RGBD image
        rgbd = np.concatenate(
            [rgb, depth[..., None]],
            axis=-1,
        )

        # Apply preprocessing (RGB + depth handled jointly)
        if self.torchvision_transform is not None:
            rgbd = self.torchvision_transform(rgbd)

        return {"pixels": rgbd}, t1 - t0

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        pixels, info["render_time"] = self._get_pixels()
        info.update(pixels)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pixels, info["render_time"] = self._get_pixels()
        info.update(pixels)
        return obs, reward, terminated, truncated, info


class AddPointMapWrapper(gym.Wrapper): 
    """Adds rendered RGB+D observations to info["pixels"]."""
    
    def __init__(
        self,
        env: gym.Env,
        pixels_shape: tuple[int, int] = (84, 84),
        torchvision_transform=None,
        resample=None,
        camera_name=None,   # kept for API compatibility
    ):
        super().__init__(env)

        self.pixels_shape = pixels_shape
        self.torchvision_transform = torchvision_transform

        from PIL import Image
        self.Image = Image
        self.resample = (resample if resample is not None else Image.BILINEAR)

    
    def _get_pixels(self) -> tuple[dict[str, np.ndarray], float]:
        """Render DEPTH and return a 3D POINT MAP"""

        t0 = time.time()
        rgb, K, T, depth = _extract_data(self.env, "front_pixels")
        t1 = time.time()

        # Resize + clip depth
        depth = _resize_depth_map_scipy(np.clip(depth, 0.5, 3.0),self.pixels_shape,)

        # Get the point map 
        point_map_with_colors = _depths_to_world_points_with_colors(depth=depth, K=K, ext_w2c=T)

        # Apply preprocessing 
        if self.torchvision_transform is not None:
            point_map_with_colors = self.torchvision_transform(point_map_with_colors)

        return {"pixels": point_map_with_colors}, t1 - t0

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        pixels, info["render_time"] = self._get_pixels()
        info.update(pixels)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pixels, info["render_time"] = self._get_pixels()
        info.update(pixels)
        return obs, reward, terminated, truncated, info
    




class AddGeometryWrapper(gym.Wrapper):
    """Adds rendered required environment geometry prior to info dict."""

    def __init__(
        self,
        env: gym.Env,
        pixels_shape: tuple[int, int] = (84, 84),  # (height, width)
        torchvision_transform: Callable[[Any], Any] | None = None,
        resample: int | None = None,
        camera_name: list[str] | None = None,
        add_depths: bool = False,
        add_normals: bool = False,
        add_points: bool = False,
    ):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            pixels_shape: Target (height, width) for rendered pixels.
            torchvision_transform: Optional transform to apply to the pixels.
            resample: PIL resample filter (e.g. ``Image.BILINEAR``,
                ``Image.NEAREST``). Defaults to BILINEAR.
            camera_name: a List of camera name defined in the environment.
            add_depths: set True to add depth map. Default False
            add_normals: set True to add surface normal map. Default False
            add_points: set True to add depth map. Default False

            TODO: fit transformation to depth/normal/points map
            
        """
        super().__init__(env)
        self.pixels_shape = pixels_shape
        self.torchvision_transform = torchvision_transform
        self.cam_name = camera_name
        self.depths = add_depths
        self.normals = add_normals
        self.points = add_points
        # For resizing, use PIL (required for torchvision transforms)
        from PIL import Image

        self.Image = Image
        self.resample = resample if resample is not None else Image.BILINEAR

    

    def _get_geometry(self) -> tuple[dict[str, np.ndarray], float]:
        """Render environment and process geometry prior.

        Returns:
            A tuple of (pixels dictionary, render time).
        """
        # # Render the environment as an RGB array
        # render = getattr(self.env.unwrapped, 'render_multiview', None)
        # render_fn = render if callable(render) else self.env.render

        def _process_img(img_array: np.ndarray) -> np.ndarray:
            # Convert to PIL Image for resizing
            pil_img = self.Image.fromarray(img_array)
            height, width = self.pixels_shape
            pil_img = pil_img.resize((width, height), self.resample)
            # Optionally apply torchvision transform
            if self.torchvision_transform is not None:
                pixels = self.torchvision_transform(pil_img)
            else:
                pixels = np.array(pil_img)
            return pixels
        
        t0 = time.time()
        if self.cam_name != None:
            img = {}
            K = {}
            T = {}
            depth = {}
            for cam in self.cam_name:
                image, K[cam], T[cam], dep = _extract_data(self.env, cam)
                depth[cam] = _resize_depth_map_scipy(np.clip(dep, 0.5, 2.0), self.pixels_shape)
                img[cam] = _process_img(image)
        else:
            img, K, T, depth = _extract_data(self.env, "front_pixels")
            depth = _resize_depth_map_scipy(np.clip(depth, 0.5, 2.0), self.pixels_shape)
            img = _process_img(img)
        # img = render_fn()
        t1 = time.time()

        geometry = {}
        # get depth
        if self.depths:
            if isinstance(img, dict):
                geometry.update({f'pixels_depth.{k}': v for k, v in depth.items()})
            elif isinstance(img, (list | tuple)):
                geometry.update({f'pixels_depth.{i}': v for i, v in enumerate(depth)})
            else:
                geometry.update({'pixels_depth': depth})

        # get normal map
        if self.normals:
            if isinstance(img, dict):
                geometry.update({
                    f'pixels_normal.{k}': _depth_to_normals_with_intrinsics(depth[k], K[k]) 
                    for k in depth.keys()
                })
            elif isinstance(img, (list | tuple)):
                geometry.update({
                    f'pixels_normal.{i}': _depth_to_normals_with_intrinsics(dep, intr) 
                    for i, (dep, intr) in enumerate(zip(depth, K))
                })
            else:
                geometry.update({'pixels_normal': _depth_to_normals_with_intrinsics(depth, K)})

        # get point map
        if self.points:
            if isinstance(img, dict):
                geometry.update({
                    f'pixels_points.{k}': _depths_to_world_points_with_colors(depth[k], K[k], T[k]) 
                    for k in depth.keys()
                })
            elif isinstance(img, (list | tuple)):
                geometry.update({
                    f'pixels_points.{i}': _depths_to_world_points_with_colors(dep, intr, extr) 
                    for i, (dep, intr, extr) in enumerate(zip(depth, K, T))
                })
            else:
                geometry.update({'pixels_points': _depths_to_world_points_with_colors(depth, K, T)})

        return geometry, t1 - t0

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and add geometry prior to info.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        obs, info = self.env.reset(*args, **kwargs)
        geometry, info['render_time'] = self._get_geometry()
        info.update(geometry)
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and add geometry prior to info.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        geometry, info['render_time'] = self._get_geometry()
        info.update(geometry)
        return obs, reward, terminated, truncated, info


class ResizeGoalWrapper(gym.Wrapper):
    """Resizes goal images in info dict."""

    def __init__(
        self,
        env: gym.Env,
        pixels_shape: tuple[int, int] = (84, 84),  # (height, width)
        torchvision_transform: Callable[[Any], Any] | None = None,
        resample: int | None = None,
    ):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            pixels_shape: Target (height, width) for resizing goal images.
            torchvision_transform: Optional transform to apply to goal images.
            resample: PIL resample filter (e.g. ``Image.BILINEAR``,
                ``Image.NEAREST``). Defaults to BILINEAR.
        """
        super().__init__(env)
        self.pixels_shape = pixels_shape
        self.torchvision_transform = torchvision_transform
        # For resizing, use PIL (required for torchvision transforms)
        from PIL import Image

        self.Image = Image
        self.resample = resample if resample is not None else Image.BILINEAR

    def _format(self, img: np.ndarray) -> np.ndarray:
        """Resize and transform a goal image.

        Args:
            img: The original goal image as a numpy array.

        Returns:
            The processed goal image.
        """
        # Convert to PIL Image for resizing
        pil_img = self.Image.fromarray(img)
        height, width = self.pixels_shape
        pil_img = pil_img.resize((width, height), self.resample)
        # Optionally apply torchvision transform
        if self.torchvision_transform is not None:
            pixels = self.torchvision_transform(pil_img)
        else:
            pixels = np.array(pil_img)
        return pixels

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and format goal image.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        obs, info = self.env.reset(*args, **kwargs)
        if 'goal' in info:
            info['goal'] = self._format(info['goal'])
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and format goal image.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if 'goal' in info:
            info['goal'] = self._format(info['goal'])
        return obs, reward, terminated, truncated, info


_RESAMPLE_ALIASES = {
    'nearest': 'NEAREST',
    'bilinear': 'BILINEAR',
    'bicubic': 'BICUBIC',
    'lanczos': 'LANCZOS',
    'box': 'BOX',
    'hamming': 'HAMMING',
}


def _resolve_resample(resample: str | int | None) -> int | None:
    if resample is None or isinstance(resample, int):
        return resample
    from PIL import Image

    key = resample.lower()
    if key not in _RESAMPLE_ALIASES:
        raise ValueError(
            f'Unknown resample mode {resample!r}; '
            f'choose from {sorted(_RESAMPLE_ALIASES)}.'
        )
    return getattr(Image, _RESAMPLE_ALIASES[key])


class MegaWrapper(gym.Wrapper):
    """Combines multiple wrappers for comprehensive environment preprocessing."""

    def __init__(
        self,
        env: gym.Env,
        image_shape: tuple[int, int] = (84, 84),
        pixels_transform: Callable[[Any], Any] | None = None,
        goal_transform: Callable[[Any], Any] | None = None,
        required_keys: Iterable[str] | None = None,
        separate_goal: bool = True,
        image_resample: str | int | None = None,
        add_pixels: bool = True,
        add_geometry: bool = False,
        add_depths: bool = False,
        add_normals: bool = False,
        add_points: bool = False,
        camera_name: list[str] | None = None,
        modality: str = None,
    ):
        """Initialize the mega wrapper pipeline.

        Args:
            env: The environment to wrap.
            image_shape: Target (height, width) for all image processing.
            pixels_transform: Optional transform for rendered pixels.
            goal_transform: Optional transform for goal images.
            required_keys: Keys that must be present in info dict.
            separate_goal: Whether to handle goal separately.
            image_resample: PIL resample mode used when resizing pixels and
                goal images. Accepts a PIL constant or a string in
                ``{'nearest','bilinear','bicubic','lanczos','box','hamming'}``.
                Defaults to bilinear. Use ``'nearest'`` for crisp pixel art.
            add_pixels: If True (default), render the env and add a ``pixels``
                key (resized to ``image_shape``), require it, and resize goal
                images. Set False for envs with no pixel observations (e.g.
                audio); the raw observation is still lifted into info.
            modality: define which information will be saved in the info["pixels"]
                        "RGB": rgb image as (H, W, 3)
                        "NRM": surface normal map as (H, W, 3)
        """
        super().__init__(env)

        req_keys = list(required_keys) if required_keys is not None else []

        if add_pixels:
            req_keys.append(r'^pixels(?:\..*)?$')
            resample = _resolve_resample(image_resample)
            # this adds `pixels` key to info with optional transform
            # env = AddPixelsWrapper(
            #     env, image_shape, pixels_transform, resample, camera_name
            # )

            if modality == "RGB":
                env = AddRGBWrapper(
                    env, image_shape, pixels_transform, resample, camera_name
                )
            elif modality == "NRM":
                env = AddNormalWrapper(
                    env, image_shape, pixels_transform, resample, camera_name
                )
            elif modality == "RGB_D":
                env = AddRGBDWrapper(
                    env, image_shape, pixels_transform, resample, camera_name
                )
            elif modality == "DEPTH": 
                env = AddDepthWrapper(
                    env, image_shape, pixels_transform, resample, camera_name
                )
            elif modality == "POINTMAP":
                env = AddPointMapWrapper(
                    env, image_shape, pixels_transform, resample, camera_name
                )

            else: 
                print("Unknown modality: need to define a ModalityWrapper inside default.py first !")
                exit(1)

        if add_geometry:
            resample = _resolve_resample(image_resample)
            # this adds optional transform for pixels to info
            env = AddGeometryWrapper(
                env, image_shape, pixels_transform, resample, camera_name, add_depths, add_normals, add_points
            )              


        # this removes the info output, everything is in observation!
        env = EverythingToInfoWrapper(env)
        # check that necessary keys are in the observation
        env = EnsureInfoKeysWrapper(env, req_keys)

        if add_pixels:
            env = ResizeGoalWrapper(env, image_shape, goal_transform, resample)

        self.env = env
