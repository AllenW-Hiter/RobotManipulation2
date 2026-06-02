from __future__ import annotations

import logging
import os

from robosuite import macros

macros.IMAGE_CONVENTION = "opencv"
import dexmimicgen  # noqa: F401
import gymnasium as gym
import numpy as np
import robosuite
import torch
from robosuite import load_composite_controller_config

# 从规范环境名到对应机器人模型列表的映射
# 注意：添加新任务时，始终引用 *实际的* robosuite 环境
# 类名（即 `robosuite.make` 期望的名称）。
ENV_ROBOTS = {
    # ------------------------------------------------------------------
    # DexMimicGen 多臂任务
    # ------------------------------------------------------------------
    "TwoArmThreading": ["Panda", "Panda"],
    "TwoArmThreePieceAssembly": ["Panda", "Panda"],
    "TwoArmTransport": ["Panda", "Panda"],
    "TwoArmLiftTray": ["PandaDexRH", "PandaDexLH"],
    "TwoArmBoxCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmDrawerCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmCoffee": ["GR1FixedLowerBody"],
    "TwoArmPouring": ["GR1FixedLowerBody"],
    "TwoArmCanSortRandom": ["GR1ArmsOnly"],
    # ------------------------------------------------------------------
    # Robomimic benchmark 任务（除非特别说明，否则为单臂）
    # ------------------------------------------------------------------
    # Lift task – single Panda arm
    "Lift": ["Panda"],
    # Can task – implemented in robosuite as `PickPlaceCan`
    "PickPlaceCan": ["Panda"],
    # Square task – implemented in robosuite as `NutAssemblySquare`
    "NutAssemblySquare": ["Panda"],
}
# 创建命名 logger
logger = logging.getLogger(__name__)


class RobosuiteGymWrapper:
    """
    Gym-like wrapper for robosuite environments to make them compatible with the training script.

    Robosuite environments use the old gym API (step 返回 4 values) and have different
    observation/action interfaces, so this wrapper adapts them to work with our training loop.

    Idxs	Meaning (all values are per-time-step targets, sent at 20 Hz)
    0-2	Right wrist Δpos Cartesian x / y / z offset (metres) for the EE site gripper0_right_grip_site.
    3-5	Right wrist Δrot Axis-angle components rx,ry,rzrx,ry,rz; ‖r‖ = rotation angle (rad).
    6-11	Right Inspire-hand joints (joint-position targets, rad)
    6 Thumb flexion
    7 Thumb roll / opposition
    8 Index flexion
    9 Middle flexion
    10 Ring flexion
    11 Pinky flexion
    12-14	Left wrist Δpos Cartesian x / y / z offset for gripper0_left_grip_site.
    15-17	Left wrist Δrot Axis-angle components for left EE orientation.
    18-23	Left Inspire-hand joints (same ordering as right).
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int = 1,
        video_key: str = "observation.images.agentview",
        render_gpu_device_id: int = 0,
        camera_size: int = 84,
        render_size: int | None = None,
        env_id: int = 0,
        expected_image_keys: list[str] | None = None,
        seed: int | None = None,
    ):
        # ------------------------------------------------------------------
        # 允许 Robomimic 文献中常用的别名。
        # 这些别名会映射到实际的 robosuite 环境名。
        # ------------------------------------------------------------------
        alias_map = {
            # Robomimic papers / datasets refer to these tasks without the
            # full robosuite class name. We translate them here so that
            # callers can simply pass "Lift", "Can", "Square", or
            # "Transport" and things will work out of the box.
            "Can": "PickPlaceCan",
            "Square": "NutAssemblySquare",
            "Transport": "TwoArmTransport",
        }

        # 保留用户提供的原始名称，用于日志和启发式逻辑
        self.original_env_name = env_name
        # 解析为规范 robosuite 环境名（如果存在别名）
        env_name = alias_map.get(env_name, env_name)

        self.env_name = env_name
        self.num_envs = num_envs
        self.render_gpu_device_id = render_gpu_device_id
        self.camera_size = camera_size
        self.render_size = render_size if render_size is not None else (240, 320)
        self.env_id = env_id

        # ------------------------------------------------------------------
        # Episode horizon – 为部分长时 DexMimicGen 任务覆盖默认值.
        # For all other tasks we fall back to robosuite's default of 800. - Hongsuk. Oct 26, 2025.
        # ------------------------------------------------------------------
        self.horizon = {
            "TwoArmCoffee": 400,
            "TwoArmBoxCleanup": 400,
            "Lift": 100,
            "PickPlaceCan": 300,
            "NutAssemblySquare": 400,
            "TwoArmLiftTray": 1000,
            "TwoArmThreading": 400,
        }.get(env_name, 800)

        # 添加 Gymnasium 所需属性
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 20, "horizon": self.horizon}
        self.spec = None  # Not required for vectorization
        self.render_mode = "rgb_array"  # Default render mode for camera observations

        # 存储用于渲染的视频相机 key (will be set by create_vectorized_env)
        self.video_key = video_key
        logger.info(f"Video key: {self.video_key}")

        self.episode_steps = 0

        if env_name not in ENV_ROBOTS:
            raise ValueError(f"Unknown robosuite environment: {env_name}")

        robots = ENV_ROBOTS[env_name]

        # 获取该环境预期的图像 key
        # 如果提供自定义 key 则使用它们，否则回退到默认值
        if expected_image_keys is None:
            expected_image_keys = self._get_expected_image_keys(env_name)

        # 移除 '_image' 后缀以得到 robosuite 相机名
        camera_names = [key.replace("_image", "") for key in expected_image_keys]

        self.expected_image_keys = expected_image_keys  # Store for use in _process_obs
        logger.info(f"Using image observation keys: {expected_image_keys}")

        # 使用 robosuite.make() 创建环境
        env_kwargs = {
            "env_name": env_name,
            "robots": robots,
            "controller_configs": load_composite_controller_config(robot=robots[0]),
            "has_renderer": False,  # No rendering during training
            "has_offscreen_renderer": True,
            "ignore_done": False,
            "use_camera_obs": True,
            "control_freq": 20,
            "camera_names": camera_names,
            "camera_heights": self.camera_size,
            "camera_widths": self.camera_size,
            "horizon": self.horizon,
            "renderer": "mujoco",
            "render_gpu_device_id": self.render_gpu_device_id,
        }

        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(self.render_gpu_device_id)

        print(f"render_gpu_device_id: {self.render_gpu_device_id}")

        # NOTE: This is a crucial change for the rollouts to work -- should it live here or elsewhere?
        if "composite_controller_specific_configs" in env_kwargs["controller_configs"]:
            env_kwargs["controller_configs"]["composite_controller_specific_configs"]["ik_input_ref_frame"] = "world"

        self.env = robosuite.make(**env_kwargs)


        logger.info(
            f"Successfully created {env_name} environment via robosuite.make() with cameras at {camera_size}x{camera_size}"
        )
        logger.info(f"Configured cameras: {camera_names}")

        # For now, we only support single environment (num_envs=1)
        # TODO: Could be extended to support multiple parallel environments
        if num_envs != 1:
            logger.warning(f"DexMimicGen wrapper currently only supports num_envs=1, got {num_envs}")

        # 环境创建后定义 action 和 observation space
        self._setup_spaces()

    def _setup_spaces(self):
        """Setup observation and action spaces for Gymnasium compatibility."""
        # 从 robosuite 环境获取 action space 维度
        action_dim = self.env.action_dim
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # 获取样例 observation 以确定实际 shape
        try:
            sample_obs_raw = self.env.reset()
            sample_obs = self._process_obs_for_space_inference(sample_obs_raw)

            # 基于实际 observation shape 设置 observation space
            obs_spaces = {}
            for key, value in sample_obs.items():
                obs_spaces[key] = gym.spaces.Box(
                    low=-np.inf if "state" in key else 0.0,
                    high=np.inf if "state" in key else 1.0,
                    shape=value.shape,
                    dtype=value.dtype,
                )

            self.observation_space = gym.spaces.Dict(obs_spaces)

        except Exception as e:
            logger.warning(f"Failed to infer observation space from sample: {e}")
            # 回退到估计的 space
            self._setup_fallback_spaces()

    def _process_obs_for_space_inference(self, obs):
        """Process observations for space inference without storing _last_obs."""
        processed_obs = {}

        # 提取机器人状态
        expected_low_dim_keys = self._get_expected_low_dim_keys(self.env_name)
        state_components = []

        for key in expected_low_dim_keys:
            if key in obs:
                obs_data = obs[key]
                if obs_data.ndim == 0:  # scalar
                    obs_data = np.array([obs_data])
                state_components.append(obs_data)

        if state_components:
            concatenated_state = np.concatenate(state_components)
            processed_obs["observation.state"] = concatenated_state.astype(np.float32)

        # 提取相机 observation
        for img_key in self.expected_image_keys:
            camera_name = img_key.replace("_image", "")
            robosuite_key = f"{camera_name}_image"

            if robosuite_key in obs:
                img = obs[robosuite_key]
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)

                clean_key = img_key.replace("_image", "")
                processed_obs[f"observation.images.{clean_key}"] = img

        return processed_obs

    def _setup_fallback_spaces(self):
        """Fallback observation space setup if inference fails."""
        obs_spaces = {}

        # 估计的 state space
        expected_low_dim_keys = self._get_expected_low_dim_keys(self.env_name)
        state_dim = 0
        for key in expected_low_dim_keys:
            if "pos" in key:
                state_dim += 3
            elif "quat" in key:
                state_dim += 4
            elif "qpos" in key:
                state_dim += 1
            else:
                state_dim += 1

        obs_spaces["observation.state"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        # 图像 observation space
        for img_key in self.expected_image_keys:
            clean_key = img_key.replace("_image", "")
            obs_spaces[f"observation.images.{clean_key}"] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(3, self.camera_size, self.camera_size), dtype=np.float32
            )

        self.observation_space = gym.spaces.Dict(obs_spaces)

    def seed(self, seed=None):
        """Seed the environment's random number generator."""
        # For robosuite environments, we don't have direct seeding control
        # This is a no-op for compatibility
        return [seed]

    def reset(self, *, seed=None, options=None):
        """Reset the environment and return initial observation."""
        # Gymnasium interface: reset can accept seed and options
        # For robosuite environments, we'll ignore these for now
        obs = self.env.reset()
        processed_obs = self._process_obs(obs)
        self._last_obs = processed_obs  # Store for video recording
        self.episode_steps = 0
        return processed_obs, {}

    def step(self, action):
        """Step the environment with the given action."""
        # 必要时将 action 从 torch tensor 转为 numpy
        if hasattr(action, "cpu"):
            action = action.cpu().numpy()
        if action.ndim > 1:
            action = action[0]  # Take first action if batched

        obs, reward, done, info = self.env.step(action)
        self.episode_steps += 1
        # 转换为预期格式
        processed_obs = self._process_obs(obs)

        # 返回 scalar values - Gymnasium will handle device placement and batching in vectorized env
        reward_scalar = float(reward)

        # 成功时终止 episode (reward == 1) to shortcircuit successful rollouts
        success = reward == 1.0
        terminated_scalar = bool(success)
        truncated_scalar = bool(done)  # Robosuite 返回 done when timeout

        if terminated_scalar or truncated_scalar:
            info = {
                **info,
                "success": success,
                "episode_steps": self.episode_steps,
            }
            self.episode_steps = 0

        return processed_obs, reward_scalar, terminated_scalar, truncated_scalar, info

    def _process_obs(self, obs):
        """Process robosuite observations to match expected format."""
        processed_obs = {}

        # 使用与数据集转换相同的 feature 提取机器人状态
        state_components = []

        # 获取该环境预期的 low_dim_keys (same as dataset conversion)
        expected_low_dim_keys = self._get_expected_low_dim_keys(self.env_name)

        for key in expected_low_dim_keys:
            if key in obs:
                obs_data = obs[key]
                if obs_data.ndim == 0:  # scalar
                    obs_data = np.array([obs_data])
                state_components.append(obs_data)
            else:
                logger.warning(f"Expected state key '{key}' not found in environment observations")

        if state_components:
            concatenated_state = np.concatenate(state_components)
            # 返回 numpy array - Gymnasium will handle device placement and batching
            processed_obs["observation.state"] = concatenated_state.astype(np.float32)

        # 提取相机 observation using the same logic as dataset conversion
        for img_key in self.expected_image_keys:
            # 转换为 robosuite 相机名 (remove '_image' suffix)
            camera_name = img_key.replace("_image", "")
            robosuite_key = f"{camera_name}_image"

            # Robosuite 图像为 (H, W, C) in uint8, need (C, H, W) in float32
            if robosuite_key in obs:
                img = obs[robosuite_key]
                img = img.astype(np.float32) / 255.0  # Convert to float32 and normalize
                img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)

                # 使用与数据集转换相同的命名约定: observation.images.{clean_key}
                clean_key = img_key.replace("_image", "")
                processed_obs[f"observation.images.{clean_key}"] = img

        # 保存最近一次 obs 用于录制视频 (convert back to torch tensors on device)
        self._last_obs = {}
        for key, value in processed_obs.items():
            self._last_obs[key] = value

        # 记录 observation key 以便调试 (only on first call)
        if not hasattr(self, "_logged_obs_keys"):
            logger.info(f"Created observation keys: {list(processed_obs.keys())}")
            self._logged_obs_keys = True

        return processed_obs

    def _get_expected_image_keys(self, env_name: str):
        """Return the expected image keys for a given environment.

        The logic mirrors what is used during dataset conversion so that the
        training / rollout code sees the exact same observation structure.
        """
        env_lower = env_name.lower()

        # ------------------------------------------------------------------
        # Match with ReinFlow / DPPO experiments - Hongusk. Oct 26, 2025.
        # ------------------------------------------------------------------
        if env_lower in {"lift", "can", "pickplacecan"}:
            return ["robot0_eye_in_hand_image"]
        elif env_lower in {"square", "nutassemblysquare", "transport", "twoarmtransport"}:
            return ["agentview_image"]

        # ---------------------------------------------
        # 规范相机 key 集合
        # ---------------------------------------------
        panda_image_keys_single = [
            "agentview_image",
            "robot0_eye_in_hand_image",
        ]

        panda_image_keys_multi = [
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot1_eye_in_hand_image",
        ]

        panda_transport_image_keys = [
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot1_eye_in_hand_image",
            "shouldercamera0_image",
            "shouldercamera1_image",
        ]

        humanoid_image_keys = [
            "agentview_image",
            "robot0_eye_in_left_hand_image",
            "robot0_eye_in_right_hand_image",
        ]

        humanoid_can_sort_image_keys = [
            "frontview_image",
            "robot0_eye_in_left_hand_image",
            "robot0_eye_in_right_hand_image",
        ]

        # Transport (two-arm Panda)
        if "transport" in env_lower:
            return panda_transport_image_keys

        # Humanoid 变体 -------------------------------------------------
        if "can_sort" in env_lower:
            return humanoid_can_sort_image_keys
        if any(task in env_lower for task in ["pouring", "coffee"]):
            return humanoid_image_keys

        # 单臂 Panda 任务 (Lift, Can, Square, etc.) ------------------
        if env_lower in {"lift", "can", "pickplacecan", "square", "nutassemblysquare"}:
            return panda_image_keys_single

        # 回退到双臂 Panda 相机 --------------------------------
        return panda_image_keys_multi

    def _get_expected_low_dim_keys(self, env_name: str):
        """返回 the expected low-dimensional state keys for a given environment."""

        panda_low_dim_keys_single = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]

        panda_low_dim_keys_multi = [
            *panda_low_dim_keys_single,
            "robot1_eef_pos",
            "robot1_eef_quat",
            "robot1_gripper_qpos",
        ]

        humanoid_low_dim_keys = [
            "robot0_right_eef_pos",
            "robot0_right_eef_quat",
            "robot0_right_gripper_qpos",
            "robot0_left_eef_pos",
            "robot0_left_eef_quat",
            "robot0_left_gripper_qpos",
        ]

        env_lower = env_name.lower()

        # Humanoid 变体
        if any(task in env_lower for task in ["pouring", "coffee", "can_sort"]):
            return humanoid_low_dim_keys

        # 单臂 Panda 任务
        if env_lower in {"lift", "can", "pickplacecan", "square", "nutassemblysquare"}:
            return panda_low_dim_keys_single

        # 默认：双臂 Panda
        return panda_low_dim_keys_multi

    def render(self):
        """返回 an RGB frame (H, W, 3, uint8) for video recording."""

        if self.video_key is None:
            frame = self.env.sim.render(camera_name="agentview", height=self.render_size[0], width=self.render_size[1])[
                ::-1
            ]
        else:
            frame = self.env.sim.render(camera_name=self.video_key, height=self.render_size[0], width=self.render_size[1])[
                ::-1
            ]

        return frame

    def set_video_key(self, video_key: str):
        """设置 which observation key to use for video recording."""
        self.video_key = video_key

    def close(self):
        """Close the environment."""
        self.env.close()

    @property
    def unwrapped(self):
        """Access to the underlying robosuite environment."""
        return self

    def get_wrapper_attr(self, name: str):
        """Utility for Gymnasium AsyncVectorEnv compatibility.

        Gymnasium's AsyncVectorEnv workers rely on every environment (or wrapper) exposing
        a `get_wrapper_attr` method that walks through potential wrapper chains to
        retrieve attributes. Since this class is not derived from `gym.Wrapper`, we
        provide a minimal implementation that simply 返回 the attribute from this
        instance (if it exists) or raises an `AttributeError` otherwise. This is
        sufficient because the environment is not wrapped multiple times on the
        worker side.
        """
        if hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    def set_wrapper_attr(self, name: str, value):
        """Counterpart to `get_wrapper_attr` expected by Gymnasium.

        Allows the vectorised worker to set attributes on the environment even when
        it is not a `gym.Wrapper`.
        """
        setattr(self, name, value)


def make_dexmimicgen_env(
    env_name: str,
    video_key: str,
    camera_size: int = 84,
    render_size: int | None = None,
    render_gpu_device_id: int = 0,
    env_id: int = 0,
    expected_image_keys: list[str] | None = None,
    seed: int | None = None,
):
    """Factory function to create a DexMimicGen environment for vectorization."""

    def _make():
        return RobosuiteGymWrapper(
            env_name=env_name,
            num_envs=1,
            video_key=video_key,
            render_gpu_device_id=render_gpu_device_id,
            camera_size=camera_size,
            render_size=render_size,
            env_id=env_id,
            expected_image_keys=expected_image_keys,
            seed=seed,
        )

    return _make


class VectorizedEnvWrapper:
    """Simple wrapper around gymnasium vectorized environments to add rendering capability."""

    def __init__(
        self, vec_env: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv, video_key: str, device: str = "cpu"
    ):
        self.vec_env = vec_env
        self.video_key = video_key
        self._last_obs = None
        self.device = device

    def reset(self, **kwargs):
        obs, info = self.vec_env.reset(**kwargs)
        self._last_obs = obs
        obs = self._convert_obs_to_torch(obs, self.device)
        return obs, info

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        obs, rewards, terminated, truncated, info = self.vec_env.step(actions)
        self._last_obs = obs

        # 转换为 torch tensor
        obs = self._convert_obs_to_torch(obs, self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool)
        truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool)

        return obs, rewards, terminated, truncated, info

    def render(self) -> np.ndarray:
        """返回 RGB frames from all environments (num_envs, H, W, 3, uint8) for video recording."""
        frames: np.ndarray | None = self.vec_env.render()
        if frames is None:
            raise RuntimeError("No frames returned from vectorized environment")
        return frames

    @property
    def fps(self):
        return self.vec_env.metadata["render_fps"]

    def close(self):
        return self.vec_env.close()

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying vectorized environment."""
        return getattr(self.vec_env, name)

    def _convert_obs_to_torch(self, obs, device):
        """Convert numpy observations from vectorized env to PyTorch tensors for policy."""
        if isinstance(obs, dict):
            torch_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    torch_obs[key] = torch.from_numpy(value).to(device)
                else:
                    torch_obs[key] = value
            return torch_obs
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs).to(device)
        return obs


def create_vectorized_env(
    env_name: str,
    num_envs: int,
    device: str = "cpu",
    camera_size: int = 84,
    render_size: int | None = None,
    debug: bool = False,
    vectorization: str = "async",
    async_context: str = "spawn",
    async_shared_memory: bool = True,
    video_key: str = "agentview",
    rank: int = 0,
    expected_image_keys: list[str] | None = None,
    seeds: list[int] | None = None,
):
    """Create vectorized environment using Gymnasium's vector environments.

    Args:
        expected_image_keys: List of image observation keys to use (e.g., ["agentview_image", "robot0_eye_in_hand_image"]).
                           If None, uses default keys based on environment name.
    """

    # 创建环境工厂函数列表

    env_fns = []

    # If CUDA_VISIBLE_DEVICES is set, all environments should render on device 0 (the only one visible).
    # Otherwise, distribute rendering across available GPUs.
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        # When this is set, the process only sees the specified GPUs, and they are indexed from 0.
        # For rendering in robosuite, we should always use device 0.
        # visible_device_ids = [0]
        visible_device_ids = [int(id) for id in cuda_visible_devices.split(",")]
    else:
        # If not set, assume all devices from 0 to torch.cuda.device_count()-1 are visible
        visible_device_ids = list(range(torch.cuda.device_count()))

    num_visible_gpus = len(visible_device_ids) if visible_device_ids else 1

    print(f"num_visible_gpus: {num_visible_gpus}", visible_device_ids)

    if seeds is None:
        seeds = [None] * num_envs
    for env_id in range(num_envs):
        if num_visible_gpus > 1:
            render_gpu_device_id = visible_device_ids[(rank * num_envs + env_id) % num_visible_gpus]
        else:
            render_gpu_device_id = visible_device_ids[0] if visible_device_ids else 0

        # if rank != -1:
        #     render_gpu_device_id = rank
        env_fns.append(
            make_dexmimicgen_env(
                env_name,
                video_key,
                camera_size,
                render_size,
                render_gpu_device_id,
                env_id,
                expected_image_keys,
                seeds[env_id],
            )
        )

    if debug or vectorization == "sync":
        # 使用同步向量化环境以便调试
        logger.info("Using gymnasium.vector.SyncVectorEnv")
        vec_env = gym.vector.SyncVectorEnv(
            env_fns,
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
    elif vectorization == "async":
        # 使用异步向量化环境以提升性能；context/shared_memory 可按机器稳定性调整。
        logger.info(
            "Using gymnasium.vector.AsyncVectorEnv "
            f"(context={async_context}, shared_memory={async_shared_memory})"
        )
        vec_env = gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=async_shared_memory,
            copy=True,
            context=async_context,
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
    else:
        raise ValueError(f"Unknown vectorization mode: {vectorization}")

    # 使用自定义 wrapper 以支持渲染
    # video key is actually set in the make_dexmimicgen_env function. I don't know why this is passed in here - Hongsuk.
    wrapped_env = VectorizedEnvWrapper(vec_env, video_key, device)

    # 添加环境元数据供后续读取
    wrapped_env.env_name = env_name
    wrapped_env.camera_size = camera_size
    wrapped_env.render_size = render_size


    logger.info(f"Created {num_envs} vectorized {env_name} environments")
    logger.info(f"Set video key to '{video_key}' for video recording")
    return wrapped_env
