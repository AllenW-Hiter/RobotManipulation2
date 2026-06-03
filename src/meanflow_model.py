#!/usr/bin/env python

"""Meanflow Policy 实现"""
from collections import deque
from typing import Union

import torch
import torch.nn.functional as F
import torchvision
from diffusers import EMAModel
from lerobot.common.constants import ACTION, OBS_IMAGES
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.types import FeatureType, PolicyFeature
from torch import Tensor, nn

from .meanflow_model_config import MeanFlowConfig
from .flow_net_mlp import MeanflowMLPModel
from .noise_injection_network import NoiseInjectionNetwork

# from .flow_net_unet import FlowMatchingUnetModel
# from .flow_net_residual_mlp import FlowMatchingResidualMLPModel


# 视觉编码器辅助函数
def get_resnet(name: str, weights=None, **kwargs) -> nn.Module:
    """获取移除最终 FC 层的 ResNet 模型。"""
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = nn.Identity()
    return resnet


def replace_bn_with_gn(module: nn.Module, features_per_group: int = 16) -> nn.Module:
    """将所有 BatchNorm 层替换为 GroupNorm。"""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_groups = child.num_features // features_per_group
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
        else:
            replace_bn_with_gn(child, features_per_group)
    return module

class MeanFlowPolicy(PreTrainedPolicy):
    """
    用于行为克隆的 Flow Matching Policy 实现。
    使用 continuous normalizing flow 和 flow matching loss。
    """

    config_class = MeanFlowConfig
    name = "meanflow"

    def __init__(
        self,
        config: MeanFlowConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # 创建用于归一化的 feature 字典
        input_feature_dict = {}
        for feat in config.input_features:
            # 根据名称判断 feature 类型
            if "image" in feat:
                feat_type = FeatureType.VISUAL
                shape = config.input_shapes.get(feat, (3, 96, 96))
                # 必要时将 HWC 转为 CHW 格式
                if len(shape) == 3 and shape[2] <= 4:  # Likely HWC format (e.g., 84, 84, 3)
                    shape = (shape[2], shape[0], shape[1])  # Convert to CHW
            elif "state" in feat or "pos" in feat:
                feat_type = FeatureType.STATE
                shape = config.input_shapes.get(feat, (2,))
            else:
                feat_type = FeatureType.ENV
                shape = config.input_shapes.get(feat, (1,))

            input_feature_dict[feat] = PolicyFeature(type=feat_type, shape=shape)

        output_feature_dict = {}
        for feat in config.output_features:
            if feat == "action":
                feat_type = FeatureType.ACTION
                shape = config.output_shapes.get(feat, (2,))
            else:
                feat_type = FeatureType.ACTION
                shape = config.output_shapes.get(feat, (1,))

            output_feature_dict[feat] = PolicyFeature(type=feat_type, shape=shape)

        # 归一化层
        self.normalize_inputs = Normalize(input_feature_dict, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(output_feature_dict, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(output_feature_dict, config.normalization_mapping, dataset_stats)

        # TODO：这里先实现 MLP
        # 根据网络结构初始化 meanflow model
        if config.network_architecture == "mlp":
            self.model = MeanflowMLPModel(config)
        else:
            raise ValueError(f"Meanflow Policy 目前只支持 MLP 版本。")

        

        # 当 ema_power > 0 时初始化 EMA model
        self.ema_model = None
        if config.ema_power > 0:
            self.ema_model = EMAModel(
                self.model.parameters(),
                power=config.ema_power,
                model_cls=type(self.model),
                model_config=None,
            )

        # 探索噪声值 (can be set dynamically)
        self.exploration_noise_std = config.exploration_noise_std  

        self.num_envs = None
        self.action_buffers = None

        self.reset()

    # 初始化空 action buffer 字典
    def init_action_buffers(self, num_envs: int): 
        self.num_envs = num_envs
        self.action_buffers = {
            env_id: deque([], maxlen=self.config.n_action_steps) for env_id in range(self.num_envs)
        }
        self.mdp_x_t_path_buffers = {
            env_id: deque([], maxlen=self.config.n_action_steps) for env_id in range(self.num_envs)
        }
    
    # 获取优化器参数，并为 backbone 与其他部分使用不同学习率(不学习视觉部分)
    def get_optim_params(self) -> dict:
        """获取优化器参数，并为 backbone 与其他部分使用不同学习率。"""
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.vision_encoder") and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if n.startswith("model.vision_encoder") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]
    
    # 重置 重置指定环境的 action buffer；如果 env_ids 为 None，则重置全部环境。
    def reset(self, env_ids: Tensor | None = None):
        if env_ids is None:
            if self.num_envs is None:
                self.action_buffers = {}
                self.mdp_x_t_path_buffers = {}

            else:
                # 重置所有 buffer
                for env_id in range(self.num_envs):
                    self.action_buffers[env_id] = deque([], maxlen=self.config.n_action_steps)
                    self.mdp_x_t_path_buffers[env_id] = deque([], maxlen=self.config.n_action_steps)
        else:
            # 只重置指定环境的 buffer
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids)

            for env_id in env_ids.tolist():
                if env_id in self.action_buffers:
                    self.action_buffers[env_id] = deque([], maxlen=self.config.n_action_steps)
                    self.mdp_x_t_path_buffers[env_id] = deque([], maxlen=self.config.n_action_steps)
    
    # 使用当前模型参数更新 EMA model。
    def step_ema(self):
        """使用当前模型参数更新 EMA model。"""
        if self.ema_model is not None:
            self.ema_model.step(self.model.parameters())

    # 切换为使用 EMA 权重进行推理。
    def enable_ema(self):
        if self.ema_model is not None:
            # 复制 EMA 权重前保存当前参数
            self.ema_model.store(self.model.parameters())
            # 将 EMA 权重复制到模型
            self.ema_model.copy_to(self.model.parameters())

    # 用 EMA 后恢复原始权重。
    def disable_ema(self):
        if self.ema_model is not None:
            # 恢复参数 that were stored in enable_ema()
            self.ema_model.restore(self.model.parameters())

    #TODO：需要修改
    # 获取 flow matching schedule。
    def get_schedule(self, device: torch.device) -> Tensor:
        return torch.linspace(1.0, 0.0, self.config.sampling_steps + 1, device=device)

    # MeanFlow 区间采样推理
    @torch.no_grad
    def predict_action_chunk(self, batch: dict[str, Tensor], zero_sampling: bool = False, sde_sampling: bool = False) -> Tensor:
        """MeanFlow 区间采样推理。

        使用区间 [t_i, t_{i+1}] 调用网络预测平均速度，
        沿时间方向逐步将噪声转化为动作。

        参数:
            batch: 观测批次
            zero_sampling: 若 True，从零初始化；否则从高斯噪声初始化
            sde_sampling: MeanFlow BC 阶段不支持，设为 True 会报错

        返回:
            (actions, mdp_x_t_path)
        """
        # 传入参数校验
        if sde_sampling:
            raise NotImplementedError(
                "MeanFlow BC 阶段不支持 sde_sampling。SDE 采样需要后续设计。"
            )
        
        # 切换回验证模式
        self.eval()

        # 归一化输入
        batch = self.normalize_inputs(batch)

        # 获取 observation conditioning (B, T * total_dim)
        obs_cond = self.model.encode_observations(batch)

        # 从高斯噪声初始化
        B = obs_cond.shape[0]
        if zero_sampling:
            x_t = torch.zeros((B, self.config.horizon, self.model.action_dim), device=obs_cond.device)
        else:
            x_t = torch.randn((B, self.config.horizon, self.model.action_dim), device=obs_cond.device)

        # 获取时间调度 t_path = torch.linspace(1.0, 0.0, flow_steps + 1, device=x_t.device)
        flow_steps = self.config.sampling_steps
        t_path = self.get_schedule(x_t.device)
        mdp_x_t_path = torch.zeros((B, flow_steps, self.config.horizon, self.model.action_dim), device=x_t.device)

        # 区间采样: 从 t=1 到 t=0
        for i in range(flow_steps):
            t_current = t_path[i]
            t_next = t_path[i + 1]
            
            # 构造批次级别的 t 和 r Tensor
            t_batch = torch.full((B,), t_current, device=x_t.device, dtype=x_t.dtype)
            r_batch = torch.full((B,), t_next, device=x_t.device, dtype=x_t.dtype)
            time_emb = self.model.encode_time_pair(t_batch, r_batch)
            
            # 网络预测区间平均速度
            u = self.model.forward(x_t=x_t,t_emb=time_emb,obs_cond=obs_cond)
            u = self.config.mlp_output_scale * u
            assert u.shape == (B, self.config.horizon, self.model.action_dim), "network_output shape should be (B, horizon, action_dim)"

            # 如果已配置则应用 clipping
            if self.config.transported_clip_value is not None:
                u = u.clamp(
                    -self.config.transported_clip_value, 
                    self.config.transported_clip_value
                    )

            # x_r = x_t - (t - r) * u_theta
            dt = t_current - t_next
            x_t = x_t - dt * u
            mdp_x_t_path[:, i] = x_t

        # 缩放 action
        actions = self.config.actor_scale * x_t
        
        # 添加探索噪声（仅在训练模式且 exploration_noise_std > 0 时）
        if self.training and self.exploration_noise_std > 0:
            noise = self.exploration_noise_std * torch.randn_like(actions)
            actions = actions + noise

        # 反归一化 action
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

        return actions, mdp_x_t_path

    # 使用独立 buffer 为多个环境选择 action
    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], zero_sampling: bool = False, sde_sampling: bool = False) -> Tensor:
        """使用独立 buffer 为多个环境选择 action。
        
        Args:
            batch: Dictionary of observations with tensors of shape (num_envs, ...)
            
        Returns:
            Tensor of actions with shape (num_envs, action_dim)
        """
        # 切换回验证模式
        self.eval()
        
        # 确保 action_buffers 在之前已经初始化了
        if self.num_envs is None and self.action_buffers is None:
            raise ValueError("Action buffers not initialized. Call init_action_buffers first.")
        
        # 检查哪些环境需要新的 action chunk
        envs_needing_actions = []
        for env_id in range(self.num_envs):
            if len(self.action_buffers[env_id]) == 0:
                envs_needing_actions.append(env_id)
        
        # 为需要的环境生成新的 action chunk
        if envs_needing_actions:
            # 为需要 action 的环境创建 batch
            sub_batch = {}
            for key, value in batch.items():
                sub_batch[key] = value[envs_needing_actions]

            # 预测这些环境的 action chunk
            action_chunks, mdp_x_t_path = self.predict_action_chunk(sub_batch, zero_sampling=zero_sampling, sde_sampling=sde_sampling)
            action_chunks = action_chunks[:, :self.config.n_action_steps, :] # TODO n_action_steps 目前需要等于 horizon
            mdp_x_t_path = mdp_x_t_path[:, :, :self.config.n_action_steps, :] # TODO n_action_steps 目前需要等于 horizon
            assert mdp_x_t_path.shape[1] == self.config.sampling_steps, "mdp_x_t_path second axis should be the flow sampling steps"
            assert action_chunks.shape[1] == self.config.n_action_steps, "action_chunks second axis should be the number of action steps"

            # 填充这些环境的 buffer
            for i, env_id in enumerate(envs_needing_actions):
                # Transpose to get actions for this environment across timesteps
                self.action_buffers[env_id].extend(action_chunks[i])
                # (sampling_steps, n_action_steps, action_dim) - > (n_action_steps, sampling_steps, action_dim)
                self.mdp_x_t_path_buffers[env_id].extend(mdp_x_t_path[i].transpose(0,1)) 

        # 收集所有环境的 action
        actions = []
        mdp_x_t_paths = []
        for env_id in range(self.num_envs):
            actions.append(self.action_buffers[env_id].popleft())
            mdp_x_t_paths.append(self.mdp_x_t_path_buffers[env_id].popleft())
        assert len(actions) == self.num_envs, "actions length should be the number of environments"
        assert len(mdp_x_t_paths) == self.num_envs, "mdp_x_t_paths length should be the number of environments"
        return torch.stack(actions), torch.stack(mdp_x_t_paths)

    
    # 正向运行
    def forward(
        self, 
        batch: dict[str, Tensor], 
        n_action_samples: int = 1, 
        cfm_loss_t: Tensor = None, 
        cfm_loss_r: Tensor = None,
        cfm_loss_eps: Tensor = None, 
        debug=False, 
        is_dppo: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        if is_dppo:
            return self.forward_dppo(batch, debug)
        else:
            return self.forward_fpo(
                batch,
                n_action_samples=n_action_samples,
                cfm_loss_t=cfm_loss_t,
                cfm_loss_eps=cfm_loss_eps,
                cfm_loss_r=cfm_loss_r,
                debug=debug,
            )
    
    # 采样 (t, r) 时间对，满足 0 <= r <= t <= 1
    def _sample_meanflow_time_pair(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if cfg.meanflow_time_sampling == "logit_normal_pair":
            # DM1 实际使用的 logit-normal pair 采样
            pair = torch.randn(batch_size, 2, device=device, dtype=dtype)
            pair = torch.sigmoid(pair * cfg.meanflow_logit_sigma + cfg.meanflow_logit_mu)
        elif cfg.meanflow_time_sampling == "uniform_pair":
            pair = torch.rand(batch_size, 2, device=device, dtype=dtype)
        else:
            raise ValueError(f"未知的时间采样方式: {cfg.meanflow_time_sampling}")
        
        # 取最大值作为 t，最小值作为 r，保证 r <= t
        t = pair.max(dim=1).values
        r = pair.min(dim=1).values
        
        # 按 flow_ratio 比例让部分样本的 r = t（退化为瞬时速度场景）
        if cfg.meanflow_flow_ratio > 0:
            n_flow = int(batch_size * cfg.meanflow_flow_ratio)
            if n_flow > 0:
                indices = torch.randperm(batch_size, device=device)[:n_flow]
                r[indices] = t[indices]
        
        return t, r
        
    
    # MeanFlow 训练损失
    def get_meanflow_loss(
        self,
        batch: dict[str, Tensor],
        meanflow_loss_eps: Tensor | None = None,
        meanflow_loss_t: Tensor | None = None,
        meanflow_loss_r: Tensor | None = None,
        non_reduction: bool = False,
    ) -> tuple[Tensor, dict]:
        # 归一化输入和目标
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        
        # 获取 observation conditioning
        obs_cond = self.model.encode_observations(batch)
        
        # 获取 clean actions (x0)
        x_data = batch[ACTION]  # (B, T, D)
        B, T_horizon, D = x_data.shape
        
        # 传入检验
        supplied = (meanflow_loss_eps is not None,meanflow_loss_t is not None,meanflow_loss_r is not None,)
        if any(supplied) and not all(supplied):
            raise ValueError("MeanFlow loss replay requires eps, t, and r together.")
        
        if all(supplied):
            x_noise = meanflow_loss_eps
            t = meanflow_loss_t.reshape(B)
            r = meanflow_loss_r.reshape(B)
        else:
            x_noise = torch.randn_like(x_data)
            t, r = self._sample_meanflow_time_pair(B, x_data.device, x_data.dtype)
        t_view = t.view(B, 1, 1)
        r_view = r.view(B, 1, 1)
        
        # 构造带噪动作: x_t = (1 - t) * x_data + t * x_noise
        x_t = (1 - t_view) * x_data + t_view * x_noise
        v = x_noise - x_data  # 从 data 到 noise 的速度方向
        
        # 定义网络函数用于 JVP 计算
        def network_fn(z: Tensor, t_in: Tensor, r_in: Tensor) -> Tensor:
            time_emb = self.model.encode_time_pair(t_in, r_in)
            output = self.model.forward(x_t=z, t_emb=time_emb, obs_cond=obs_cond)
            output = self.config.mlp_output_scale * output
            if self.config.transported_clip_value is not None:
                output = output.clamp(-self.config.transported_clip_value,self.config.transported_clip_value,)
            return output
        
        # 使用 JVP 计算网络输出沿轨迹的导数 du_dt
        # tangent 对 (x_t, t, r) 分别为 (v, 1, 0)
        u, du_dt = torch.autograd.functional.jvp(
            network_fn,
            (x_t, t, r),
            (v, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=True,
        )
        
        # MeanFlow 目标: v - (t - r) * du_dt
        target = v - (t_view - r_view) * du_dt
        
        # 损失目标不反传；adaptive L2 按 DM1 的 stopped weighting 实现。
        diff = u - target.detach()
        if self.config.meanflow_use_adaptive_loss:
            squared_error = diff.square()
            if "action_is_pad" in batch:
                mask = (~batch["action_is_pad"]).unsqueeze(-1).expand_as(squared_error)
                valid_count = mask.sum(dim=(1, 2)).clamp_min(1)
                delta_sq = (squared_error * mask).sum(dim=(1, 2)) / valid_count
            else:
                delta_sq = squared_error.mean(dim=(1, 2))
            power = 1.0 - self.config.meanflow_adaptive_gamma
            weight = (delta_sq + self.config.meanflow_adaptive_c).pow(-power).detach()
            if non_reduction:
                return squared_error * weight.view(B, 1, 1)
            loss = (weight * delta_sq).mean()
            return loss, {"meanflow_loss": loss.item()}

        if self.config.cfm_loss_use_huber:
            abs_diff = torch.abs(diff)
            delta = self.config.cfm_loss_huber_delta
            loss = torch.where(
                abs_diff <= delta,
                diff ** 2,
                2 * delta * abs_diff - delta ** 2,
            )
        else:
            loss = diff.square()
            
        if non_reduction:
            return loss
        
        # 如果存在 padding mask 则处理它
        if "action_is_pad" in batch:
            mask = ~batch["action_is_pad"].unsqueeze(-1)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        loss_dict = {"flow_loss": loss.item()}
        return loss, loss_dict
        
        
    
    # 正向运行 fpo 模式
    def forward_fpo(
        self,
        batch: dict[str, Tensor],
        n_action_samples: int = 1,
        cfm_loss_t: Tensor | None = None,
        cfm_loss_eps: Tensor | None = None,
        cfm_loss_r: Tensor | None = None,
        debug: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # 读取动作
        actions = batch[ACTION] # (num_envs, horizon, action_dim)
        B, T, D = actions.shape
        
        # 检验传入参数
        supplied = (cfm_loss_t is not None, cfm_loss_eps is not None, cfm_loss_r is not None)
        if any(supplied) and not all(supplied):
            raise ValueError("MeanFlow FPO replay 需要 eps, t,  r 都有，若是强化学习，需要将所有都传入")
        # 此时为模仿学习
        if not all(supplied):
            flat_t, flat_r = self._sample_meanflow_time_pair(
                B * n_action_samples, actions.device, actions.dtype
            )
            cfm_loss_t = flat_t.view(B * n_action_samples, 1, 1)
            cfm_loss_r = flat_r.view(B * n_action_samples, 1, 1)
            cfm_loss_eps = torch.randn((B * n_action_samples, T, D), device=actions.device, dtype=actions.dtype)

        # 如果采样大于1，复制增加采样维度
        if n_action_samples > 1:
            # 复制 observation n_action_samples times
            for img_key in self.config.image_features:
                if img_key in batch:
                    value = batch[img_key]
                    batch[img_key] = value.unsqueeze(1).expand(B, n_action_samples, *value.shape[1:]).reshape(-1, *value.shape[1:])
            for state_key in self.config.state_features:
                if state_key in batch:
                    value = batch[state_key]
                    batch[state_key] = value.unsqueeze(1).expand(B, n_action_samples, *value.shape[1:]).reshape(-1, *value.shape[1:])
            # 重塑 action in to (B * n_action_samples, T, D)
            actions = actions.unsqueeze(1).expand(B, n_action_samples, T, D).reshape(-1, T, D)
            batch[ACTION] = actions
        
        # 计算 Meanflow 损失
        loss = self.get_meanflow_loss(
            batch,
            meanflow_loss_eps=cfm_loss_eps,
            meanflow_loss_t=cfm_loss_t,
            meanflow_loss_r=cfm_loss_r,
            non_reduction=True,
        )
        
        # (B * n_action_samples, T) -> (T,B*n_action_samples) -> (T,B,n_action_samples)
        loss = loss.mean(-1).transpose(0, 1).reshape(T, B, n_action_samples)
        
        # (B*n_action_samples, 1, 1) -> (T, B, n_action_samples)
        loss_t = cfm_loss_t.squeeze(-1).transpose(0, 1).expand(T, -1).reshape(T, B, n_action_samples)
        loss_r = cfm_loss_r.squeeze(-1).transpose(0, 1).expand(T, -1).reshape(T, B, n_action_samples)
        
        # (B*n_action_samples, T, D) -> (T, B, n_action_samples, D)
        loss_eps = cfm_loss_eps.permute(1, 0, 2).reshape(T, B, n_action_samples, D)
        
        return loss, loss_t,loss_r,loss_eps 

    # ------------------------------------------------------------------
    # 确保 DPPO/RF 路径被阻止
    # ------------------------------------------------------------------
    def forward_dppo(self, batch: dict[str, Tensor], debug=False) -> tuple[Tensor, Tensor, Tensor]:
        """MeanFlow BC 不支持 DPPO 轨迹概率计算。"""
        raise NotImplementedError(
            "MeanFlow BC 阶段不支持 DPPO 去噪似然计算。"
            "后续 RL 适配版本将提供。"
        )
    def get_sde_sigma(self, obs_cond: Tensor, device: torch.device) -> Tensor:
        """MeanFlow BC 不支持 SDE sigma。"""
        raise NotImplementedError(
            "MeanFlow BC 阶段不支持 SDE sigma 查询。"
            "后续 RL 适配版本将提供。"
        )

    # ------------------------------------------------------------------
    # 确保 DPPO/RF 路径被阻止
    # ------------------------------------------------------------------
    
    
    def initialize_noise_injection_network(self):
        """Initialize the noise injection network.

        Creates a NoiseInjectionNetwork that takes obs_cond as input and outputs
        a tensor with shape (B, T, 1, D) where:
            - B: batch size
            - T: action horizon (chunk size)
            - 1: singleton dimension for broadcasting across flow sampling steps
            - D: action dimension
        """
        # 获取 observation conditioning dimension from the model
        obs_cond_dim = self.model.global_cond_dim
        action_dim = self.model.action_dim
        horizon = self.config.horizon

        min_noise = getattr(self.config, 'noise_injection_min', 0.2)
        max_noise = getattr(self.config, 'noise_injection_max', 0.5)

        self.noise_injection_network = NoiseInjectionNetwork(
            obs_cond_dim=obs_cond_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dims=[256, 256],
            min_noise=min_noise,
            max_noise=max_noise,
        )

    def get_cfm_loss(
        self,
        batch: dict[str, Tensor],
        cfm_loss_eps: Tensor | None = None,
        cfm_loss_t: Tensor | None = None,
        non_reduction: bool = False,
    ) -> tuple[Tensor, dict] | Tensor:
        """兼容 BC 训练循环的 MeanFlow loss 入口。

        `pretrain_flow_bc.py` 统一调用 `get_cfm_loss()`；MeanFlow 实际使用
        `get_meanflow_loss()`，并且 replay loss 需要同时提供 eps/t/r。BC
        训练不传 replay 张量，因此这里直接转发。
        """
        if cfm_loss_eps is not None or cfm_loss_t is not None:
            raise ValueError("MeanFlow replay loss 需要 eps、t、r 三者；请直接调用 get_meanflow_loss()。")
        return self.get_meanflow_loss(batch, non_reduction=non_reduction)

    def _compute_cfm_loss_weight(self, t: Tensor) -> Tensor:
        """计算 timestep 相关权重 for CFM loss."""
        # t has shape (B, 1, 1)
        t_scalar = t.squeeze(-1).squeeze(-1)  # Shape: (B,)
        
        if self.config.cfm_loss_weight_from_t == "constant":
            weight = torch.ones_like(t)
        elif self.config.cfm_loss_weight_from_t == "linear_1_to_0.1":
            # Weight goes from 1.0 at t=0 to 0.1 at t=1
            weight = 1.0 - 0.9 * t
        elif self.config.cfm_loss_weight_from_t == "linear_1_to_0.01":
            # Weight goes from 1.0 at t=0 to 0.01 at t=1
            weight = 1.0 - 0.99 * t
        
        return weight

    def _compute_squared_error(
        self, predictions: Tensor, targets: Tensor
    ) -> torch.Tensor:
        """计算平方误差 with optional Huber loss."""
        if self.config.cfm_loss_use_huber:
            # 修改版 Huber loss to match MSE when |error| <= delta
            # L2 for |error| <= delta, linear for |error| > delta
            diff = predictions - targets
            abs_diff = torch.abs(diff)
            huber_loss = torch.where(
                abs_diff <= self.config.cfm_loss_huber_delta,
                diff**2,  # No 0.5 factor, matches MSE
                2 * self.config.cfm_loss_huber_delta * abs_diff - self.config.cfm_loss_huber_delta**2,
            )
            return huber_loss
        else:
            return F.mse_loss(predictions, targets, reduction="none")
