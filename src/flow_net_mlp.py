#!/usr/bin/env python


"""使用 MLP 结构的 Flow Matching Policy"""

import math
from typing import Union

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

from .flow_model_config import FlowMatchingConfig
from .meanflow_model_config import MeanFlowConfig
from .vit import VitEncoder, VitEncoderConfig


class SinusoidalPosEmb(nn.Module):
    """用于 timestep 的正弦位置 embedding。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


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


class FlowMatchingMLPModel(nn.Module):
    """用于行为克隆的 MLP 版 Flow Matching model。"""

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config

        # 从输出 shape 获取 action 维度
        self.action_dim = 2  # Default, will be overridden from config if available
        if hasattr(config, "output_shapes") and "action" in config.output_shapes:
            shape = config.output_shapes["action"]
            if isinstance(shape, (list, tuple)):
                self.action_dim = shape[-1] if len(shape) > 0 else 2
            else:
                self.action_dim = shape

        # 视觉编码器
        self.vision_encoder = self._init_vision_encoder(config)
        self.vision_feature_dim = self._get_vision_feature_dim(config.vision_backbone)

        # 计算 observation 维度
        obs_dim = 0
        if config.image_features:
            obs_dim += self.vision_feature_dim * len(config.image_features)
        if config.state_features:
            for f in config.state_features:
                if f in config.input_shapes:
                    shape = config.input_shapes[f]
                    if isinstance(shape, (list, tuple)):
                        obs_dim += shape[-1] if len(shape) > 0 else 1
                    else:
                        obs_dim += shape

        # 全局 conditioning 维度
        self.global_cond_dim = obs_dim

        # 时间 embedding 维度
        time_embed_dim = config.timestep_embed_dim

        # MLP 结构
        # Input: [noisy_action (action_dim * horizon), time_embedding, observation_conditioning]
        input_dim = self.action_dim * config.horizon + time_embed_dim + self.global_cond_dim

        # 从配置获取 MLP 维度, default to [512, 512, 512]
        mlp_dims = getattr(config, 'mlp_dims', [512, 512, 512])

        # 构建 MLP 层
        layers = []
        prev_dim = input_dim

        for hidden_dim in mlp_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Mish())
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, self.action_dim * config.horizon))

        self.mlp = nn.Sequential(*layers)

        # 时间步编码器 (sinusoidal positional encoding)
        self.diffusion_step_encoder = SinusoidalPosEmb(time_embed_dim)

    def _init_vision_encoder(self, config):
        """初始化 视觉编码器."""
        if config.vision_backbone.startswith("resnet"):
            encoder = get_resnet(config.vision_backbone, weights=config.pretrained_backbone_weights)
            if config.obs_encoder_group_norm:
                encoder = replace_bn_with_gn(encoder)
            return encoder
        elif config.vision_backbone == "vit":
            # 从数据集实际输入 shape 推断图像尺寸
            img_h, img_w = None, None
            if config.image_features and len(config.image_features) > 0:
                first_img_key = config.image_features[0]
                if first_img_key in config.input_shapes:
                    img_shape = config.input_shapes[first_img_key]
                    # Shape 可能是 (H, W, C) or (C, H, W)
                    if isinstance(img_shape, (list, tuple)) and len(img_shape) == 3:
                        if img_shape[0] == 3 or img_shape[0] == 1:
                            # (C, H, W) format
                            img_h, img_w = img_shape[1], img_shape[2]
                        else:
                            # (H, W, C) format
                            img_h, img_w = img_shape[0], img_shape[1]

            # 回退到配置值或默认值
            if img_h is None or img_w is None:
                img_size = getattr(config, "image_size", 84)
                img_h, img_w = img_size, img_size
                print(f"[ViT Init] Using default/config image size: {img_h}x{img_w}")
            else:
                print(f"[ViT Init] Inferred image size from dataset: {img_h}x{img_w}")

            vit_config = VitEncoderConfig(
                patch_size=getattr(config, "vit_patch_size", 8),
                depth=getattr(config, "vit_depth", 1),
                embed_dim=getattr(config, "vit_embed_dim", 128),
                num_heads=getattr(config, "vit_num_heads", 4),
            )
            encoder = VitEncoder(
                obs_shape=[3, img_h, img_w],
                cfg=vit_config,
                num_channel=3,
                img_h=img_h,
                img_w=img_w,
            )
            return encoder
        raise ValueError(f"Unsupported vision backbone: {config.vision_backbone}")

    def _get_vision_feature_dim(self, backbone_name):
        """获取 视觉编码器输出维度."""
        if backbone_name == "resnet18" or backbone_name == "resnet34":
            return 512
        if backbone_name == "resnet50":
            return 2048
        if backbone_name == "vit":
            # 对于 ViT，展平 patch: repr_dim = embed_dim * num_patches
            return self.vision_encoder.repr_dim
        return 512  # Default

    def encode_observations(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode observations into a conditioning vector."""
        B = next(iter(batch.values())).shape[0]
        features = []

        # 编码图像 observation
        if self.config.image_features:
            for img_key in self.config.image_features:
                if img_key in batch:
                    img_tensor = batch[img_key]
                    # img_tensor shape: (B, T, C, H, W) or (B, C, H, W)
                    if len(img_tensor.shape) == 5:
                        B, T = img_tensor.shape[:2]
                        img_flat = img_tensor.flatten(end_dim=1)  # (B*T, C, H, W)
                    else:
                        B = img_tensor.shape[0]
                        T = 1
                        img_flat = img_tensor  # Already (B, C, H, W)

                    # 通过视觉编码器
                    if self.config.vision_backbone == "vit":
                        # ViT 期望 0-255 输入 and 返回 (B, num_patches, embed_dim)
                        # 需要展平 to (B, repr_dim)
                        img_features = self.vision_encoder(img_flat * 255.0, flatten=True)  # (B*T, repr_dim) or (B, repr_dim)
                    else:
                        # ResNet 已经输出 (B, D)
                        img_features = self.vision_encoder(img_flat)  # (B*T, D) or (B, D)

                    # 重塑回原形状
                    if len(img_tensor.shape) == 5:
                        img_features = img_features.reshape(B, T, -1)  # (B, T, D)
                    else:
                        img_features = img_features.reshape(B, 1, -1)  # (B, 1, D)
                    features.append(img_features)

        # 编码状态 observation
        if self.config.state_features:
            for state_key in self.config.state_features:
                if state_key in batch:
                    state_tensor = batch[state_key]  # (B, T, D) or (B, D)
                    # 确保为 3D tensor (B, T, D)
                    if len(state_tensor.shape) == 2:
                        state_tensor = state_tensor.unsqueeze(1)  # (B, 1, D)
                    features.append(state_tensor)

        # 拼接所有 feature
        if features:
            # 确保所有 feature 具有相同时间维度
            max_T = max(f.shape[1] for f in features)
            padded_features = []
            for f in features:
                if f.shape[1] < max_T:
                    # 重复最后一帧以匹配时间维度
                    padding = f[:, -1:, :].repeat(1, max_T - f.shape[1], 1)
                    f = torch.cat([f, padding], dim=1)
                padded_features.append(f)

            obs_features = torch.cat(padded_features, dim=-1)  # (B, T, total_dim)
            # 展平时间维度用于全局 conditioning
            obs_cond = obs_features.flatten(start_dim=1)  # (B, T * total_dim)
        else:
            obs_cond = torch.zeros((B, 1), device=next(iter(batch.values())).device)

        return obs_cond

    def forward(self, x_t: Tensor, t_emb: Tensor, obs_cond: Tensor) -> Tensor:
        """
        前向传播 through the MLP.

        Args:
            x_t: (B, T, D) noisy actions where T is horizon and D is action_dim
            t_emb: (B, time_embed_dim) time embedding (already encoded)
            obs_cond: (B, obs_dim) observation conditioning

        Returns:
            (B, T, D) predicted velocity or x0
        """
        B, T, D = x_t.shape
        original_T = T
        if T < self.config.horizon:
            padding = x_t[:, -1:, :].repeat(1, self.config.horizon - T, 1)
            x_t = torch.cat([x_t, padding], dim=1)
            T = self.config.horizon
        elif T > self.config.horizon:
            raise ValueError(f"Action horizon {T} exceeds configured horizon {self.config.horizon}")

        # 展平 action to (B, T*D)
        x_t_flat = x_t.reshape(B, -1)

        # 拼接所有输入 (t_emb is already encoded)
        mlp_input = torch.cat([x_t_flat, t_emb, obs_cond], dim=-1)  # (B, input_dim)

        # 通过 MLP
        output = self.mlp(mlp_input)  # (B, T*D)

        # 重塑回原形状 to (B, T, D)
        output = output.reshape(B, T, D)

        return output[:, :original_T, :]

    def initialize_layers(self, init_fn=None):
        """
        初始化 all layers in the MLP with a given initialization function.
        If no function is provided, uses Kaiming normal initialization for Linear layers.
        """
        def default_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        init = init_fn if init_fn is not None else default_init

        # 初始化 MLP 层
        self.mlp.apply(init)
        # 初始化时间编码器
        self.diffusion_step_encoder.apply(init)


class MeanflowMLPModel(nn.Module):
    """用于行为克隆的 MLP 版 Meanflow model。"""

    def __init__(self, config: MeanFlowConfig):
        super().__init__()
        self.config = config

        # 从输出 shape 获取 action 维度
        self.action_dim = 2  # Default, will be overridden from config if available
        if hasattr(config, "output_shapes") and "action" in config.output_shapes:
            shape = config.output_shapes["action"]
            if isinstance(shape, (list, tuple)):
                self.action_dim = shape[-1] if len(shape) > 0 else 2
            else:
                self.action_dim = shape

        # 视觉编码器
        self.vision_encoder = self._init_vision_encoder(config)
        self.vision_feature_dim = self._get_vision_feature_dim(config.vision_backbone)

        # 计算 observation 维度
        obs_dim = 0
        if config.image_features:
            obs_dim += self.vision_feature_dim * len(config.image_features)
        if config.state_features:
            for f in config.state_features:
                if f in config.input_shapes:
                    shape = config.input_shapes[f]
                    if isinstance(shape, (list, tuple)):
                        obs_dim += shape[-1] if len(shape) > 0 else 1
                    else:
                        obs_dim += shape

        # 全局 conditioning 维度(注意，这里不含时间和动作)
        self.global_cond_dim = obs_dim

        # 时间 embedding 维度
        time_embed_dim = config.timestep_embed_dim

        # MLP 结构
        # 输入: [noisy_action, time_embedding(s), observation_conditioning]。
        # MeanFlow 默认编码 [t, r]；开启 meanflow_encode_t_minus_r 时额外编码区间 [t, r, t-r]。
        if getattr(config, "meanflow_encode_t_minus_r", False):
            self.meanflow_encode_t_minus_r = True
            num_time_embed = 3
        else:
            self.meanflow_encode_t_minus_r = False
            num_time_embed = 2
        input_dim = self.action_dim * config.horizon + time_embed_dim * num_time_embed + self.global_cond_dim

        # 从配置获取 MLP 维度, default to [512, 512, 512]
        mlp_dims = getattr(config, 'mlp_dims', [512, 512, 512])

        # 构建 MLP 层
        layers = []
        prev_dim = input_dim

        for hidden_dim in mlp_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Mish())
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, self.action_dim * config.horizon))

        self.mlp = nn.Sequential(*layers)

        # 时间步编码器（正弦位置编码）。默认 t、r、t-r 共享编码器；
        # separate=True 时保留独立模块接口，便于后续替换为可学习编码器。
        if getattr(config, "meanflow_separate_time_encoders", False):
            self.meanflow_separate_time_encoders = True
            self.time_t_encoder = SinusoidalPosEmb(time_embed_dim)
            self.time_r_encoder = SinusoidalPosEmb(time_embed_dim)
            self.time_t_minus_r_encoder = SinusoidalPosEmb(time_embed_dim)
        else:
            self.meanflow_separate_time_encoders = False
            self.time_t_encoder = SinusoidalPosEmb(time_embed_dim)
            self.time_r_encoder = self.time_t_encoder
            self.time_t_minus_r_encoder = self.time_t_encoder

    def _init_vision_encoder(self, config):
        """初始化 视觉编码器."""
        if config.vision_backbone.startswith("resnet"):
            encoder = get_resnet(config.vision_backbone, weights=config.pretrained_backbone_weights)
            if config.obs_encoder_group_norm:
                encoder = replace_bn_with_gn(encoder)
            return encoder
        elif config.vision_backbone == "vit":
            # 从数据集实际输入 shape 推断图像尺寸
            img_h, img_w = None, None
            if config.image_features and len(config.image_features) > 0:
                first_img_key = config.image_features[0]
                if first_img_key in config.input_shapes:
                    img_shape = config.input_shapes[first_img_key]
                    # Shape 可能是 (H, W, C) or (C, H, W)
                    if isinstance(img_shape, (list, tuple)) and len(img_shape) == 3:
                        if img_shape[0] == 3 or img_shape[0] == 1:
                            # (C, H, W) format
                            img_h, img_w = img_shape[1], img_shape[2]
                        else:
                            # (H, W, C) format
                            img_h, img_w = img_shape[0], img_shape[1]

            # 回退到配置值或默认值
            if img_h is None or img_w is None:
                img_size = getattr(config, "image_size", 84)
                img_h, img_w = img_size, img_size
                print(f"[ViT Init] Using default/config image size: {img_h}x{img_w}")
            else:
                print(f"[ViT Init] Inferred image size from dataset: {img_h}x{img_w}")

            vit_config = VitEncoderConfig(
                patch_size=getattr(config, "vit_patch_size", 8),
                depth=getattr(config, "vit_depth", 1),
                embed_dim=getattr(config, "vit_embed_dim", 128),
                num_heads=getattr(config, "vit_num_heads", 4),
            )
            encoder = VitEncoder(
                obs_shape=[3, img_h, img_w],
                cfg=vit_config,
                num_channel=3,
                img_h=img_h,
                img_w=img_w,
            )
            return encoder
        raise ValueError(f"Unsupported vision backbone: {config.vision_backbone}")

    def _get_vision_feature_dim(self, backbone_name):
        """获取 视觉编码器输出维度."""
        if backbone_name == "resnet18" or backbone_name == "resnet34":
            return 512
        if backbone_name == "resnet50":
            return 2048
        if backbone_name == "vit":
            # 对于 ViT，展平 patch: repr_dim = embed_dim * num_patches
            return self.vision_encoder.repr_dim
        return 512  # Default
    def encode_time_pair(self, t: Tensor, r: Tensor) -> Tensor:
        """编码 MeanFlow 时间输入。

        默认返回 [emb(t), emb(r)]；开启 meanflow_encode_t_minus_r 时返回
        [emb(t), emb(r), emb(t-r)]。这里 t-r 是额外区间特征，不替换 r。
        """
        t_flat = t.reshape(-1)
        r_flat = r.reshape(-1)

        t_emb = self.time_t_encoder(t_flat)
        r_emb = self.time_r_encoder(r_flat)

        if not self.meanflow_encode_t_minus_r:
            return torch.cat([t_emb, r_emb], dim=-1)

        t_minus_r_flat = (t - r).reshape(-1)
        t_minus_r_emb = self.time_t_minus_r_encoder(t_minus_r_flat)
        return torch.cat([t_emb, r_emb, t_minus_r_emb], dim=-1)

    def encode_observations(self, batch: dict[str, Tensor]) -> Tensor:
        """将观测值编码到一个条件向量中"""
        B = next(iter(batch.values())).shape[0]
        features = []

        # 编码图像 observation
        if self.config.image_features:
            for img_key in self.config.image_features:
                if img_key in batch:
                    img_tensor = batch[img_key]
                    # img_tensor shape: (B, T, C, H, W) or (B, C, H, W)
                    if len(img_tensor.shape) == 5:
                        B, T = img_tensor.shape[:2]
                        img_flat = img_tensor.flatten(end_dim=1)  # (B*T, C, H, W)
                    else:
                        B = img_tensor.shape[0]
                        T = 1
                        img_flat = img_tensor  # Already (B, C, H, W)

                    # 通过视觉编码器
                    if self.config.vision_backbone == "vit":
                        # ViT 期望 0-255 输入 and 返回 (B, num_patches, embed_dim)
                        # 需要展平 to (B, repr_dim)
                        img_features = self.vision_encoder(img_flat * 255.0, flatten=True)  # (B*T, repr_dim) or (B, repr_dim)
                    else:
                        # ResNet 已经输出 (B, D)
                        img_features = self.vision_encoder(img_flat)  # (B*T, D) or (B, D)

                    # 重塑回原形状
                    if len(img_tensor.shape) == 5:
                        img_features = img_features.reshape(B, T, -1)  # (B, T, D)
                    else:
                        img_features = img_features.reshape(B, 1, -1)  # (B, 1, D)
                    features.append(img_features)

        # 编码状态 observation
        if self.config.state_features:
            for state_key in self.config.state_features:
                if state_key in batch:
                    state_tensor = batch[state_key]  # (B, T, D) or (B, D)
                    # 确保为 3D tensor (B, T, D)
                    if len(state_tensor.shape) == 2:
                        state_tensor = state_tensor.unsqueeze(1)  # (B, 1, D)
                    features.append(state_tensor)

        # 拼接所有 feature
        if features:
            # 确保所有 feature 具有相同时间维度
            max_T = max(f.shape[1] for f in features)
            padded_features = []
            for f in features:
                if f.shape[1] < max_T:
                    # 重复最后一帧以匹配时间维度
                    padding = f[:, -1:, :].repeat(1, max_T - f.shape[1], 1)
                    f = torch.cat([f, padding], dim=1)
                padded_features.append(f)

            obs_features = torch.cat(padded_features, dim=-1)  # (B, T, total_dim)
            # 展平时间维度用于全局 conditioning
            obs_cond = obs_features.flatten(start_dim=1)  # (B, T * total_dim)
        else:
            obs_cond = torch.zeros((B, 1), device=next(iter(batch.values())).device)

        return obs_cond

    def forward(self, x_t: Tensor, t_emb: Tensor, obs_cond: Tensor) -> Tensor:
        """
        前向传播 through the MLP.

        Args:
            x_t: (B, T, D) noisy actions where T is horizon and D is action_dim
            t_emb: (B, time_embed_dim) time embedding (already encoded)
            obs_cond: (B, obs_dim) observation conditioning

        Returns:
            (B, T, D) predicted velocity or x0
        """
        B, T, D = x_t.shape
        original_T = T
        if T < self.config.horizon:
            padding = x_t[:, -1:, :].repeat(1, self.config.horizon - T, 1)
            x_t = torch.cat([x_t, padding], dim=1)
            T = self.config.horizon
        elif T > self.config.horizon:
            raise ValueError(f"Action horizon {T} exceeds configured horizon {self.config.horizon}")

        # 展平 action to (B, T*D)
        x_t_flat = x_t.reshape(B, -1)

        # 拼接所有输入（t_emb 已提前编码）
        mlp_input = torch.cat([x_t_flat, t_emb, obs_cond], dim=-1)  # (B, input_dim)

        # 通过 MLP
        output = self.mlp(mlp_input)  # (B, T*D)

        # 重塑回原形状 to (B, T, D)
        output = output.reshape(B, T, D)

        return output[:, :original_T, :]

    def initialize_layers(self, init_fn=None):
        """
        初始化 all layers in the MLP with a given initialization function.
        If no function is provided, uses Kaiming normal initialization for Linear layers.
        """
        def default_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        init = init_fn if init_fn is not None else default_init

        # 初始化 MLP 层
        self.mlp.apply(init)
        # 初始化时间编码器
        if self.meanflow_separate_time_encoders:
            self.time_r_encoder.apply(init)
            self.time_t_encoder.apply(init)
            self.time_t_minus_r_encoder.apply(init)
        else:
            self.time_r_encoder.apply(init)
