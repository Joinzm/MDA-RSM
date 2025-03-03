import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from torch.cuda.amp import autocast, GradScaler
from hilbert import decode, encode
from pyzorder import ZOrderIndexer
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from .mda_block import MDASSBlock

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Decoder_Block(nn.Module):
    """Basic block in decoder."""

    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert (
            out_channel == in_channel // 2
        ), "the out_channel is not in_channel//2 in decoder block"
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel + out_channel,
                out_channels=out_channel,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, de, en):
        de = self.up(de)
        output = torch.cat([de, en], dim=1)
        output = self.fuse(output)

        return output


class MDA_RSM(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        # =========================
        drop_path_rate=0.1,
        patch_norm=True,
        norm_layer="LN",
        use_checkpoint=False,
        # ming
        scan_types=["zorder"],
        image_size=512,
        per_scan_num=8,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2**i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        if isinstance(norm_layer, str) and norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if isinstance(mlp_act_layer, str) and mlp_act_layer.lower() in [
            "silu",
            "gelu",
            "relu",
        ]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        _make_patch_embed = self._make_patch_embed_v2
        self.patch_embed = _make_patch_embed(
            in_chans, dims[0], patch_size, patch_norm, norm_layer
        )

        _make_downsample = self._make_downsample_v3

        # self.encoder_layers = [nn.ModuleList()] * self.num_layers
        self.encoder_layers = []
        self.decoder_layers = []

        for i_layer in range(self.num_layers):
            # downsample = _make_downsample(
            #     self.dims[i_layer],
            #     self.dims[i_layer + 1],
            #     norm_layer=norm_layer,
            # ) if (i_layer < self.num_layers - 1) else nn.Identity()

            downsample = (
                _make_downsample(
                    self.dims[i_layer - 1],
                    self.dims[i_layer],
                    norm_layer=norm_layer,
                )
                if (i_layer != 0)
                else nn.Identity()
            )  # ZSJ 修改为i_layer != 0，也就是第一层不下采样，和论文的图保持一致，也方便我取出每个尺度处理好的特征

            self.encoder_layers.append(
                self._make_layer(
                    dim=self.dims[i_layer],
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    use_checkpoint=use_checkpoint,
                    norm_layer=norm_layer,
                    downsample=downsample,
                    # =================
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    # =================
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    # ming
                    size=int(image_size / 4) // (2**i_layer),
                    scan_types=scan_types,
                    per_scan_num=per_scan_num,
                )
            )
            if i_layer != 0:
                self.decoder_layers.append(
                    Decoder_Block(
                        in_channel=self.dims[i_layer],
                        out_channel=self.dims[i_layer - 1],
                    )
                )

        (
            self.encoder_block1,
            self.encoder_block2,
            self.encoder_block3,
            self.encoder_block4,
        ) = self.encoder_layers
        self.deocder_block1, self.deocder_block2, self.deocder_block3 = (
            self.decoder_layers
        )

        self.upsample_x4 = nn.Sequential(
            nn.Conv2d(
                self.dims[0], self.dims[0] // 2, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(self.dims[0] // 2),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.dims[0] // 2, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.conv_out_seg = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed_v2(
        in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm
    ):
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96,
        drop_path=[0.1, 0.1],
        use_checkpoint=False,
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        size=8,
        scan_types=[""],
        per_scan_num=8,
        **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(
                MDASSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    use_checkpoint=use_checkpoint,
                    # ming
                    size=size,
                    scan_types=scan_types,
                    per_scan_num=per_scan_num,
                )
            )

        return nn.Sequential(
            OrderedDict(
                downsample=downsample,
                blocks=nn.Sequential(
                    *blocks,
                ),
            )
        )

    def forward(self, x1: torch.Tensor):
        x1 = self.patch_embed(x1)
        x1_1 = self.encoder_block1(x1)
        x1_2 = self.encoder_block2(x1_1)
        x1_3 = self.encoder_block3(x1_2)
        x1_4 = self.encoder_block4(x1_3)  # b,h,w,c

        x1_1 = rearrange(x1_1, "b h w c -> b c h w").contiguous()
        x1_2 = rearrange(x1_2, "b h w c -> b c h w").contiguous()
        x1_3 = rearrange(x1_3, "b h w c -> b c h w").contiguous()
        x1_4 = rearrange(x1_4, "b h w c -> b c h w").contiguous()

        decode_3 = self.deocder_block3(x1_4, x1_3)
        decode_2 = self.deocder_block2(decode_3, x1_2)
        decode_1 = self.deocder_block1(decode_2, x1_1)

        output = self.upsample_x4(decode_1)
        output = self.conv_out_seg(output)

        return output


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    net = MDA_RSM(
        in_chans=3,
        image_size=512,
        patch_size=4,
        num_classes=1000,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        # =========================
        drop_path_rate=0.1,
        patch_norm=True,
        norm_layer="LN",
        use_checkpoint=False,
        # ming
        scan_types=["zorder", "zigzag"],
        per_scan_num=2,
    )
    net.to("cuda")
    image = torch.randn((8, 3, 512, 512)).cuda()
    print(image.shape)
    x = net(image)
    print(x.shape)
