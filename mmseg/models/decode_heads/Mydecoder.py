# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import warnings
import math
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize

class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@MODELS.register_module()
class DesignedDecoder(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.
    this upsample is implementation of
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py#L451

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self,input_channel=1280,feature_pyramid=True,pyramid_scales=[4.0, 2.0, 1.0, 0.5] ,
                 pyr_out_channel=256,
                 interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.hieracical_feature=feature_pyramid
        self.interpolate_mode = interpolate_mode
        self.input_channel=input_channel
        self.pyramid_out=pyr_out_channel
        #pyramid sampling
        if self.hieracical_feature:
            self.pyramid_branches = nn.ModuleList()
            self.pyra_out=[]
            self.sample_scales=pyramid_scales
            dim=self.input_channel
            out_channels=self.pyramid_out
            for idx, scale in enumerate(self.sample_scales):
                out_dim = dim
                if scale == 4.0:
                    layers = [
                        nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                        LayerNorm(dim // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                    ]
                    out_dim = dim // 4
                elif scale == 2.0:
                    layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                    out_dim = dim // 2
                elif scale == 1.0:
                    layers = []
                elif scale == 0.5:
                    layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

                layers.extend(
                    [
                        ConvModule(
                            out_dim,
                            out_channels,
                            kernel_size=1,
                            norm_cfg=self.norm_cfg,

                        ),
                        ConvModule(
                            out_channels,
                            out_channels,
                            kernel_size=3,
                            padding=1,
                            norm_cfg=self.norm_cfg,
                        ),
                    ]
                )
                layers = nn.Sequential(*layers)
                # self.add_module(f"Pyramid_{idx}", layers)
                self.pyra_out.append(out_channels)
                self.pyramid_branches.append(layers) #模型没有载入cuda

        #seghead
        num_inputs = len(pyramid_scales)
        self.in_index=[0,1,2,3]
        assert num_inputs == len(self.in_index)
        self.in_channels=self.pyra_out
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,  #segformer 512
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    # def _init_weights(self, m):#prompt generator
    #     if isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32 256,256,256,256
        results=[]
        inputs = inputs[0]
        for layer in self.pyramid_branches:
            results.append(layer(inputs))
        outs = []
        inputs = self._transform_inputs(results)
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    # input=x,
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1)) #1,512,512,512

        out = self.cls_seg(out) #1,2,512,512

        return out
