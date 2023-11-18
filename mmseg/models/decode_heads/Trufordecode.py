from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


from ..losses import accuracy
from ..builder import build_loss
import torch
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList

from mmcv.cnn import ConvModule
import warnings
import math
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Linear Embedding:
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DecoderHead(nn.Module):#需要实现anomaly decoder和 confidence decoder的不同loss计算
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 in_index=[0,1,2,3],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_cfg=None,
                 embed_dim=768,
                 align_corners=False
                 ):              #继承自basedecoder自动实现loss 我在这里覆写即可。

        super(DecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        self.in_index=in_index
        self.channels=embed_dim
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.in_channels = in_channels
        self.norm_cfg=norm_cfg



        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # self.linear_fuse = nn.Sequential(
        #     nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
        #     norm_layer(embedding_dim),
        #     nn.ReLU(inplace=True)
        # )
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.linear_fuse=ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            bias=True,
            norm_cfg=self.norm_cfg)
        #default Relu activation
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, return_feats=False):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3]).contiguous()
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3]).contiguous()
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3]).contiguous()
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3]).contiguous()

        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        x = self.linear_fuse(_c)
        x = self.dropout(x)
        x = self.linear_pred(x)
        # x = self.cls_seg(x) #包括drop以及linear_pred

        if return_feats:
            return x, _c
        else:
            return x
"""
    decode_head=dict(
        type='DualDecoderHead', #
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        conf_cfg=True,  #CONF
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=3.0,class_weight=[0.5,2.5]),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=7.0)
        ]),
    """


@MODELS.register_module()
class DualDecoderHead(BaseDecodeHead):#需要实现anomaly decoder和 confidence decoder的不同loss计算
    def __init__(self,conf_cfg:bool=False,
                 detect_head_cfg=False,
                 det_loss_decode=dict(),
                 **kwargs,
                 ):              #继承自basedecoder自动实现loss 我在这里覆写即可。

        super().__init__(input_transform='multiple_select', **kwargs)
        #传参进入self.loss_decode已经变成nn module_list
        """
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
        """
        self.conf_cfg=conf_cfg
        self.use_detect_head=detect_head_cfg
        if self.conf_cfg and self.use_detect_head:#如果使用conf以及detect则使用phase 3 detect 损失 损失需要在损失模块定义并注册
            if isinstance(det_loss_decode, dict):
                self.det_loss_decode = build_loss(det_loss_decode)
            elif isinstance(det_loss_decode, (list, tuple)):
                self.det_loss_decode = nn.ModuleList()
                for loss in det_loss_decode:
                    self.det_loss_decode.append(build_loss(loss))
            else:
                raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                    but got {type(det_loss_decode)}')

        self.Seg_decoder=DecoderHead(
            in_channels=self.in_channels,
            in_index=self.in_index,
            embed_dim=self.channels,
            dropout_ratio=self.dropout_ratio,
            num_classes=self.num_classes,
            norm_cfg=self.norm_cfg,
            align_corners=self.align_corners
        )
        if self.conf_cfg:
            self.Conf_decoder = DecoderHead(
                in_channels=self.in_channels,
                in_index=self.in_index,
                embed_dim=self.channels,
                dropout_ratio=self.dropout_ratio,
                num_classes=1,
                norm_cfg=self.norm_cfg,
                align_corners=self.align_corners
            )
        else:
            self.Conf_decoder =None

        if self.use_detect_head:
            self.detection_head = nn.Sequential(
                nn.Linear(in_features=8, out_features=128),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=128, out_features=1),
            )
        else:
            self.detection_head=None
        self.conv_seg=None #这里有一个坑，由于mmseg basedecode 初始化自动把 num_class作为out_channels之后构建
        #conv_seg用于最后的分类头,这里我使用双decode且自己构建函数，需要把原先规定的一个decode自动输出函数去掉，不然多卡
        #训练的时候 self.conv_seg没有梯度回传会报错



    def zhs_weighted_statistics_pooling(self,x, log_w=None, pad=None):
        b = x.shape[0]
        c = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]

        padding_info=pad
        if pad is not None:
            mask = torch.full((b, 1, H, W), float("-inf"), device=x.device)
            w_mask=torch.full((b, 1, H, W), float(0), device=x.device)
            condition = torch.full((b, 1, H, W), bool(False), device=x.device)
            for i in range(pad.shape[0]):
                padding_left, padding_right, padding_top, padding_bottom = \
                    padding_info[i].int()
                condition[i, :, padding_top:(H - padding_bottom), padding_left:(W - padding_right)] = True

        x = x.view(b, c, -1)  # 展平了
        if pad is not None:
            mask=mask.view(b, c, -1)
            w_mask = w_mask.view(b ,c ,-1)
            condition=condition.view(b, c, -1)
            x_p = torch.where(condition, x, mask).to(x.device)
            x_n = torch.where(condition, -x, mask).to(x.device)
            x_w  = torch.where(condition , x , w_mask ).to(x.device)
        else:
            x_p , x_n , x_w = x , -x , x
        if log_w is None:
            log_w = torch.zeros((b, 1, x.shape[-1]), device=x.device)
        else:
            assert log_w.shape[0] == b
            assert log_w.shape[1] == 1
            log_w = log_w.view(b, 1, -1)

            assert log_w.shape[-1] == x.shape[-1]


        log_w = F.log_softmax(log_w, dim=-1) #LogSoftmax其实就是对softmax的结果进行log，即Log(Softmax(x))
        # x_min = -torch.logsumexp(log_w - x, dim=-1)
        # x_max = torch.logsumexp(log_w + x, dim=-1)
        new_x_min = -torch.logsumexp(log_w + x_n, dim=-1)
        new_x_max = torch.logsumexp(log_w + x_p, dim=-1)

        w = torch.exp(log_w)
        # x_avg = torch.sum(w * x, dim=-1)
        # x_msq = torch.sum(w * x * x, dim=-1)
        new_x_avg = torch.sum(w * x_w, dim=-1)
        new_x_msq = torch.sum(w * x_w * x_w, dim=-1)

        x = torch.cat((new_x_min, new_x_max, new_x_avg, new_x_msq), dim=1)

        return x

    def forward(self, inputs, ori_shape,pad_list=None):
        seg_logits = self.Seg_decoder.forward(inputs)
        seg_logits = resize(
            input=seg_logits,
            size=ori_shape,
            mode='bilinear',
            align_corners=self.align_corners)

        if self.Conf_decoder is not None:
            conf = self.Conf_decoder.forward(inputs)
            conf = resize(
                input=conf,
                size=ori_shape,
                mode='bilinear',
                align_corners=self.align_corners)
        else:
            conf=None
        if self.detection_head is not None:
            #插入一个crop padding 函数
            f1 = self.zhs_weighted_statistics_pooling(conf,pad=pad_list).view(seg_logits.shape[0],-1)
            f2 = self.zhs_weighted_statistics_pooling(seg_logits[:,1:2,:,:]-seg_logits[:,0:1,:,:], F.logsigmoid(conf),
                                                  pad=pad_list).view(seg_logits.shape[0],-1)
            det = self.detection_head(torch.cat((f1,f2),-1)) #B,1
            #8D -> 128D -> 1D
        else:
            det=None


        return seg_logits, conf , det


    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,forward_size:None,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        #self.loss_decode是一个nn list 通过cfg传递构建各种loss
        """
        
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')"""
        img_meta = [ data_sample.metainfo for data_sample in batch_data_samples]
        # remove padding area #不需要pad 测试
        padding_size_list=[]
        for i in range(len(img_meta)):
            if 'img_padding_size' not in img_meta:
                padding_size = img_meta[i].get('padding_size', [0] * 4)
            else:
                padding_size = img_meta[i]['img_padding_size']
            padding_size_list.append(torch.Tensor(padding_size))
        pad=torch.stack(padding_size_list, dim=0)
        seg_logits,conf,det= self.forward(inputs,forward_size,pad)# original shape and label
        if conf is not None and det is not None:
            losses = self.stage2_loss_by_feat(seg_logits, conf, det, batch_data_samples,pad)
        else:
            losses = self.stage1_loss_by_feat(seg_logits, batch_data_samples)
        return losses




    def _stack_batch_label(self, batch_data_samples: SampleList) -> Tensor:
        device = batch_data_samples[0].gt_sem_seg.data.device
        tensor_label = torch.tensor([
            data_sample.global_class for data_sample in batch_data_samples
        ],device=device)
        return tensor_label

    def stage2_loss_by_feat(self, seg_logits: Tensor, conf:Tensor,det:Tensor,
                     batch_data_samples: SampleList , pad: Tensor) -> dict:
        """Compute segmentation loss or detection loss depends on the stage config like conf

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples) #B 1 H W
        det_label=  self._stack_batch_label(batch_data_samples) #B
        loss = dict()
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None #None
        seg_label = seg_label.squeeze(1) #B H W

        if conf is not None or det is not None:
            # 如果有conf则使用det_loss_decode
            if not isinstance(self.det_loss_decode, nn.ModuleList):
                losses_decode = [self.det_loss_decode]
            else:
                losses_decode = self.det_loss_decode
            for loss_decode in losses_decode:
                if  loss_decode.loss_sta=='confidence':
                    if loss_decode.loss_name not in loss :
                        loss[loss_decode.loss_name] = loss_decode(
                            seg_logits,
                            conf,
                            target=seg_label,
                            pad=pad,)
                    else:
                        loss[loss_decode.loss_name] += loss_decode(
                            seg_logits,
                            conf,
                            target=seg_label,
                            pad=pad,)

                elif loss_decode.loss_sta=='detection':
                    if loss_decode.loss_name not in loss:
                        loss[loss_decode.loss_name] = loss_decode(
                            det,
                            det_label)

                    else:
                        loss[loss_decode.loss_name] += loss_decode(
                            det,
                            det_label)

                else:
                    if loss_decode.loss_name not in loss:
                        loss[loss_decode.loss_name] = loss_decode(
                            seg_logits,
                            seg_label,
                            weight=seg_weight,
                            ignore_index=self.ignore_index)
                    else:
                        loss[loss_decode.loss_name] += loss_decode(
                            seg_logits,
                            seg_label,
                            weight=seg_weight,
                            ignore_index=self.ignore_index)


        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)

        return loss

    def stage1_loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss or detection loss depends on the stage config like conf

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if True:
            if not isinstance(self.loss_decode, nn.ModuleList):
                losses_decode = [self.loss_decode]
            else:
                losses_decode = self.loss_decode
            for loss_decode in losses_decode:
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logits,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logits,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)


        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)

        return loss

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],forward_size:None,
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits , conf , det  = self.forward(inputs,forward_size)
        seg_logits=self.predict_by_feat(seg_logits, batch_img_metas) #resize 函数
        conf=self.predict_by_feat(conf, batch_img_metas) if conf is not None else  None

        return seg_logits , conf , det


