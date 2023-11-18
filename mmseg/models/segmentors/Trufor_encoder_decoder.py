# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional
import os
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmseg.structures import SegDataSample
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from ..backbones import make_net
import torch
from mmengine.structures import BaseDataElement, PixelData
from mmengine.structures import PixelData
from ..utils import resize
import logging

@MODELS.register_module()
class MyEncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSampel`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        self._init_decode_head(decode_head)

        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        num_levels = 17
        out_channel = 1
        self.dncnn = make_net(3, kernels=[3, ] * num_levels,
                              features=[64, ] * (num_levels - 1) + [out_channel],
                              bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
                              acts=['relu', ] * (num_levels - 1) + ['linear', ],
                              dilats=[1, ] * num_levels,
                              bn_momentum=0.1, padding=1)
        self.dncnn_init_weights() #Dncnn载入权重
        assert self.with_decode_head
        self.stage=self.backbone.stage
        self._freeze_stages()




    def _freeze_stages(self):
        # freeze position embedding
        if self.stage == 'detection':
            self.backbone.eval()
            self.backbone.requires_grad_(False)
            self.decode_head.Seg_decoder.eval()
            self.decode_head.Seg_decoder.requires_grad_(False)

        self.dncnn.eval()
        self.dncnn.requires_grad_(False)
    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners #为了slide推理 暂时不用管
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels
        self.use_conf=self.decode_head.conf_cfg
        self.is_detect_stage=self.decode_head.use_detect_head

    def dncnn_init_weights(self):
        np_weights = "/home/ipad_ind/hszhu/pretrained/NP++.pth"
        assert os.path.isfile(np_weights)
        dat = torch.load(np_weights)
        print(f'Noiseprint++ weights: {np_weights}')
        if 'network' in dat:
            dat = dat['network']
        self.dncnn.load_state_dict(dat)


    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor ,modal_x:Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs,modal_x)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        modal_x = self.dncnn(inputs)
        modal_x = torch.tile(modal_x, (3, 1, 1))
        inputs = self.preprc_imagenet_torch(inputs)  # imagenet标准化 在/256.0之后
        x = self.extract_feat(inputs ,modal_x)
        self.forward_size=modal_x.shape[2:]
        seg_logits, conf, det = self.decode_head.predict(x, batch_img_metas, #自动上采样
                                              self.forward_size,
                                              self.test_cfg)

        return seg_logits, conf, det ,modal_x

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,self.forward_size,self.train_cfg,)

        if self.use_conf and self.is_detect_stage:
            losses.update(add_prefix(loss_decode, 'detect_decode'))
        else:
            losses.update(add_prefix(loss_decode, 'Segdecode'))

        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def preprc_imagenet_torch(self,x):
        mean = torch.Tensor([0.485, 0.456, 0.406]).to(x.device)
        std = torch.Tensor([0.229, 0.224, 0.225]).to(x.device)
        x = (x - mean[None, :, None, None]) / std[None, :, None, None]
        return x

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        modal_x=self.dncnn(inputs)
        modal_x = torch.tile(modal_x, (3, 1, 1))
        inputs=self.preprc_imagenet_torch(inputs) #imagenet标准化 在/256.0之后
        x = self.extract_feat(inputs,modal_x)
        self.forward_size=modal_x.shape[2:]
        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)



        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:#测试用
        """Predict results from a batch of inputs and data samples with post-
        processing.
        #predict函数
        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        #此处pred函数只考虑我要输出的东西 mask  detection
        seg_logits,conf,det, npp = self.inference(inputs, batch_img_metas) #conf npp 为了可视化需要
        #seg: 1 c h w ,conf 1 1 h w det 1 1 npp 1 3 h w
        npp=npp[:,0,:,:].unsqueeze(0)

           #1 1 h w
        # if det is None:
        #     return self.postprocess_result(seg_logits, data_samples) #stage 2 out predict
        #用 softmax 不用argmax所以不用原始部署，而且测试时我最多做resize做一尺寸
        #self.out(npp,conf) #conf npp 后处理就三步 去padding 去 flip  resize 回输入尺寸
        #如果输入没有任何增强，是否需要还有padding需要去除是一个问题。假设没有那么不需要后处理npp conf
        #如果要统一resize则最好在postprocess将这几个conf npp一起传给 metric
        # else:
        return self.multi_postprocess_result(seg_logits,det,conf,npp,data_samples)


    def multi_postprocess_result(self,
                        seg_logits: Tensor,
                        det :None,
                        conf: None,
                        npp: Tensor,
                        data_samples: OptSampleList = None) -> SampleList:

        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area #不需要pad 测试
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom = \
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                               padding_top:H - padding_bottom,
                               padding_left:W - padding_right]
                if conf is not None:
                    i_conf = conf[i:i + 1, :,
                                   padding_top:H - padding_bottom,
                                   padding_left:W - padding_right]
                i_npp = npp[i:i + 1, :,
                               padding_top:H - padding_bottom,
                               padding_left:W - padding_right]
                #这一步之后还是 1 c h w

                # flip = img_meta.get('flip', None)
                # if flip:
                #     flip_direction = img_meta.get('flip_direction', None)
                #     assert flip_direction in ['horizontal', 'vertical']
                #     if flip_direction == 'horizontal':
                #         i_seg_logits = i_seg_logits.flip(dims=(3,))
                #     else:
                #         i_seg_logits = i_seg_logits.flip(dims=(2,))
                #测试不flip
                # resize as original shape 测试有可能到512 这一步回到原图 虽然npp可能因此无效，所以最好测试不要resize
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
                if conf is not None:
                    i_conf = resize(
                        i_conf,
                        size=img_meta['ori_shape'],
                        mode='bilinear',
                        align_corners=self.align_corners,
                        warning=False).squeeze(0)
                i_npp = resize(
                    i_npp,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
                # i_npp = i_npp.squeeze()
            else:
                i_seg_logits = seg_logits[i]# C H W
                if conf:
                    i_conf=conf[i]
                i_npp=npp[i]

            #
            i_seg_pred = F.softmax(i_seg_logits, dim=0)[1].unsqueeze(0) #1 H W
            i_conf = torch.sigmoid(i_conf)[0] if (conf is not None) else None #1 H W
            det_sig = torch.sigmoid(det).item() if (det is not None) else None #1 1
            i_npp=i_npp # 1 H W

            data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': i_seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': i_seg_pred}),
                    'confidence_map':
                        BaseDataElement(**{'data': i_conf}),
                    'noiseprint':
                        PixelData(**{'data': i_npp}),
                    'scores':
                        BaseDataElement(**{'data': det_sig}),
                })




        return data_samples

        # 可视化输出可以写在metric里面,包括 squeeze而这一步只需要传递数据就够了 data_sample
        # out_dict = dict()
        # out_dict['map'] = pred
        # out_dict['imgsize'] = tuple(rgb.shape[2:])
        # if det is not None:
        #     out_dict['score'] = det_sig
        # if conf is not None:
        #     out_dict['conf'] = conf
        # if save_np:
        #     out_dict['np++'] = npp
        #
        # from os import makedirs
        #
        # makedirs(os.path.dirname(filename_out), exist_ok=True)
        # np.savez(filename_out, **out_dict)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:#训练用
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        modal_x = self.dncnn(inputs)
        modal_x = torch.tile(modal_x, (3, 1, 1))
        inputs = self.preprc_imagenet_torch(inputs)  # imagenet标准化 在/256.0之后
        x = self.extract_feat(inputs,modal_x)
        return self.decode_head.forward(x,modal_x.shape[2:])

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit,conf,det = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits, conf, det , npp = self.encode_decode(inputs, batch_img_metas) #seg_log与npp尺寸相同 conf det 默认是none

        return seg_logits, conf, det , npp

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit, conf, det , npp = self.whole_inference(inputs, batch_img_metas)

        return seg_logit,conf,det ,npp

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
