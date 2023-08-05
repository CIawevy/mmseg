# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union
import mmengine
import cv2
import mmcv
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.utils import is_tuple_of
from numpy import random
from scipy.ndimage import gaussian_filter

from mmseg.datasets.dataset_wrappers import MultiImageMixDataset
from mmseg.registry import TRANSFORMS
import mmcv




@TRANSFORMS.register_module()
class SelfRandomResize(BaseTransform):


    def __init__(
        self,
        ratio_range: Tuple[float, float] = (0.5,1.5),
        backend: str = 'cv2',
        interpolation='bilinear',
    ) -> None:
        self.ratio_range = ratio_range
        self.backend = backend
        self.interpolation = interpolation

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, ``img``, ``gt_bboxes``, ``gt_semantic_seg``,
            ``gt_keypoints``, ``scale``, ``scale_factor``, ``img_shape``, and
            ``keep_ratio`` keys are updated in result dict.
        """

        H,W = results['img'].shape[:2]
        min_ratio, max_ratio = self.ratio_range
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(W * ratio),int(H * ratio)
        img=mmcv.imresize(img=results["img"],size=scale,interpolation=self.interpolation,backend=self.backend)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        gt_seg=mmcv.imresize(img=results['gt_seg_map'],size=scale,interpolation='nearest',backend=self.backend)#mmcv.imresize 必须W，H的顺序
        results['gt_seg_map'] = gt_seg
        return results

@TRANSFORMS.register_module()
class JPEGcompression(BaseTransform):


    def __init__(
        self,
        quality_factor: Tuple[float, float] = (30,100),

    ) -> None:
        self.quality_factor = quality_factor


    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, ``img``, ``gt_bboxes``, ``gt_semantic_seg``,
            ``gt_keypoints``, ``scale``, ``scale_factor``, ``img_shape``, and
            ``keep_ratio`` keys are updated in result dict.
        """
        min,max=self.quality_factor
        factor=np.random.randint(low=min,high=max)
        params = [cv2.IMWRITE_JPEG_QUALITY, factor]  # factor:0~100
        msg = cv2.imencode(".jpg", results["img"], params)[1]
        msg = (np.array(msg)).tobytes()
        img = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
        results["img"]=img
        return results