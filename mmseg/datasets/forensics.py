# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseTampDataset

@DATASETS.register_module()
class ForensicsDataset(BaseTampDataset):
    """ forensics dataset.

    In segmentation map annotation for Forensics, 0 stands for background,
    255 stands for tampered object .
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('original', 'tampered'),
        palette=([0,0,0],[255,255,255]))

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            ignore_index=255,
            **kwargs)
