from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ForensicsDataset(BaseSegDataset):
    """COD dataset.

    In segmentation map annotation for Forgery, 0 stands for background,
    255 stands for camouflage object .
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
     global_classe = 1 for fake, 0 for real
    """

    METAINFO = dict(
        classes=('original', 'tampered'),
        palette=[0, 255],
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 global_class=1,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            global_class=global_class,
            **kwargs)
