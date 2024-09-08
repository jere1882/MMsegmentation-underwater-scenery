from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class UnderwaterScenery(BaseSegDataset):
    """UnderwaterScenery dataset.
    """
    METAINFO = dict(
        classes=(
            'sea_floor_and_rocks', \
            'divers', \
            'wrecks_and_ruins', \
            'vegetation', \
            'fish_and_vertebrates', \
            'reefs_and_invertebrates', \
            'robots',
            'background'),
        palette=[ \
            [255, 255, 255],
            [0, 0, 255],
            [0, 255, 255],
            [0, 255, 0],
            [255, 255, 0],
            [255, 0, 255],
            [255, 0, 0],
            [0,0,0]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
