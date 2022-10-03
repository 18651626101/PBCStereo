from .kitti_dataset import KITTIDataset
from .kitti12_dataset import KITTI_12_Dataset
from .sceneflow_dataset import SCENEFLOW_Dataset
from .kitti2015submission import KITTI2015submission

__datasets__ = {
    "kitti": KITTIDataset,
    "kitti2015submission": KITTI2015submission,
    "kitti_12": KITTI_12_Dataset,
    "sceneflow": SCENEFLOW_Dataset,
}
