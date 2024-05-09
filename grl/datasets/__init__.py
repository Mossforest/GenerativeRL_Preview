from .d4rl import D4RLDataset
from .minecraft import MineRLImageDataset, MineRLVideoDataset
from .qgpo import QGPOCustomizedDataset, QGPOD4RLDataset, QGPODataset, QGPOOnlineDataset
from .minari_dataset import MinariDataset

DATASETS = {
    "MineRLVideoDataset".lower(): MineRLVideoDataset,
    "MineRLImageDataset".lower(): MineRLImageDataset,
    "QGPOD4RLDataset".lower(): QGPOD4RLDataset,
    "QGPODataset".lower(): QGPODataset,
    "D4RLDataset".lower(): D4RLDataset,
    "QGPOOnlineDataset".lower(): QGPOOnlineDataset,
    "QGPOCustomizedDataset".lower(): QGPOCustomizedDataset,
}


def get_dataset(type: str):
    if type.lower() not in DATASETS:
        raise KeyError(f"Invalid dataset type: {type}")
    return DATASETS[type.lower()]


def create_dataset(config, **kwargs):
    return get_dataset(config.type)(**config.args, **kwargs)
