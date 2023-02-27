from functools import partial
from pathlib import Path

import pytest
import torch

from src.data.ssc_datamodule import SSCDataModule
from src.data.components.nyucad_pc import NYUCAD
from src.data.components.ssc_pc import SSCPC


@pytest.mark.parametrize("batch_size", [32, 128])
def test_nyucad_datamodule(batch_size):
    data_dir = "/home/docker/CasFusionNet/data/NYUCAD-PC"
    dataset = partial(NYUCAD)
    dm = SSCDataModule(dataset, data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "train").exists()
    assert Path(data_dir, "test").exists()

    dm.setup(stage="fit")
    dm.setup(stage="test")
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 795 + 654 + 654

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


@pytest.mark.parametrize("batch_size", [32, 128])
def test_sscpc_datamodule(batch_size):
    data_dir = "/home/docker/CasFusionNet/data/SSC-PC"
    dataset = partial(SSCPC)
    dm = SSCDataModule(dataset, data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "Bathroom").exists()
    assert Path(data_dir, "Bedroom").exists()
    assert Path(data_dir, "Livingroom").exists()
    assert Path(data_dir, "Office").exists()

    dm.setup(stage="fit")
    dm.setup(stage="test")
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 1543 + 398 + 398

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32