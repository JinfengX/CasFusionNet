import logging
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class SSCDataModule(LightningDataModule):
    def __init__(
        self,
        dataset,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super(SSCDataModule, self).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = dataset

        # data transformations
        self.transforms = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        if stage == "fit" and not self.data_train and not self.data_val:
            self.data_train = self.dataset(
                data_path=self.hparams.data_dir, train=True, transform=self.transforms
            )
            logging.info(f"Train dataset loaded, size: {len(self.data_train)}")
            self.data_val = self.dataset(
                data_path=self.hparams.data_dir, train=False, transform=self.transforms
            )
            logging.info(f"Validate dataset loaded, size: {len(self.data_val)}")
        # Assign test dataset for use in dataloader(s)
        if stage == "test" and not self.data_test:
            self.data_test = self.dataset(
                self.hparams.data_dir, train=False, transform=self.transforms
            )
            logging.info(f"Test dataset loaded, size: {len(self.data_test)}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
