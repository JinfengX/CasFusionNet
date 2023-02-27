from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy


class SSCLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        dataset: str,
        class_num: int,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net", "loss"])

        self.net = net

        # loss function
        self.criterion = loss

        self.data_set = dataset

        # metric objects for calculating and averaging acc,miou across batches
        self.train_mAcc = MulticlassAccuracy(class_num)
        self.train_mIoU = MulticlassJaccardIndex(class_num)
        self.val_mAcc = MulticlassAccuracy(class_num)
        self.val_mIoU = MulticlassJaccardIndex(class_num)
        self.test_mAcc = MulticlassAccuracy(class_num)
        self.test_mIoU = MulticlassJaccardIndex(class_num)

        # for averaging loss across batches
        self.train_cd = MeanMetric()
        self.train_seg = MeanMetric()
        self.val_cd = MeanMetric()
        self.val_seg = MeanMetric()
        self.test_cd = MeanMetric()
        self.test_seg = MeanMetric()

        self.val_cd_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_cd_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y, self.current_epoch)
        sum_loss, last_cd, last_seg = (
            loss["sum_loss"],
            loss["last_cd"],
            loss["last_seg"],
        )
        preds, gt_seg = loss["pred_seg"], loss["gt_seg"]

        return (sum_loss, last_cd, last_seg), preds, gt_seg

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        loss, last_cd, last_seg = loss[0], loss[1], loss[2]
        preds = preds if self.data_set == "ssc_pc" else preds[:, :, 1:]
        preds = preds.transpose(1, 2)
        # update and log metrics
        self.train_cd(last_cd)
        self.train_seg(last_seg)
        targets = targets if self.data_set == "ssc_pc" else targets - 1
        _ = self.train_mAcc(preds, targets)
        _ = self.train_mIoU(preds, targets)
        self.log("train/cd", self.train_cd, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/seg", self.train_seg, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/mAcc", self.train_mAcc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/mIoU", self.train_mIoU, on_step=False, on_epoch=True, prog_bar=True
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        loss, last_cd, last_seg = loss[0], loss[1], loss[2]
        preds = preds if self.data_set == "ssc_pc" else preds[:, :, 1:]
        preds = preds.transpose(1, 2)
        # update and log metrics
        self.val_cd(last_cd)
        self.val_seg(last_seg)
        targets = targets if self.data_set == "ssc_pc" else targets - 1
        _ = self.val_mAcc(preds, targets)
        _ = self.val_mIoU(preds, targets)
        self.log("val/cd", self.val_cd, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/seg", self.val_seg, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mAcc", self.val_mAcc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mIoU", self.val_mIoU, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        val_cd = self.val_cd.compute()
        self.val_cd_best(val_cd)  # update best so far val acc
        val_best = self.val_cd_best.compute()
        if val_cd == val_best:
            # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log("val_best/cd", self.val_cd_best.compute(), prog_bar=True)
            self.log("val_best/seg", self.val_seg.compute(), prog_bar=True)
            self.log("val_best/mAcc", self.val_mAcc.compute(), prog_bar=True)
            self.log("val_best/mIoU", self.val_mIoU.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        loss, last_cd, last_seg = loss[0], loss[1], loss[2]
        preds = preds if self.data_set == "ssc_pc" else preds[:, :, 1:]
        preds = preds.transpose(1, 2)
        # update and log metrics
        self.test_cd(last_cd)
        self.test_seg(last_seg)
        targets = targets if self.data_set == "ssc_pc" else targets - 1
        _ = self.test_mAcc(preds, targets)
        _ = self.test_mIoU(preds, targets)
        # acc = accuracy_score(targets, preds)
        # miou = jaccard_score(targets, preds, average="macro")
        self.log("test/cd", self.test_cd, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/seg", self.test_seg, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/mAcc", self.test_mAcc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test/mIoU", self.test_mIoU, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
