import os
from torch import nn
import torch
from typing import List, Any, Dict

#from common.utils import *
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from clearml import Dataset as ClearML_Dataset
from common.utils import *
from models.RENET import RENet
import ipdb


class GraphModel(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""

    def __init__(self, cfg, task):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.clearml_logger = self.task.get_logger()
        self.model = RENet(cfg)
        self.metrics = {}

    def forward(self, batch):
        batch_data = batch['s']
        ce_loss_s, c_loss_s = self.model(batch_data, subject=True)
        batch_data = batch['o']
        ce_loss_o, c_loss_o = self.model(batch_data, subject=False)
        ce_loss = (ce_loss_s + ce_loss_o) / 2
        c_loss = (c_loss_s + c_loss_o) / 2
        loss = ce_loss + self.cfg["c_lambda"] * c_loss
        return loss

    def training_step(self, batch, batch_nb: int):
        """Call the forward pass then return loss"""
        loss = self(batch)
        return {"loss": loss}

    # def training_epoch_end(self, outputs: List):
    #     total_loss = []
    #     for batch in outputs:
    #         total_loss.append(batch["loss"])
    #     self.log("train_loss", sum(total_loss) / len(total_loss))

    def eval_step(self, batch):
        batch_result = torch.tensor(
            [0, 0, 0, 0], dtype=torch.float)

        obj_pred, groundtruth_obj = self.model(
            batch['s'], subject=True, return_prob=True)

        sub_pred, groundtruth_sub = self.model(
            batch['o'], subject=False, return_prob=True)

        batch_result += self.model.test(obj_pred, groundtruth_obj) * \
            groundtruth_obj.size(0)

        batch_result += self.model.test(sub_pred, groundtruth_sub) * \
            groundtruth_sub.size(0)

        return batch_result

    def validation_step(self, batch, batch_nb: int):
        """Call the forward pass then return loss"""
        batch_results = self.eval_step(batch)
        return {'batch_results': batch_results}

    def validation_epoch_end(self, outputs: List):
        total_results = torch.tensor(
            [0, 0, 0, 0], dtype=torch.float)
        for batch in outputs:
            total_results += batch['batch_results']
        total_results = total_results / \
            (2*len(outputs)*self.cfg.batch_size)
        self.log("MRR", total_results[0])
        self.log("Hits@1", total_results[1])
        self.log("Hits@3", total_results[2])
        self.log("Hits@10", total_results[3])

    def test_step(self, batch, batch_nb: int):
        """Call the forward pass then return loss"""
        batch_results = self.eval_step(batch)
        return {'batch_results': batch_results}

    def test_epoch_end(self, outputs: List):
        total_results = torch.tensor(
            [0, 0, 0, 0], dtype=torch.float)
        for batch in outputs:
            total_results += batch['batch_results']
        total_results = total_results / \
            (2*(len(outputs)*self.cfg.batch_size)).tolist()
        self.log("MRR", total_results[0])
        self.log("Hits@1", total_results[1])
        self.log("Hits@3", total_results[2])
        self.log("Hits@10", total_results[3])

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wd)

        return {
            "optimizer": optimizer,
        }
