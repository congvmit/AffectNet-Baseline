import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import pytorch_lightning as pl
from affectnet.losses import get_loss_fn
from affectnet.opts import get_optimizer
from affectnet.schedulers import get_scheduler


class BasedModel(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.hparams = vars(args)
        # self.save_hyperparameters(self.hparams)
        self.args = args
        self.model = model
        self.loss_fn = get_loss_fn(self.args.loss_fn)
        self.tacc = pl.metrics.Accuracy()
        self.vacc = pl.metrics.Accuracy()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.args.optimizer, self.model, self.args)
        lr_scheduler = get_scheduler(self.args.lr_scheduler, self.args)
        if lr_scheduler is not None:
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            return {'optimizer': optimizer}

    def training_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['expression']

        pred = self.forward(x=image)
        
        # Loss
        loss = self.loss_fn(pred, target)

        # Acc
        pred = torch.softmax(pred, dim=1)
        tacc = self.vacc(pred, target)

        self.log('tloss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('tacc', tacc, prog_bar=True, on_step=True, on_epoch=True)

        return {'target': target,
                'pred': pred,
                'loss': loss
                }

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['expression']

        pred = self.forward(x=image)

        # Loss
        loss = self.loss_fn(pred, target)
        
        # Acc
        pred = torch.softmax(pred, dim=1)
        vacc = self.vacc(pred, target)

        self.log('vloss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('vacc', vacc, prog_bar=True, on_step=False, on_epoch=True)

        return {'target': target,
                'pred': pred,
                'loss': loss
                }


    # def test_step(self, batch, batch_idx):
    #     image = batch['image']
    #     target = batch['expression']

    #     pred = self.forward(x=image)

    #     # Loss
    #     loss = self.loss_func(pred, target)

    #     self.log("test_loss", self.test_acc, on_step=False, on_epoch=True)

    #     return {'target': target,
    #             'pred': pred,
    #             'loss': loss
    #             }
