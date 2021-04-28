import random
import os
import argparse
from icecream import ic

import torch
import torch.nn as nn

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


from affectnet.models.net import get_model
from affectnet.base import BasedModel
from affectnet.dataloader import AffectNetDataloader
from affectnet.dataloader import AffectNetDataset
from affectnet.transforms import get_train_transform, get_val_transform
import mipkit

# import warnings
# warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def parse_args(add_pl_args=True, is_notebook=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'test'],
                        type=str, default='train')

    # ===============================================================================
    # Data Path Arguments
    # ===============================================================================
    # parser.add_argument('--data_dir', type=str,
    #                     help='Path to preprocessed data npy files/ csv files')

    # parser.add_argument('--train_csv_path', type=str,
    #                     help='Path to preprocessed data npy files/ csv files')

    # parser.add_argument('--val_csv_path', type=str,
    #                     help='Path to preprocessed data npy files/ csv files')

    # parser.add_argument('--test_csv_path', type=str,
    #                     help='Path to preprocessed data npy files/ csv files')

    # ===============================================================================
    # Training Arguments
    # ===============================================================================
    parser.add_argument('-c', '--config_file', type=str,
                        help='configuration file')
    parser.add_argument('-j', '--num_workers', type=int,
                        default=8, help='number of workers')

    # ===============================================================================
    # Learing Rate and Scheduler
    # ===============================================================================
    parser.add_argument('--lr_step_size', type=int,
                        default=7, help='learning rate step size')

    parser.add_argument('--lr_gamma', type=float,
                        default=0.1, help='learning rate gamma factor')

    parser.add_argument('--lr_scheduler', type=str, default=None,
                        choices=['multistep'], help='learning rate scheduler')

    parser.add_argument('--lr_multistep_milestones', type=str,
                        default='10,20,30',
                        help='learning rate scheduler')

    parser.add_argument('--debug', action='store_true',
                        help='use global contextfeatures')

    parser.add_argument('--pretrained_model_dir',
                        type=str, default='models',
                        help='directory to download pretrained models')

    parser.add_argument('--log_dir', type=str,
                        default='lightning_logs',
                        help='pytorch lightning log directory')

    parser.add_argument('--seed',
                        type=int,
                        default=2021,
                        help='random seed')

    # ===============================================================================
    # Testing
    # ===============================================================================
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='resuming checkpoint')

    if add_pl_args:
        parser = pl.Trainer.add_argparse_args(parser)
    else:
        parser.add_argument('--gpus', type=int,
                            help='Number of GPUS')

    # Generate args
    if is_notebook:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    from argparse import Namespace
    config_args = mipkit.load_yaml_config(args.config_file, to_args=True)

    return Namespace(**vars(args), **vars(config_args))


def cli_main():

    # ===============================================================================
    # Arguments
    # ===============================================================================
    args = parse_args()

    print(args)
    mipkit.seed_everything(args.seed)

    # ===============================================================================
    # Dataset
    # ===============================================================================

    train_dataset = AffectNetDataset(data_dir=args.dataloader.data_dir,
                                     split='train',
                                     csv_path=args.dataloader.train_csv_path,
                                     transform=get_train_transform(args.model.image_size))

    val_dataset = AffectNetDataset(data_dir=args.dataloader.data_dir,
                                   csv_path=args.dataloader.val_csv_path,
                                   transform=get_val_transform(
                                       args.model.image_size),
                                   split='val')

    dataloader = AffectNetDataloader(train_dataset=train_dataset,
                                     val_dataset=val_dataset,
                                     batch_size=args.dataloader.batch_size,
                                     num_workers=args.num_workers,
                                     pin_memory=True)

    model = get_model(model_name=args.model.model_name, num_classes=7)

    if args.mode == 'train':
        # ===============================================================================
        # Callbacks
        # ===============================================================================

        ckpt_cb = ModelCheckpoint(
            save_top_k=3,
            verbose=True,
            monitor='vloss',
            mode='min',
            save_last=True,
            filename='checkpoint_{epoch:03d}-{step}'
        )

        tb_logger = TensorBoardLogger(save_dir=args.log_dir)

        # lr_logger = LearningRateMonitor(logging_interval='epoch')

        # ===============================================================================
        # Training
        # ===============================================================================
        trainer = pl.Trainer.from_argparse_args(args=args,
                                                logger=True,
                                                # , lr_logger],
                                                callbacks=[ckpt_cb],
                                                profiler=True,
                                                resume_from_checkpoint=args.resume_ckpt if args.resume_ckpt else None)

        # ===============================================================================
        # Modeling
        # ===============================================================================
        pl_model = BasedModel(model=model, args=args)

        trainer.fit(model=pl_model, datamodule=dataloader)

    # ===============================================================================
    # Testing
    # ===============================================================================
    elif args.mode == 'test':
        trainer = pl.Trainer(logger=False,
                             checkpoint_callback=False).from_argparse_args(args=args)

        pl_model = BasedModel.load_from_checkpoint(args=args,
                                                   model=model,
                                                   checkpoint_path=args.resume_ckpt)
        trainer.test(model=pl_model, datamodule=dataloader)


if __name__ == "__main__":
    cli_main()
