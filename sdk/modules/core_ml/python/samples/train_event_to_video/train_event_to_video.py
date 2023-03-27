# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
E2V Training Script
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from metavision_core_ml.event_to_video.lightning_model import EventToVideoLightningModel
from metavision_core_ml.event_to_video.lightning_model import EventToVideoCallback
from metavision_core_ml.event_to_video.data_module import EventToVideoDataModule
from metavision_core_ml.utils.train_utils import search_latest_checkpoint


import os
import numpy as np
import argparse

torch.manual_seed(0)
np.random.seed(0)


def train_parser():
    parser = argparse.ArgumentParser()
    # dir params
    parser.add_argument('root_dir', type=str, default='', help='logging directory')
    parser.add_argument('dataset_path', type=str, default='', help='path')

    # train params
    parser.add_argument('--lr', type=float, default=0.0007, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--demo_iter', type=int, default=10, help='run demo for X batches')
    parser.add_argument('--precision', type=int, default=16, help='precision 32 or 16')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='accumulate gradient for more than a single batch')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--demo_every', type=int, default=1, help='run demo every X epoch')
    parser.add_argument('--val_every', type=int, default=1, help='validate every X epochs')
    parser.add_argument('--save_every', type=int, default=1, help='save every X epochs')
    parser.add_argument('--just_test', action='store_true', help='launches demo video')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default='', help='resume from specific checkpoint')
    parser.add_argument('--mask_loss_no_events_yet', action='store_true', help='mask loss where no events')
    parser.add_argument('--limit_train_batches', type=int, default=10000, help='run training epoch for X batches')
    parser.add_argument('--limit_val_batches', type=int, default=100, help='run training epoch for X batches')
    parser.add_argument('--data_device', type=str, default='cuda', help='run simulation on the cpu/gpu')
    parser.add_argument('--event_volume_depth', type=int, default=10, help='event volume depth')

    # network params
    parser.add_argument('--cin', type=int, default=10, help='num inputs, check in your data')
    parser.add_argument('--cout', type=int, default=1, help='num channels prediction, check your gt')
    parser.add_argument('--base', type=int, default=8, help='width of network')
    parser.add_argument('--num_layers', type=int, default=3, help='depth of network')
    parser.add_argument('--cell', type=str, choices=['lstm', 'brc'], default='lstm', help='rnn cell')
    parser.add_argument('--separable', action='store_true', help='use separable convolutions')
    parser.add_argument('--separable_hidden', action='store_true', help='use separable for rnn convolutions')
    parser.add_argument('--archi', type=str, default='all_rnn', help='archi type of unet')
    parser.add_argument('--plot_flow', action='store_true', help='plot the flow')

    # data params
    parser.add_argument('--height', type=int, default=180, help='image height')
    parser.add_argument('--width', type=int, default=240, help='image width')
    parser.add_argument('--num_tbins', type=int, default=10, help="timesteps per batch tbppt")
    parser.add_argument('--min_frames_per_video', type=int, default=100, help='max frames per video')
    parser.add_argument('--max_frames_per_video', type=int, default=10000, help='max frames per video')
    parser.add_argument(
        '--batch_sampling_mode', type=str, choices=['frames', 'delta_t', 'random'],
        default='random', help='how to sample video batches')
    parser.add_argument('--batch_min_frames', type=int, default=1, help='how to sample videos')
    parser.add_argument('--batch_max_frames', type=int, default=10, help='how to sample videos')
    parser.add_argument('--batch_min_delta_t', type=int, default=1000, help='how to sample videos')
    parser.add_argument('--batch_max_delta_t', type=int, default=30000, help='how to sample videos')
    parser.add_argument('--num_workers', type=int, default=2, help='number of threads')

    # display params
    parser.add_argument("--no_window", action="store_true",
                        help="Disable output window during demo (only write a video)")
    return parser


def train(params: argparse.Namespace):
    print(params)

    model = EventToVideoLightningModel(params)
    if not params.cpu:
        model.cuda()

    if params.resume:
        ckpt = search_latest_checkpoint(params.root_dir)
    elif params.checkpoint != "":
        ckpt = params.checkpoint
    else:
        ckpt = None
    print('ckpt: ', ckpt)

    tmpdir = os.path.join(params.root_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, every_n_epochs=params.save_every)

    logger = TensorBoardLogger(save_dir=os.path.join(params.root_dir, 'logs'))

    if ckpt is not None and params.just_test:
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu') if params.cpu else torch.device("cuda"))
        model.load_state_dict(checkpoint['state_dict'])

    # Data Setup
    data = EventToVideoDataModule(params)

    if params.just_test:
        if not params.cpu:
            model = model.cuda()
        model.demo_video(data.val_dataloader(), -1)
    else:
        demo_callback = EventToVideoCallback(data, params.demo_every, show_window=not params.no_window)
        trainer = pl.Trainer(
            default_root_dir=params.root_dir,
            callbacks=[checkpoint_callback, demo_callback],
            logger=logger,
            gpus=0 if params.cpu else 1,
            precision=params.precision,
            accumulate_grad_batches=params.accumulate_grad_batches,
            max_epochs=params.epochs,
            resume_from_checkpoint=ckpt,
            log_every_n_steps=5,
            limit_train_batches=params.limit_train_batches,
            limit_val_batches=params.limit_val_batches,
            accelerator="cpu" if params.cpu else None
        )
        trainer.fit(model, data)


def main():
    """Using Pytorch Lightning to train our model

    you can visualize logs with tensorboard:

    %tensorboard --logdir my_root_dir/lightning_logs/
    """
    print('pl version: ', pl.__version__)
    parser = train_parser()
    params, _ = parser.parse_known_args()
    train(params)


if __name__ == '__main__':
    main()
