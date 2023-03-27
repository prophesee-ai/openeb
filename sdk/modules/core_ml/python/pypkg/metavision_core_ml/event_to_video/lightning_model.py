# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
Pytorch Lightning module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import os
import argparse
import numpy as np
import cv2

from types import SimpleNamespace
from torchvision.utils import make_grid
from tqdm import tqdm
from itertools import islice

from metavision_core_ml.core.temporal_modules import time_to_batch, seq_wise
from metavision_core_ml.event_to_video.event_to_video import EventToVideo
from metavision_core_ml.utils.torch_ops import normalize_tiles
from metavision_core_ml.utils.show_or_write import ShowWrite
from metavision_core_ml.utils.torch_ops import cuda_tick, viz_flow

from metavision_core_ml.losses.perceptual_loss import VGGPerceptualLoss
from metavision_core_ml.losses.warp import ssl_flow_l1
from kornia.losses import ssim_loss


class EventToVideoCallback(pl.callbacks.Callback):
    """
    callbacks to our model
    """

    def __init__(self, data_module, video_result_every_n_epochs=2, show_window=False):
        super().__init__()
        self.data_module = data_module
        self.video_every = int(video_result_every_n_epochs)
        self.show_window = show_window

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch and not (trainer.current_epoch % self.video_every):
            pl_module.demo_video(self.data_module.val_dataloader(), trainer.current_epoch, show_video=self.show_window)


class EventToVideoLightningModel(pl.LightningModule):
    """
    EventToVideo: Train your EventToVideo
    """

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        self.save_hyperparameters(hparams, logger=False)
        self.model = EventToVideo(
            self.hparams.cin,
            self.hparams.cout,
            self.hparams.num_layers,
            self.hparams.base,
            self.hparams.cell,
            self.hparams.separable,
            self.hparams.separable_hidden,
            self.hparams.archi)
        self.vgg_perc_l1 = VGGPerceptualLoss()

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        hparams = argparse.Namespace(**checkpoint['hyper_parameters'])
        model = cls(hparams)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def forward(self, x):
        state = self.model(x)
        return self.model.predict_gray(state)

    def compute_loss(self, x, y, reset_mask):
        self.model.reset(reset_mask)

        target = (y / 255.0)
        state = self.model(x)
        pred = self.model.predict_gray(state)

        bw_flow = self.model.predict_flow(state)

        pred_flat = time_to_batch(pred)[0].float()
        target_flat = time_to_batch(target)[0].float()
        loss_dict = {}
        loss_dict['ssim'] = ssim_loss(pred_flat, target_flat, 5)
        loss_dict['smooth_l1'] = F.smooth_l1_loss(pred_flat, target_flat, beta=0.11)

        loss_dict['vgg_perc_l1'] = self.vgg_perc_l1(pred_flat, target_flat) * 0.5

        # loss_dict['target_flow_l1'] = ssl_flow_l1(target, bw_flow)
        loss_dict['pred_flow_l1'] = ssl_flow_l1(pred, bw_flow)
        for k, v in loss_dict.items():
            assert v >= 0, k
        return loss_dict

    def training_step(self, batch, batch_nb):
        batch = SimpleNamespace(**batch)
        loss_dict = self.compute_loss(batch.inputs, batch.images, batch.reset)

        loss = sum([v for k, v in loss_dict.items()])

        assert loss.item() >= 0
        logs = {'loss': loss}
        logs.update({'train_' + k: v.item() for k, v in loss_dict.items()})

        self.log('train_loss', loss)
        for k, v in loss_dict.items():
            self.log('train_' + k, v)

        return loss

    def validation_step(self, batch, batch_nb):
        batch = SimpleNamespace(**batch)
        loss_dict = self.compute_loss(batch.inputs, batch.images, batch.reset)
        loss = sum([v for k, v in loss_dict.items()])
        assert loss.item() >= 0

        logs = {'val_loss': loss}
        logs.update({'val_' + k: v.item() for k, v in loss_dict.items()})

        self.log('val_loss', loss)
        for k, v in loss_dict.items():
            self.log('val_' + k, v)
        return loss

    def validation_epoch_end(self, outputs):
        val_loss_avg = torch.FloatTensor([item for item in outputs]).mean()
        self.log('val_acc', val_loss_avg)
        return val_loss_avg

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def demo_video(self, dataloader, epoch=0, show_video=True):
        print('Demo')
        height, width = self.hparams.height, self.hparams.width
        batch_size = self.hparams.batch_size
        nrows = 2 ** ((batch_size.bit_length() - 1) // 2)
        ncols = int(np.ceil(self.hparams.batch_size / nrows))

        self.model.eval()

        video_name = os.path.join(self.hparams.root_dir, 'videos', f'video#{epoch:d}.mp4')
        dir = os.path.dirname(video_name)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        window_name = None
        if show_video:
            window_name = "test_epoch {:d}".format(epoch)
        show_write = ShowWrite(window_name, video_name)

        with torch.no_grad():
            for batch in tqdm(islice(dataloader, self.hparams.demo_iter), total=self.hparams.demo_iter):

                batch = SimpleNamespace(**batch)
                batch.inputs = batch.inputs.to(self.device)
                batch.reset = batch.reset.to(self.device)

                x = batch.inputs
                x = x[:, :, :3]
                x = 255 * normalize_tiles(x)
                y = batch.images

                self.model.reset(batch.reset)

                s = self.model(batch.inputs)
                o = self.model.predict_gray(s)
                o = normalize_tiles(o, num_stds=6)
                o = 255 * o

                if self.hparams.plot_flow:
                    f = self.model.predict_flow(s)
                    f = seq_wise(viz_flow)(f)
                for t in range(len(x)):
                    gx = make_grid(x[t], nrow=nrows, padding=0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    gy = make_grid(y[t], nrow=nrows, padding=0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    go = make_grid(o[t], nrow=nrows, padding=0).permute(1, 2, 0).data.cpu().numpy().astype(np.uint8)

                    if self.hparams.plot_flow:
                        gf = make_grid(
                            f[t],
                            nrow=nrows, padding=0).permute(
                            1, 2, 0).data.cpu().numpy().astype(
                            np.uint8)
                        cat = np.concatenate([gx, gy, gf, go], axis=1)
                    else:
                        cat = np.concatenate([gx, gy, go], axis=1)
                    show_write(cat)
