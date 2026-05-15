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
import pytorch_lightning as pl

import os
import argparse
import numpy as np

from tqdm import tqdm
from itertools import islice

from metavision_core_ml.utils.show_or_write import ShowWrite
from metavision_core_ml.corner_detection.firenet import FireNet


class CornerDetectionCallback(pl.callbacks.Callback):
    """
    callbacks to our model
    """

    def __init__(self, data_module, video_result_every_n_epochs=2):
        super().__init__()
        self.data_module = data_module
        self.video_every = int(video_result_every_n_epochs)

    def on_train_epoch_end(self, trainer, pl_module):
        torch.save(pl_module.model,
                   os.path.join(trainer.log_dir, "whole-model-epoch-{}.ckpt".format(trainer.current_epoch)))
        if trainer.current_epoch and not (trainer.current_epoch % self.video_every):
            pl_module.video(self.data_module.train_dataloader(), trainer.current_epoch, set="train")
            pl_module.video(self.data_module.val_dataloader(), trainer.current_epoch, set="val")


class CornerDetectionLightningModel(pl.LightningModule):
    """
    Corner Detection: Train your FireNet model to predict corners as a heatmap
    """

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        self.save_hyperparameters(hparams, logger=False)
        self.model = FireNet(cin=self.hparams.cin, cout=self.hparams.cout, base=12)
        self.classification_loss = nn.functional.binary_cross_entropy_with_logits
        self.l1_loss = nn.L1Loss()
        self.use_the_whole_mask = True
        self.overlap_heat_map = True
        self.ratio_negative = 3
        self.sigmoid_fn = torch.nn.Sigmoid()

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        hparams = argparse.Namespace(**checkpoint['hyper_parameters'])
        model = cls(hparams)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def compute_loss(self, events, corner_mask, reset_mask):
        loss_dict = {}
        self.model.reset(reset_mask)

        pred = self.model(events.float())

        mask_positive = corner_mask > 0
        mask_negative = corner_mask == 0
        threshold = torch.topk(pred[mask_negative], self.ratio_negative*mask_positive.sum())[0][-1]
        hard_negative_mining_mask = pred >= threshold
        positive_and_negative_mask = (mask_positive + hard_negative_mining_mask) > 0
        loss_dict["whole_image_cross_entropy"] = self.classification_loss(pred.float(),
                                                                          corner_mask/255.,
                                                                          weight=positive_and_negative_mask)

        return loss_dict

    def training_step(self, batch, batch_nb):
        loss_dict = self.compute_loss(batch["inputs"], batch["corners"], batch["reset"])
        loss = sum([v for k, v in loss_dict.items()])
        logs = {'loss': loss}
        logs.update({'train_' + k: v.item() for k, v in loss_dict.items()})

        self.log('train_loss', loss)
        for k, v in loss_dict.items():
            self.log('train_' + k, v)

        return logs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def make_heat_map_image(self, pred, divide_max=True):
        image = np.zeros((pred.shape[3], pred.shape[4]))
        for t in range(pred.shape[2]):
            pred_t = pred[0, 0, t]
            image = image + pred_t.cpu().numpy()
        if (image.max() != 0) and divide_max:
            image /= image.max()
        image *= 255
        image = np.concatenate([np.expand_dims(image, 2)] * 3, axis=2)
        return image.astype(np.uint8)

    def make_color_heat_map_image(self, pred):
        image = np.zeros((pred.shape[3], pred.shape[4], 3))
        pred = pred.cpu().numpy()
        for t in range(pred.shape[2]):
            pred_t = 1*(pred[0, 0, t] > 0.1)
            image[pred_t != 0] = np.array([0, (pred.shape[2]-1-t)*(int(255/pred.shape[2])), 255])
        return image.astype(np.uint8)

    def image_from_events(self, events):
        events = events.sum(2).unsqueeze(2)
        events_as_image = 255 * (events > 0) + 0 * (events < 0) + 128 * (events == 0)
        return events_as_image

    def video(self, dataloader, epoch=0, set="val"):
        """

        Args:
            dataloader: data loader from train or val set
            epoch: epoch
            set: can be either train or val

        Returns:

        """
        print('Video on {} set'.format(set))

        self.model.eval()

        video_name = os.path.join(self.hparams.root_dir, 'videos', 'video_{}_{}.mp4'.format(set, epoch))
        dir = os.path.dirname(video_name)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        show_write = ShowWrite(False, video_name)

        with torch.no_grad():
            for batch in tqdm(islice(dataloader, self.hparams.demo_iter), total=self.hparams.demo_iter):
                events = batch["inputs"].to(self.device)
                self.model.reset(batch["reset"])

                pred = self.model(events.float())
                # Draw GT corners on images
                ground_truth = batch["corners"]
                image = self.image_from_events(events)

                for t in range(pred.shape[0]):
                    heat_map_gt = self.make_color_heat_map_image(ground_truth[t, 0].unsqueeze(0).unsqueeze(1)/255.)
                    heat_map_image = self.make_color_heat_map_image(
                        self.sigmoid_fn(pred[t, 0].unsqueeze(0).unsqueeze(1)))
                    events_image = image[t, 0, 0].cpu().numpy().astype(np.uint8)
                    events_image = np.concatenate([np.expand_dims(events_image, 2)] * 3, axis=2)
                    if self.overlap_heat_map:
                        heat_map_gt_mask = heat_map_gt.sum(2) == 0
                        heat_map_gt[heat_map_gt_mask] = events_image[heat_map_gt_mask]
                        heat_map_gt[~heat_map_gt_mask] = heat_map_gt[~heat_map_gt_mask]
                        heat_map_image_mask = heat_map_image.sum(2) == 0
                        heat_map_image[heat_map_image_mask] = events_image[heat_map_image_mask]
                        heat_map_image[~heat_map_image_mask] = heat_map_image[~heat_map_image_mask]
                    cat = np.concatenate([events_image, heat_map_gt, heat_map_image], axis=1)
                    # event_image is an image created from events
                    # heatmap gt is the ground truth heatmap of corners overlaid with the events
                    # heatmap image is the predicted corners overlaid with the events
                    show_write(cat)
