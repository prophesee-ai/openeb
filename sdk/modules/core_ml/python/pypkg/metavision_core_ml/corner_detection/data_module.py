# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
Here we reuse the GPUSimulator from OpenEB to stream synthetic events.
"""
import os
import pytorch_lightning as pl
from metavision_core_ml.corner_detection.gpu_corner_esim import GPUEBSimCorners


class EventToCornerDataModule(pl.LightningDataModule):
    """
    Simulation gives events + frames + corners
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.split_names = ['train', 'val', 'test']

    def get_dataloader(self, dataset_path):
        dataloader = GPUEBSimCorners.from_params(
            dataset_path,
            self.hparams.num_workers,
            self.hparams.batch_size,
            self.hparams.num_tbins,
            self.hparams.event_volume_depth,
            self.hparams.height,
            self.hparams.width,
            self.hparams.min_frames_per_video,
            self.hparams.max_frames_per_video,
            self.hparams.number_of_heatmaps,
            self.hparams.randomize_noises,
            self.hparams.data_device)
        return dataloader

    def train_dataloader(self):
        path = os.path.join(self.hparams.dataset_path, self.split_names[0])
        return self.get_dataloader(path)

    def val_dataloader(self):
        path = os.path.join(self.hparams.dataset_path, self.split_names[1])
        return self.get_dataloader(path)

    def test_dataloader(self):
        path = os.path.join(self.hparams.dataset_path, self.split_names[1])
        return self.get_dataloader(path)
