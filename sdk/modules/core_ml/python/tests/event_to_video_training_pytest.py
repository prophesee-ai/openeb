# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import sys
import os
import argparse
import pytest
import torch
import glob

test_dir_path = os.path.dirname(os.path.realpath(__file__))
sample_dir_path = os.path.join(test_dir_path, '..', 'samples', 'train_event_to_video')
assert os.path.isdir(sample_dir_path)
sys.path.append(sample_dir_path)

from train_event_to_video import train_parser, train


def get_argparse_defaults(parser):
    defaults = {}
    for action in parser._actions:
        if not action.required and action.dest != "help":
            defaults[action.dest] = action.default
    return defaults


def pytestcase_training_mini_dataset(tmpdir, dataset_dir):
    """
    This is a functional test
    Tests if we can create a LightningDetectionModel and train for a few batches.
    """
    # Parse Params
    parser = train_parser()
    params = get_argparse_defaults(parser)
    params['root_dir'] = tmpdir
    params['dataset_path'] = os.path.join(dataset_dir, 'openeb', 'core_ml', 'mini_image_dataset')
    params['lr'] = 0.0001
    params['event_volume_depth'] = 10
    params['batch_size'] = 2
    params['cout'] = 1
    params['precision'] = 32
    params['num_layers'] = 2
    params['cell'] = 'lstm'
    params['archi'] = 'all_rnn'
    params['epochs'] = 2
    params['base'] = 2
    params['demo_every'] = 100
    params['limit_train_batches'] = 3
    params['limit_val_batches'] = 3
    params['cpu'] = True
    params['data_device'] = 'cpu'
    params['height'] = 60
    params['width'] = 80
    params['num_tbins'] = 7
    params['num_workers'] = 0
    params['no_window'] = True
    params = argparse.Namespace(**params)

    # Actual Training
    train(params)

    # Verify checkpoint has been written
    ckpt_filenames = glob.glob(os.path.join(tmpdir, "checkpoints", "epoch=1*.ckpt"))
    assert len(ckpt_filenames) == 1
    dic = torch.load(ckpt_filenames[0])
