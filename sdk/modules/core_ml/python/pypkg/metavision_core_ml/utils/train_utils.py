# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
Tools common to training main functions.
"""
import os
from glob import glob, iglob


def extract_num(path):
    filename = os.path.splitext(path)[0]
    num = filename.split('=')[1]
    return int(num) if num.isdigit() else -1


def search_latest_checkpoint(root_dir, mode='time'):
    """looks for latest checkpoint in latest sub-directory"""
    vdir = os.path.join(root_dir, '**', 'checkpoints')
    ckpts = glob(os.path.join(vdir, '*.ckpt'), recursive=True)
    if mode == 'time':
        ckpts.sort(key=lambda x: os.path.getmtime(x))
    else:
        ckpts = sorted(glob(os.path.join(vdir, '*.ckpt')), key=extract_num)
    return ckpts[-1]
