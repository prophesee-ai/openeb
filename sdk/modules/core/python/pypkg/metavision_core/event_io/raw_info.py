# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Get Raw Duration: Either search for a json filename called "path_name_info.json" or compute duration itself.
"""
import json
from os.path import exists, splitext
from metavision_core.event_io.raw_reader import RawReader


def read_raw_info(path):
    """
    Collects information of duration by running RawReader.

    Args:
        path (str): raw path
    """
    cls = RawReader(path)
    count = 0
    n_events = 10000
    while not cls.is_done():
        evs = cls.load_n_events(n_events)
        count += len(evs)
        duration = evs['t'][-1]
    info = {'duration': int(duration), 'count': int(count)}
    return info


def get_raw_info(path):
    """
    Reads path raw info json file.
    If it does not exists, it will create it.

    Args:
        path (str): raw path
    """
    info_json_path = splitext(path)[0] + '_info.json'
    if exists(info_json_path):
        with open(info_json_path, 'r') as f:
            info = json.load(f)
    else:
        info = read_raw_info(path)
        with open(info_json_path, 'w') as f:
            json.dump(info, f)
    return info
