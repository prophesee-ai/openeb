# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Unit tests for the Video Dataset With Events iterator
"""
import os
import numpy as np
import pytest

from metavision_core_ml.video_to_event.video_stream_dataset_with_events_cpu import VideoDatasetWithEventsIterator
from metavision_core_ml.data.scheduling import Metadata


def pytestcase_load_video_with_events(tmpdir, dataset_dir):
    timestamps_filename = os.path.join(dataset_dir, "openeb", "core_ml", "ultimate_frisbee", "frames_ts.npy")
    video_filename = os.path.join(dataset_dir, "openeb", "core_ml", "ultimate_frisbee", "frames.mp4")
    assert os.path.isfile(timestamps_filename)
    assert os.path.isfile(video_filename)

    ts = np.load(timestamps_filename)
    print(ts.size)
    metadata = Metadata(video_filename, 0, ts.size)

    dic_params_video_dataset = {"height": 240, "width": 320,
                                "min_tbins": 5, "max_tbins": 6,
                                "rgb": True, "pause_probability": 0.}
    dic_params_event_simulator = {"min_Cp": 0.2, "max_Cp": 0.2, "min_Cn": 0.2, "max_Cn": 0.2,
                                  "min_sigma_threshold": 0., "max_sigma_threshold": 0.0}

    video_dataset = VideoDatasetWithEventsIterator(metadata=metadata,
                                                   dic_params_video_dataset=dic_params_video_dataset,
                                                   dic_params_events_simulator=dic_params_event_simulator,
                                                   discard_events_between_batches=True)
    video_dataset_it = iter(video_dataset)
    batch = next(video_dataset_it)
    _, _, _, _, _, events_cd, simu_params = batch
    print(simu_params)
    assert events_cd.size > 0
