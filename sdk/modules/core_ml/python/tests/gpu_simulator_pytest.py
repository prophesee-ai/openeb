# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
Unit tests for the GPU Simulator
"""
import os
import pytest
import numpy as np
import torch
import torch.nn.functional as F
from metavision_core_ml.video_to_event.gpu_simulator import GPUEventSimulator
from metavision_core_ml.video_to_event.simulator import EventSimulator, eps_log
from metavision_core_ml.video_to_event.video_stream_dataset import make_video_dataset
from metavision_core_ml.preprocessing.event_to_tensor_torch import event_volume, event_image
from itertools import islice

CUDA_NOT_AVAILABLE = not torch.cuda.is_available()


def pytestcase_num_tbins(dataset_dir):
    """
    Runs the same simu in different fashion w.r.t num_tbins
    """
    # GIVEN
    path = os.path.join(dataset_dir, "openeb", "core_ml", "mini_image_dataset")
    batch_size = 4
    height, width = 120, 180
    threshold_mu = 0.1
    refractory_period = 0
    leak_hz = 0.0
    cutoff_hz = 0

    dataloader1 = make_video_dataset(path, 0, batch_size, height, width, 30, 300, seed=1)
    batches = [batch for batch in islice(dataloader1, 10)]
    simu = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_hz, cutoff_hz)
    lstate = simu.log_states.clone()
    counts_list = []
    early_stop = False
    for i, batch_dict in enumerate(batches):
        if i >= 1 and batch_dict['first_times'].sum() > 0:
            early_stop = True
            break
        log_batch = simu.log_images(batch_dict['images'].squeeze())
        timestamps = batch_dict['timestamps']
        first_times = batch_dict['first_times']

        counts = simu.count_events(log_batch, batch_dict['video_len'], timestamps, first_times, reset=True)
        counts_list.append(counts.clone())

    counts_list = torch.cat([item[None] for item in counts_list])
    counts1 = counts_list.sum(dim=0)

    simu2 = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_hz, cutoff_hz)
    lstate2 = simu2.log_states.clone()
    if early_stop:
        batches = batches[:i]
    batch_list = [[] for _ in range(batch_size)]
    for batch in batches:
        ind = 0
        for i, end in enumerate(batch["video_len"].cumsum(0)):
            batch_list[i].append(batch['images'].squeeze()[..., ind:end])
            ind = end

    big_batch = torch.cat([torch.cat([b for b in batch], dim=-1) for batch in batch_list], dim=-1)
    video_len = torch.stack([batch['video_len'] for batch in batches]).sum(0)
    max_len = video_len.max()

    timestamps = torch.zeros((batch_size, max_len), dtype=torch.float32)
    for b in range(batch_size):
        timestamps_list = [batch["timestamps"][b, :batch["video_len"][b]] for batch in batches]
        timestamps[b, :video_len[b]] = torch.cat(timestamps_list, dim=-1)
        timestamps[b, video_len[b]:] = timestamps[b, video_len[b] - 1]

    first_times = batches[0]['first_times']
    log_batch_big = simu2.log_images(big_batch.contiguous())
    counts2 = simu2.count_events(log_batch_big, video_len.contiguous(), timestamps, first_times)

    # THEN
    diff_state = simu.log_states - simu2.log_states
    assert diff_state.abs().max().item() == 0
    diff = counts1 - counts2
    assert diff.abs().max().item() == 0


def pytestcase_new_old(dataset_dir):
    """
    Testing that Old Simulator is equivalent to the New GPU one
    """
    path = os.path.join(dataset_dir, "openeb", "core_ml", "mini_image_dataset")
    batch_size = 1
    height, width = 256, 256
    threshold_mu = 0.1
    threshold_std = 0.001
    refractory_period = 100
    nbins = 1
    dataloader = make_video_dataset(path, 0, batch_size, height, width, 30, 300, seed=1)
    batches = [batch for batch in islice(dataloader, 40)]

    event_cpu = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period)
    old_event_cpu = EventSimulator(height, width, threshold_mu, threshold_mu, refractory_period, 0)
    old_event_cpu.Cps = event_cpu.thresholds[1, 0].numpy()
    old_event_cpu.Cns = event_cpu.thresholds[0, 0].numpy()

    for i, batch_dict in enumerate(batches):
        batch = batch_dict['images'].squeeze(0)
        timestamps = batch_dict['timestamps']
        first_times = batch_dict['first_times']
        log_batch = event_cpu.log_images(batch)

        start_times = timestamps[:, 0] * first_times + (1 - first_times) * event_cpu.prev_image_ts
        durations = timestamps[:, -1] - start_times

        # old: reset
        if first_times[0] == 1:
            old_event_cpu.reset()
            old_event_cpu.Cps = event_cpu.thresholds[1, 0].numpy()
            old_event_cpu.Cns = event_cpu.thresholds[0, 0].numpy()

        # old: actual simulation
        for t in range(log_batch.shape[-1]):
            img = log_batch[:, :, t].numpy().copy()
            ts = timestamps[0, t].item()
            old_event_cpu.log_image_callback(img, ts)

        old_events = old_event_cpu.get_events().copy()
        old_event_cpu.flush_events()

        events = event_cpu.get_events(log_batch, batch_dict['video_len'], timestamps, first_times)
        t1 = events[:, -1]
        idx = torch.argsort(t1)
        events = events[idx]

        if not len(events):
            continue

        # Event-Volume Difference?
        ev_vol1 = event_volume(events, batch_size, height, width, start_times, durations, nbins, 'nearest')

        events2 = torch.zeros((len(old_events), 5), dtype=torch.int32)
        events2[:, 1] = torch.from_numpy(old_events['x'] * 1.0)
        events2[:, 2] = torch.from_numpy(old_events['y'] * 1.0)
        events2[:, 3] = torch.from_numpy(old_events['p'] * 2.0 - 1)
        events2[:, 4] = torch.from_numpy(old_events['t'] * 1.0)

        ev_vol2 = event_volume(events2, batch_size, height, width, start_times, durations, nbins, 'nearest')
        diff = (ev_vol1 - ev_vol2)
        assert diff.abs().max().item() == 0


def pytestcase_new_old_cutoff(dataset_dir):
    """
    Testing that Old Simulator is equivalent to the New GPU one
    """
    path = os.path.join(dataset_dir, "openeb", "core_ml", "mini_image_dataset")
    batch_size = 1
    height, width = 256, 256
    threshold_mu = 0.1
    threshold_std = 0.001
    refractory_period = 100
    nbins = 1
    cutoff_hz = 30
    dataloader = make_video_dataset(path, 0, batch_size, height, width, 30, 300, seed=1)
    batches = [batch for batch in islice(dataloader, 40)]

    event_cpu = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, cutoff_hz=cutoff_hz)
    old_event_cpu = EventSimulator(height, width, threshold_mu, threshold_mu,
                                   refractory_period, 0, cutoff_hz=cutoff_hz)

    for i, batch_dict in enumerate(batches):
        batch = batch_dict['images'].squeeze(dim=0)
        timestamps = batch_dict['timestamps']
        first_times = batch_dict['first_times']
        log_batch = event_cpu.dynamic_moving_average(batch, batch_dict['video_len'], timestamps, first_times)

        # old: reset
        if first_times[0] == 1:
            old_event_cpu.reset()
            old_event_cpu.Cps = event_cpu.thresholds[1, 0].numpy()
            old_event_cpu.Cns = event_cpu.thresholds[0, 0].numpy()

        # old photodiode latency simulation
        for t in range(batch.shape[-1]):
            img = batch[..., t].numpy()
            ts = timestamps[0, t].item()
            log_t = old_event_cpu.dynamic_moving_average(img, ts)
            old_event_cpu.last_img_ts = ts
            log_t_2 = log_batch[..., t].cpu().numpy()

            assert np.allclose(log_t, log_t_2, atol=1e-6)


@pytest.mark.skipif(CUDA_NOT_AVAILABLE, reason="this machine does not have gpu available")
def pytestcase_cpu_gpu_cutoff(dataset_dir):
    """
    Testing that Old Simulator is equivalent to the New GPU one
    """
    path = os.path.join(dataset_dir, "openeb", "core_ml", "mini_image_dataset")
    batch_size = 1
    height, width = 256, 256
    threshold_mu = 0.1
    threshold_std = 0.001
    refractory_period = 100
    nbins = 1
    cutoff_hz = 30
    dataloader = make_video_dataset(path, 0, batch_size, height, width, 30, 300, seed=1)
    batches = [batch for batch in islice(dataloader, 40)]

    event_cpu = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, cutoff_hz=cutoff_hz)

    event_gpu = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, cutoff_hz=cutoff_hz)
    event_gpu.to('cuda:0')

    for i, batch_dict in enumerate(batches):
        batch = batch_dict['images'].squeeze(dim=0)
        timestamps = batch_dict['timestamps']
        first_times = batch_dict['first_times']
        num_frames = batch_dict['video_len']

        log_batch = event_cpu.dynamic_moving_average(batch, num_frames, timestamps, first_times)
        log_batch2 = event_gpu.dynamic_moving_average(
            batch.to('cuda:0'),
            num_frames.to('cuda:0'),
            timestamps.to('cuda:0'),
            first_times.to('cuda:0')).to('cpu:0')

        assert torch.allclose(log_batch, log_batch2, atol=1e-6)


@pytest.mark.skipif(CUDA_NOT_AVAILABLE, reason="this machine does not have gpu available")
def pytestcase_cpu_gpu_count(dataset_dir):
    """
    Runs the CPU & GPU versions.
    Make sure they are equivalent for the counting of events.
    """
    # GIVEN
    path = os.path.join(dataset_dir, "openeb", "core_ml", "mini_image_dataset")
    batch_size = 4
    height, width = 60, 80
    threshold_mu = 0.1
    refractory_period = 10
    leak_hz = 0.1
    cutoff_hz = 60
    dataloader = make_video_dataset(path, 0, batch_size, height, width, 30, 300, seed=1)

    batches = [batch for batch in islice(dataloader, 5)]

    event_cpu = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_hz, cutoff_hz)
    event_gpu = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_hz, cutoff_hz)
    event_gpu.to('cuda:0')

    # THEN
    for batch_dict in batches:
        batch = event_gpu.log_images(batch_dict['images'].squeeze())
        timestamps = batch_dict['timestamps']
        first_times = batch_dict['first_times']
        video_len = batch_dict['video_len']
        batch_gpu = batch.to('cuda:0')
        timestamps_gpu = timestamps.to('cuda:0')
        first_times_gpu = first_times.to('cuda:0')
        counts = event_gpu.count_events(batch_gpu, video_len.cuda(), timestamps_gpu, first_times_gpu).cpu()
        counts2 = event_cpu.count_events(batch, video_len, timestamps, first_times)
        diff = counts - counts2
        assert diff.abs().max().item() == 0


@pytest.mark.skipif(CUDA_NOT_AVAILABLE, reason="this machine does not have gpu available")
def pytestcase_cpu_gpu_event_volume(dataset_dir):
    """
    Runs the CPU & GPU versions.
    Make sure they are equivalent for the voxel grid of events.
    """
    # GIVEN
    path = os.path.join(dataset_dir, "openeb", "core_ml", "mini_image_dataset")
    batch_size = 4
    nbins = 5
    height, width = 60, 80
    threshold_mu = 0.1
    refractory_period = 10
    leak_rate = 0.1
    cutoff_hz = 90
    dataloader = make_video_dataset(path, 0, batch_size, height, width, 30, 300, seed=1)

    batches = [batch for batch in islice(dataloader, 5)]

    event_cpu = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_rate, cutoff_hz)
    event_gpu = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_rate, cutoff_hz)
    event_gpu.to('cuda:0')

    # THEN
    for batch_dict in batches:
        batch = event_gpu.log_images(batch_dict['images'].squeeze())
        timestamps = batch_dict['timestamps']
        first_times = batch_dict['first_times']
        video_len = batch_dict['video_len']
        batch_gpu = batch.to('cuda:0')
        timestamps_gpu = timestamps.to('cuda:0')
        first_times_gpu = first_times.to('cuda:0')

        counts = event_gpu.event_volume(batch_gpu, video_len.cuda(), timestamps_gpu, first_times_gpu, nbins).cpu()
        counts2 = event_cpu.event_volume(batch, video_len, timestamps, first_times, nbins)
        diff = counts - counts2
        assert diff.abs().max().item() == 0


@pytest.mark.skipif(CUDA_NOT_AVAILABLE, reason="this machine does not have gpu available")
def pytestcase_cpu_gpu_event_volume_split(dataset_dir):
    """
    Runs the CPU & GPU versions.
    Make sure they are equivalent for the voxel grid of events with separated channels
    """
    # GIVEN
    path = os.path.join(dataset_dir, "openeb", "core_ml", "mini_image_dataset")
    batch_size = 4
    nbins = 5
    height, width = 60, 80
    threshold_mu = 0.1
    refractory_period = 10
    leak_rate = 0.1
    cutoff_hz = 90
    dataloader = make_video_dataset(path, 0, batch_size, height, width, 30, 300, seed=1)

    batches = [batch for batch in islice(dataloader, 5)]

    event_cpu = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_rate, cutoff_hz)
    event_gpu = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_rate, cutoff_hz)
    event_gpu.to('cuda:0')

    # THEN
    for batch_dict in batches:
        batch = event_gpu.log_images(batch_dict['images'].squeeze())
        timestamps = batch_dict['timestamps']
        first_times = batch_dict['first_times']
        video_len = batch_dict['video_len']
        batch_gpu = batch.to('cuda:0')
        timestamps_gpu = timestamps.to('cuda:0')
        first_times_gpu = first_times.to('cuda:0')
        counts = event_gpu.event_volume(batch_gpu, video_len.cuda(), timestamps_gpu, first_times_gpu, nbins,
                                        split_channels=True).cpu()
        counts2 = event_cpu.event_volume(batch, video_len, timestamps, first_times, nbins, split_channels=True)
        diff = counts - counts2
        assert diff.abs().max().item() == 0


@pytest.mark.skipif(CUDA_NOT_AVAILABLE, reason="this machine does not have gpu available")
def pytestcase_cpu_gpu_event(dataset_dir):
    """
    Runs the CPU & GPU versions.
    Make sure they are equivalent for the events
    """
    # GIVEN
    path = os.path.join(dataset_dir, "openeb", "core_ml", "mini_image_dataset")
    batch_size = 4
    height, width = 60, 80
    threshold_mu = 0.1
    refractory_period = 10
    nbins = 10
    leak_rate = 0.1
    cutoff_hz = 450
    dataloader = make_video_dataset(path, 0, batch_size, height, width, 30, 300, seed=1)

    batches = [batch for batch in islice(dataloader, 5)]

    event_cpu = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_rate, cutoff_hz)
    event_gpu = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_rate, cutoff_hz)

    event_gpu.to('cuda:0')

    # THEN
    for batch_dict in batches:
        batch = event_gpu.log_images(batch_dict['images'].squeeze())
        timestamps = batch_dict['timestamps']
        first_times = batch_dict['first_times']
        video_len = batch_dict['video_len']
        batch_gpu = batch.to('cuda:0')
        timestamps_gpu = timestamps.to('cuda:0')
        first_times_gpu = first_times.to('cuda:0')
        events = event_gpu.get_events(batch_gpu, video_len.cuda(), timestamps_gpu, first_times_gpu)
        events2 = event_cpu.get_events(batch, video_len, timestamps, first_times)

        t1 = events[:, -1]
        idx = torch.argsort(t1).to('cpu')
        events = events[idx]
        events2 = events2[idx]
        diff = (events.cpu() - events2)
        assert diff.abs().max().item() == 0


def pytestcase_event_count_equivalence(dataset_dir):
    """
    Here we make sure that:
    video -> event_count == video -> aer -> event_count
    """
    # GIVEN
    path = os.path.join(dataset_dir, "openeb", "core_ml", "mini_image_dataset")
    batch_size = 1
    height, width = 460, 480
    threshold_mu = 0.1
    refractory_period = 10
    nbins = 10
    leak_rate = 0.0
    cutoff_hz = 0

    dataloader = make_video_dataset(path, 0, batch_size, height, width, 30, 300, seed=1)
    batches = [batch for batch in islice(dataloader, 5)]

    sim1 = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_rate, cutoff_hz)
    sim2 = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_rate, cutoff_hz)

    sim2.thresholds[...] = sim1.thresholds

    # THEN
    for i, batch_dict in enumerate(batches):
        batch = sim1.log_images(batch_dict['images'].squeeze(0))
        timestamps = batch_dict['timestamps']
        first_times = batch_dict['first_times']
        video_len = batch_dict['video_len']

        start_times = sim1.prev_image_ts * (1 - first_times) + timestamps[:, 0] * first_times
        durations = timestamps[:, -1] - start_times

        counts1 = sim1.count_events(batch, video_len, timestamps, first_times)
        events = sim2.get_events(batch, video_len, timestamps, first_times)
        events[:, 3] = 1
        counts2 = event_image(events, batch_size, height, width)

        diff = counts1 - counts2
        assert torch.allclose(counts1, counts2.to(counts1), atol=1e-3)


def pytestcase_event_count_event_volume_equivalence(dataset_dir):
    """
    Here we make sure that:
    video -> event-volume (nearest) == event count
    """
    # GIVEN
    path = os.path.join(dataset_dir, "openeb", "core_ml", "mini_image_dataset")
    batch_size = 1
    height, width = 460, 480
    threshold_mu = 0.1
    refractory_period = 10
    nbins = 5
    leak_rate = 0.0
    cutoff_hz = 0
    mode = 'nearest'

    dataloader = make_video_dataset(path, 0, batch_size, height, width, 30, 300, seed=1)
    batches = [batch for batch in islice(dataloader, 5)]

    sim0 = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_rate, cutoff_hz)
    sim1 = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_rate, cutoff_hz)
    sim2 = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_rate, cutoff_hz)

    sim2.thresholds[...] = sim1.thresholds

    # THEN
    for i, batch_dict in enumerate(batches):
        batch = sim1.log_images(batch_dict['images'].squeeze(0))
        timestamps = batch_dict['timestamps']
        first_times = batch_dict['first_times']
        video_len = batch_dict['video_len']

        start_times = sim1.prev_image_ts * (1 - first_times) + timestamps[:, 0] * first_times
        durations = timestamps[:, -1] - start_times

        counts1 = sim0.count_events(batch, video_len, timestamps, first_times, True, True)

        vol2 = sim2.event_volume(batch, video_len, timestamps, first_times, nbins, mode, split_channels=True)

        counts2 = vol2.abs().sum(dim=1)
        assert torch.allclose(counts1, counts2.int(), atol=1e-3)


def pytestcase_event_volume_equivalence(dataset_dir):
    """
    Here we make sure that:
    video -> event-volume == video -> aer -> event_volume
    """
    # GIVEN
    path = os.path.join(dataset_dir, "openeb", "core_ml", "mini_image_dataset")
    batch_size = 1
    height, width = 460, 480
    threshold_mu = 0.1
    refractory_period = 10
    nbins = 4
    leak_rate = 0.0
    cutoff_hz = 0
    mode = 'nearest'

    dataloader = make_video_dataset(path, 0, batch_size, height, width, 30, 300, seed=1)
    batches = [batch for batch in islice(dataloader, 5)]

    sim1 = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_rate, cutoff_hz)
    sim2 = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_rate, cutoff_hz)

    sim2.thresholds[...] = sim1.thresholds

    # THEN
    for i, batch_dict in enumerate(batches):
        batch = sim1.log_images(batch_dict['images'].squeeze(0))
        timestamps = batch_dict['timestamps'].long()
        first_times = batch_dict['first_times']
        video_len = batch_dict['video_len']

        start_times = sim1.prev_image_ts * (1 - first_times) + timestamps[:, 0] * first_times
        durations = timestamps[:, -1] - start_times

        events = sim1.get_events(batch, video_len, timestamps, first_times)
        vol1 = event_volume(events, batch_size, height, width, start_times, durations, nbins, mode)
        vol2 = sim2.event_volume(batch, video_len, timestamps, first_times, nbins, mode)

        mean_per_bin_1 = vol1.sum(-1).sum(-1).mean(0).cpu().abs().numpy()
        mean_per_bin_2 = vol2.sum(-1).sum(-1).mean(0).cpu().abs().numpy()

        assert (mean_per_bin_1 > 0).any()
        assert (mean_per_bin_2 > 0).any()
        assert torch.allclose(vol1, vol2, atol=1e-4)


def pytestcase_new_old_different_on_off_ths(dataset_dir):
    """
    Testing that Old Simulator is equivalent to the New GPU one, 
    when usin different ON and OFF ths
    """
    path = os.path.join(dataset_dir, "openeb", "core_ml", "mini_image_dataset")
    batch_size = 1
    height, width = 256, 256
    threshold_mu_off = 0.2
    threshold_mu_on = 0.1
    threshold_std = 0.001
    refractory_period = 100
    nbins = 1
    dataloader = make_video_dataset(path, 0, batch_size, height, width, 30, 300, seed=1)
    batches = [batch for batch in islice(dataloader, 40)]

    event_cpu = GPUEventSimulator(batch_size, height, width, [threshold_mu_off, threshold_mu_on], 0, refractory_period)
    old_event_cpu = EventSimulator(height, width, threshold_mu_on, threshold_mu_off, refractory_period, 0)
    old_event_cpu.Cps = event_cpu.thresholds[1, 0].numpy()
    old_event_cpu.Cns = event_cpu.thresholds[0, 0].numpy()

    for i, batch_dict in enumerate(batches):
        batch = batch_dict['images'].squeeze(0)
        timestamps = batch_dict['timestamps']
        first_times = batch_dict['first_times']
        log_batch = event_cpu.log_images(batch)

        start_times = timestamps[:, 0] * first_times + (1 - first_times) * event_cpu.prev_image_ts
        durations = timestamps[:, -1] - start_times

        # old: reset
        if first_times[0] == 1:
            old_event_cpu.reset()
            old_event_cpu.Cps = event_cpu.thresholds[1, 0].numpy()
            old_event_cpu.Cns = event_cpu.thresholds[0, 0].numpy()

        # old: actual simulation
        for t in range(log_batch.shape[-1]):
            img = log_batch[:, :, t].numpy().copy()
            ts = timestamps[0, t].item()
            old_event_cpu.log_image_callback(img, ts)

        old_events = old_event_cpu.get_events().copy()
        old_event_cpu.flush_events()

        events = event_cpu.get_events(log_batch, batch_dict['video_len'], timestamps, first_times)
        t1 = events[:, -1]
        idx = torch.argsort(t1)
        events = events[idx]

        if not len(events):
            continue

        # Event-Volume Difference?
        ev_vol1 = event_volume(events, batch_size, height, width, start_times, durations, nbins, 'nearest')

        events2 = torch.zeros((len(old_events), 5), dtype=torch.int32)
        events2[:, 1] = torch.from_numpy(old_events['x'] * 1.0)
        events2[:, 2] = torch.from_numpy(old_events['y'] * 1.0)
        events2[:, 3] = torch.from_numpy(old_events['p'] * 2.0 - 1)
        events2[:, 4] = torch.from_numpy(old_events['t'] * 1.0)

        ev_vol2 = event_volume(events2, batch_size, height, width, start_times, durations, nbins, 'nearest')
        diff = (ev_vol1 - ev_vol2)
        assert diff.abs().max().item() == 0


def pytestcase_event_volume_equivalence_different_on_off_ths(dataset_dir):
    """
    Here we make sure that:
    video -> event-volume == video -> aer -> event_volume
    in the case we use different ON and OFF ths
    """
    # GIVEN
    path = os.path.join(dataset_dir, "openeb", "core_ml", "mini_image_dataset")
    batch_size = 1
    height, width = 460, 480
    threshold_mu = [0.1, 0.2]
    refractory_period = 10
    nbins = 4
    leak_rate = 0.0
    cutoff_hz = 0
    mode = 'nearest'

    dataloader = make_video_dataset(path, 0, batch_size, height, width, 30, 300, seed=1)
    batches = [batch for batch in islice(dataloader, 5)]

    sim1 = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_rate, cutoff_hz)
    sim2 = GPUEventSimulator(batch_size, height, width, threshold_mu, 0, refractory_period, leak_rate, cutoff_hz)

    sim2.thresholds[...] = sim1.thresholds

    # THEN
    for i, batch_dict in enumerate(batches):
        batch = sim1.log_images(batch_dict['images'].squeeze(0))
        timestamps = batch_dict['timestamps'].long()
        first_times = batch_dict['first_times']
        video_len = batch_dict['video_len']

        start_times = sim1.prev_image_ts * (1 - first_times) + timestamps[:, 0] * first_times
        durations = timestamps[:, -1] - start_times

        events = sim1.get_events(batch, video_len, timestamps, first_times)
        vol1 = event_volume(events, batch_size, height, width, start_times, durations, nbins, mode)
        vol2 = sim2.event_volume(batch, video_len, timestamps, first_times, nbins, mode)

        mean_per_bin_1 = vol1.sum(-1).sum(-1).mean(0).cpu().abs().numpy()
        mean_per_bin_2 = vol2.sum(-1).sum(-1).mean(0).cpu().abs().numpy()

        assert (mean_per_bin_1 > 0).any()
        assert (mean_per_bin_2 > 0).any()
        assert torch.allclose(vol1, vol2, atol=1e-4)
