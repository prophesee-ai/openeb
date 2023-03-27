# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
Preprocessing functions in Pytorch
They take (N,5) events in (b,x,y,p,t) format.
"""
import numpy as np
import torch

from metavision_sdk_base import EventCD


def event_cd_to_torch(events):
    """
    Converts Events to Torch array
    Note that Polarities are mapped from {0;1} to {-1;1}

    Args:
        events (EventCD): structured array containing N events

    Returns:
        (Tensor) N,5 in batch_index, x, y, polarity, timestamp (micro-seconds) where the batch index value is 0.
    """
    event_th = torch.zeros((len(events), 5), dtype=torch.int32)
    event_th[:, 1] = torch.from_numpy(events['x'].astype(np.int32))
    event_th[:, 2] = torch.from_numpy(events['y'].astype(np.int32))
    event_th[:, 3] = torch.from_numpy(2 * events['p'].astype(np.int32) - 1)
    event_th[:, 4] = torch.from_numpy(events['t'].astype(np.int32))
    return event_th


def events_cd_list_to_torch(list_of_events_np_arrays):
    """
    Converts a list of Events CD numpy arrays into torch format (N,5) :
    batch_index, x, y, polarity_timestamp (micro-seconds)
    Note that Polarities are mapped from {0;1} to {-1;1}

    Args:
        list_of_events_np_arrays (list of np.array): list of structured arrays of EventCD

    Returns:
        (Tensor) N,5: value of batch_index corresponds to the position of the event array in the input list
    """
    nb_total_events = sum([ev.size for ev in list_of_events_np_arrays])
    event_th = torch.zeros((nb_total_events, 5), dtype=torch.int32)
    B = len(list_of_events_np_arrays)
    start_idx = 0
    for i in range(B):
        events = list_of_events_np_arrays[i]
        end_idx = start_idx + events.size
        event_th[start_idx:end_idx, 0] = i
        # To save memory, probably we can set up the types as below
        event_th[start_idx:end_idx, 1] = torch.from_numpy(events['x'].astype(np.int16))
        event_th[start_idx:end_idx, 2] = torch.from_numpy(events['y'].astype(np.int16))
        event_th[start_idx:end_idx, 3] = torch.from_numpy(2 * events['p'].astype(np.int8) - 1)
        event_th[start_idx:end_idx, 4] = torch.from_numpy(events['t'].astype(np.int64))
        start_idx = end_idx
    return event_th


def tensor_to_cd_events(events, batch_size, sort=True):
    """
    Convert the event produced by the GPU to our format for later reuse.

    Args:
        events (np.array): (N,5) in batch_index, x, y, polarity, timestamp (micro-seconds)
        batch_size (int): size of the batch
        sort (boolean): whether to sort event chronologically. You might want to avoid doing it if the
                        algorithm you are using doesn't rely on them.

    Returns:
        list of EventCD arrays, of lenght batch_size
    """
    events = events.cpu().numpy()
    res = []
    if events.size > 0:
        assert events[-1, 0] < batch_size
    indices = np.searchsorted(events[:, 0], np.arange(batch_size + 1))
    for i, ind in enumerate(indices[:-1]):
        slice = events[ind:indices[i + 1]]
        slicenp = np.zeros(len(slice), dtype=EventCD)
        if sort:
            sorting_indices = np.argsort(slice[:, 4], kind='stable')
            slice = slice[sorting_indices]
        slicenp["x"] = slice[:, 1]
        slicenp["y"] = slice[:, 2]
        slicenp["p"] = slice[:, 3] > 0
        slicenp['t'] = slice[:, 4]
        res.append(slicenp)
    return res


def event_image(events, batch_size, height, width):
    """
    Densifies events into an image

    Args:
        events (tensor): N,5 (b,x,y,p,t)
        batch_size (int): batch size
        height (int): height of output image
        width (int): width of output image
    """
    bs = events[:, 0].long()
    xs = events[:, 1].long()
    ys = events[:, 2].long()
    ps = events[:, 3].float()
    img = torch.zeros((batch_size, height, width), dtype=torch.float32, device=events.device)
    img.index_put_((bs, ys, xs), ps, accumulate=True)
    return img


def event_volume(events, batch_size, height, width, start_times, durations, nbins, mode='bilinear', vol=None):
    """
    Densifies events into an volume
    (uniform cut)

    Args:
        events (tensor): N,5 (b,x,y,p,t)
        batch_size (int): batch size
        height (int): height of output volume
        width (int): width of output volume
        start_times: (B,) start times of each volume
        durations: (B,) durations for each volume
        nbins (int): number of time bins for output volume
        mode (str): either "bilinear" or "nearest" interpolation of voxels.
    """
    bs = events[:, 0].long()
    xs = events[:, 1].long()
    ys = events[:, 2].long()
    ps = events[:, 3].float()
    ts = events[:, 4].float()

    start_times = start_times[bs]
    durations = durations[bs]
    ti_star = (ts - start_times) * nbins / durations - 0.5
    lbin = torch.floor(ti_star)
    lbin = torch.clamp(lbin, min=0, max=nbins - 1)
    if vol is None:
        vol = torch.zeros((batch_size, nbins, height, width), dtype=torch.float32, device=events.device)
    if mode == 'bilinear':
        rbin = torch.clamp(lbin + 1, max=nbins - 1)
        lvals = torch.clamp(1 - torch.abs(lbin - ti_star), min=0)
        rvals = 1 - lvals
        vol.index_put_((bs, lbin.long(), ys, xs), ps * lvals, accumulate=True)
        vol.index_put_((bs, rbin.long(), ys, xs), ps * rvals, accumulate=True)
    else:
        vol.index_put_((bs, lbin.long(), ys, xs), ps, accumulate=True)
    return vol
