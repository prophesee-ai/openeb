# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# pylint: disable=no-member

"""
Module that enables Parallel Multistreaming.

We define an IterableDataset that streams several iterables.
When fed to a Pytorch DataLoader with batch_size=None,
this streams batches from one worker at a time.
This has the effect of enabling parallel streaming.

The StreamLoader is a class built on top of DataLoader,
that fuses batches together so batches are always temporally
coherent.

Notice that you can also avoid using this fusion and just use
a regular DataLoader, and have multiple neural networks indexed
by worker's id.
"""
import random
import math
import time
import torch
import numpy as np

from collections import deque
from itertools import chain, cycle
from torch.utils.data import IterableDataset, DataLoader

# https://github.com/pytorch/pytorch/issues/11201
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')  # nopep8


def split_batch_size(batch_size, num_workers):
    """Returns the number of files to handle per worker

    Args:
        batch_size (int): total batch size
        num_workers (int): number of workers
    """
    num_workers = min(num_workers, batch_size)
    split_sizes = np.zeros(num_workers, dtype=int)
    split_size, remaining_files = divmod(batch_size, num_workers)
    split_sizes[...] = split_size

    split_sizes[:remaining_files] += 1

    return split_sizes.tolist()


def split_dataset_sizes(stream_list, split_sizes):
    """Splits with different sizes proportional to the number of files each worker has to handle.

    Args:
        stream_list (list): list of stream path
        split_sizes (list): batch size per worker
    """
    out = []
    split_sizes_array = np.array(split_sizes, dtype=float)
    split_sizes_array *= len(stream_list) / split_sizes_array.sum()
    ends = np.round(np.cumsum(split_sizes_array)).astype(int)
    start = 0
    for end in ends:
        out.append(stream_list[start: end])
        start = end
    return out


def batch_to(batch, device):
    if hasattr(batch, "to"):
        return batch.to(device)
    elif isinstance(batch, list):
        return [batch_to(b, device) for b in batch]
    elif isinstance(batch, tuple):
        return tuple(batch_to(b, device) for b in batch)
    elif isinstance(batch, dict):
        return {key: batch_to(value, device) for key, value in batch.items()}
    else:
        return batch


def resample_to_batch_size(stream_list, batch_size):
    """Resamples list to fit batch_size iterators

    Args:
        stream_list (list): list of streams
        batch_size (int): batch size
    """
    stream_list = random.sample(stream_list, len(stream_list)) +\
        random.choices(stream_list, k=batch_size - len(stream_list) % batch_size)
    return stream_list


class StreamDataset(IterableDataset):
    """Stream Dataset
    An Iterable Dataset zipping a group of iterables streams together.

    Args:
        stream_list (list): list of streams (path/ metadata)
        streamer (object): an iterator (user defined)
        batch_size (int): total batch size
        padding_mode (str): "zeros" "data" or "none", see "get_zip" function
        fill_value (object): padding value
        seed (int): seed integer to make the dataloading deterministic
    """

    def __init__(self, stream_list, streamer, batch_size, padding_mode, fill_value, seed=None):
        self.stream_list = stream_list
        self.batch_size = batch_size
        self.streamer = streamer
        self.padding_mode = padding_mode
        self.fill_value = fill_value
        assert batch_size <= len(
            stream_list), f"batch_size {batch_size} is larger than the number of files {len(stream_list)} "
        assert padding_mode in ['zeros', 'data']
        if padding_mode == 'zeros':
            assert fill_value is not None
        self.seed = seed

    def shuffle(self):
        random.shuffle(self.stream_list)

    def _set_seed(self, seed=None):
        """ so that data is different along threads and epochs"""
        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        seed = int(time.time()) + worker_id if seed is None else seed + worker_id
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def _worker_init_fn(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        num_workers = 1 if not worker else worker.num_workers
        split_sizes = split_batch_size(self.batch_size, num_workers)
        stream_groups = split_dataset_sizes(self.stream_list, split_sizes)
        split_size = split_sizes[worker_id]
        stream_group = stream_groups[worker_id]
        random.shuffle(stream_group)
        return split_size, stream_group

    def __iter__(self):
        """Iterates over stream files

        Note: Here the scheduling of iterable is done at the beginning.
        Instead User can change this code to map lazily iterables.
        """
        self._set_seed(self.seed)

        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        split_size, stream_list = self._worker_init_fn()
        if len(stream_list) < split_size:
            print('worker#', worker_id, ': Stopping... Number of streams < split_size')
            return  # PEP479

        """
        Just-in-time mapping
        The scheduling is done as we iterate.
        """
        iterators = [iter(self.streamer(stream_list[i])) for i in range(split_size)]
        actives = np.ones(len(iterators), dtype=bool)
        file_pos = split_size - 1
        while actives.any():
            values = []
            for i, it in enumerate(iterators):
                try:
                    value = next(it)
                except StopIteration:
                    file_pos += 1
                    actives[i] = (file_pos < len(stream_list))
                    if self.padding_mode == 'data' or actives[i]:
                        num = file_pos % len(stream_list)
                        iterators[i] = iter(self.streamer(stream_list[num]))
                        try:
                            value = next(iterators[i])
                        except StopIteration:
                            return  # PEP479
                    elif self.padding_mode == 'zeros':
                        value = self.fill_value
                values.append(value)
            if actives.any():
                yield tuple(values), worker_id
        # current worker is done
        yield (None, worker_id)


class StreamDataLoader(object):
    """StreamDataLoader
    Wraps around the DataLoader to handle the asynchronous batches.

    Args:
        dataset (StreamDataset): dataset streaming multiple iterables
        num_workers (int): number of workers
        collate_fn (function): function to collate batch parts
    """

    def __init__(self, dataset, num_workers, collate_fn):
        self.dataset = dataset
        num_workers = min(dataset.batch_size, num_workers)
        self.num_workers = max(1, num_workers)
        assert isinstance(dataset, StreamDataset)
        self.dataloader = DataLoader(dataset, batch_size=None, num_workers=num_workers,
                                     collate_fn=lambda x: x, drop_last=False)
        self.collate_fn = collate_fn
        self.split_sizes = split_batch_size(self.dataset.batch_size, self.num_workers)
        self.device = torch.device('cpu')

    def cuda(self, device=torch.device('cuda')):
        """Sets the StreamDataLoader to copy tensors to GPU memory before returning them.

        Args:
            device (torch.device): The destination GPU device. Defaults to the current CUDA device.
        """
        assert torch.cuda.is_available()
        self.device = device
        return self

    def to(self, device):
        """Sets the StreamDataLoader to copy tensors to the given device before returning them.

        Args:
            device (torch.device): The destination GPU device. For instance `torch.device('cpu')`
                or torch.device('cuda').
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        assert isinstance(self.device, torch.device)
        return self

    def cpu(self):
        """Sets the StreamDataLoader to leave tensors on CPU."""
        self.device = torch.device('cpu')
        return self

    @property
    def batch_size(self):
        return self.dataset.batch_size

    def __iter__(self):
        cache = [deque([]) for i in range(self.num_workers)]
        last_worker_id = -1
        active_workers = np.ones(self.num_workers, dtype=bool)

        for data in self.dataloader:
            data, worker_id = data
            assert active_workers[worker_id]
            if active_workers.sum() == sum([len(v) for v in cache]):
                # All active workers should have received their data at this point

                # pytorch dataloader cycles through workers and receives their results in order
                # see: https://discuss.pytorch.org/t/in-what-order-do-dataloader-workers-do-their-job/88288/2
                # which means we can assert additional constraints
                assert worker_id <= last_worker_id
                # check non-zero queues correspond to active workers
                assert (np.array([len(v) > 0 for v in cache]) == active_workers).all()
                # check active workers have exactly one element
                assert (np.array([len(v) == 1 for v in cache]) == active_workers).all()

                # process active queues
                values = []
                for i, deq in enumerate(cache):
                    if not active_workers[i]:
                        value = [self.dataset.fill_value] * self.split_sizes[i]
                    else:
                        value = deq.popleft()
                    values.append(value)

                batch = chain.from_iterable(values)

                yield batch_to(self.collate_fn(batch), self.device)
                # check all deques are empty
                assert np.array([len(v) == 0 for v in cache]).all()

            # now process the data we've received (either add it to the queue or disable the worker)
            if data is None:
                if self.dataset.padding_mode == "data":
                    # stop as soon as the first worker is done
                    cache = [deque([]) for i in range(self.num_workers)]
                    active_workers = np.zeros(self.num_workers, dtype=bool)
                    break
                # mark the worker as inactive (padding will be used from now on)
                active_workers[worker_id] = False
            else:
                # data received for this worker is valid, add it to the queue
                assert len(cache[worker_id]) == 0
                cache[worker_id].append(data)
            last_worker_id = worker_id

        assert not active_workers.any()
        assert sum([len(v) for v in cache]) == 0
