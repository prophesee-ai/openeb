# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Unit tests for the multistream dataloader
"""
import platform
import pytest
import numpy as np

from metavision_core_ml.data.stream_dataloader import StreamDataLoader, StreamDataset
from metavision_core_ml.data.stream_dataloader import split_batch_size, split_dataset_sizes
from collections import defaultdict
from functools import partial


class DummyStream(object):
    def __init__(self, stream, num_tbins):
        self.pos = 0
        self.stream_num = stream[0]
        self.max_len = stream[1]
        self.num_tbins = num_tbins

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= self.max_len:
            raise StopIteration
        max_pos = min(self.pos + self.num_tbins, self.max_len)
        positions = [i for i in range(self.pos, max_pos)]
        self.pos = max_pos
        return positions, self.stream_num


def collate_fn(data_list):
    frame_nums, stream_nums = zip(*data_list)
    return {"frame_num": frame_nums, "stream_num": stream_nums}


class TestClassMultiStreams(object):
    def setup_dataloader(self, stream_list, num_workers, batch_size, num_tbins):
        iterator_fun = partial(DummyStream, num_tbins=num_tbins)
        fill_value = ([-1] * num_tbins, -1)
        dataset = StreamDataset(stream_list, iterator_fun, batch_size, "zeros", fill_value)
        dataloader = StreamDataLoader(dataset, num_workers, collate_fn)
        return dataloader

    def assert_all(self, dataloader, stream_list, num_tbins, batch_size):
        # WHEN
        streamed1 = defaultdict(list)
        for stream_num, stream_len in stream_list:
            stream_tuple = (stream_num, stream_len)
            stream = DummyStream(stream_tuple, num_tbins)
            for pos, _ in stream:
                streamed1[stream_num] += [pos]

        streamed2 = defaultdict(list)
        batch_number = defaultdict(list)
        for batch in dataloader:
            actual_batch_size = len(batch['stream_num'])
            # THEN: batch_size should always be equal to user defined batch_size
            assert batch_size == actual_batch_size

            for i in range(batch_size):
                stream_num = batch['stream_num'][i]
                if stream_num == -1:
                    continue
                streamed2[stream_num] += [batch['frame_num'][i]]
                batch_number[stream_num].append(i)

        # THEN: data is contiguous accross batches
        for k, v in batch_number.items():
            assert len(set(v)) == 1

        # THEN: ALL IS READ
        for k, v1 in streamed1.items():
            v2 = streamed2[k]
            assert v1 == v2

    def pytestcase_zero_pad_num_streams(self, tmpdir, dataset_dir):
        # num_streams%batch_size != 0 (2 worker)
        num_workers, num_streams, batch_size, num_tbins = 2, 11, 4, 5
        num_workers = 0 if platform.system() == 'Windows' else num_workers

        # GIVEN
        stream_list = [(i, num_tbins * np.random.randint(1, 4)) for i in range(num_streams)]
        dataloader = self.setup_dataloader(stream_list, num_workers, batch_size, num_tbins)

        # THEN
        self.assert_all(dataloader, stream_list, num_tbins, batch_size)

    def pytestcase_zero_pad_batch_size_greater_not_divisible(self, tmpdir, dataset_dir):
        # batch_size > num_streams_per_worker
        # batch_size%num_workers != 0
        num_workers, num_streams, batch_size, num_tbins = 3, 13, 7, 5
        num_workers = 0 if platform.system() == 'Windows' else num_workers

        # GIVEN
        stream_list = [(i, num_tbins * np.random.randint(1, 4)) for i in range(num_streams)]
        dataloader = self.setup_dataloader(stream_list, num_workers, batch_size, num_tbins)

        # THEN
        self.assert_all(dataloader, stream_list, num_tbins, batch_size)

    def pytestcase_zero_pad_batch_size_not_enough_streams(self, tmpdir, dataset_dir):
        # batch_size > num_streams_per_worker
        # batch_size%num_workers != 0
        num_workers, num_streams, batch_size, num_tbins = 3, 2, 7, 5
        num_workers = 0 if platform.system() == 'Windows' else num_workers

        stream_list = [(i, num_tbins * np.random.randint(1, 4)) for i in range(num_streams)]
        with pytest.raises(AssertionError):
            dataloader = self.setup_dataloader(stream_list, num_workers, batch_size, num_tbins)
            self.assert_all(dataloader, stream_list, num_tbins, batch_size)

    def pytestcase_split_size(self):
        stream_list = [i for i in range(3)]
        split_sizes = split_batch_size(batch_size=3, num_workers=2)
        stream_groups = split_dataset_sizes(stream_list, split_sizes)
        for stream_group, split_size in zip(stream_groups, split_sizes):
            assert len(stream_group) >= split_size

        # all streams must be in the splits
        assert sum([len(s) for s in stream_groups]) == len(stream_list)

    def pytestcase_split_size_correctness(self):
        # the sum of the split must be equal to the batch size, but should only differ by one at most.
        # GIVEN
        for batch_size in range(1, 10):
            for num_workers in range(1, batch_size + 1):
                # WHEN
                split_sizes = split_batch_size(batch_size, num_workers)
                # THEN
                split_sizes = np.array(split_sizes)
                assert np.sum(split_sizes) == batch_size
                assert np.max(split_sizes) - np.min(split_sizes) <= 1

    def pytestcase_split_num_workers_greater_than_batch_size(self):
        stream_list = [i for i in range(10)]
        split_sizes = split_batch_size(batch_size=3, num_workers=6)
        stream_groups = split_dataset_sizes(stream_list, split_sizes)
        for stream_group, split_size in zip(stream_groups, split_sizes):
            assert len(stream_group) >= split_size
