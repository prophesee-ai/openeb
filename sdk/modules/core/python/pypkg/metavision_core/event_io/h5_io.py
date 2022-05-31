# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# pylint: disable=W0611
"""
h5 io for event storage
you can use 2 compression backends:
    - zlib (fast read, slow write)
    - zstandard (fast read, fast write, but you have to install it)
"""
import h5py
import zlib
try:
    import zstandard
except BaseException:
    pass
import numpy as np
from metavision_sdk_base import EventCD


class H5EventsWriter(object):
    """
    Compresses & Writes Event Packets as they are read

    Args:
        out_name (str): destination path
        height (int): height of recording
        width (int): width of recording
        compression_backend (str): compression api to be called, defaults to zlib.
        If you can try to use zstandard which is faster at writing.
    """

    def __init__(self, out_name, height, width, compression_backend="zlib"):
        dt = h5py.vlen_dtype(np.dtype("uint8"))
        dt2 = np.int64
        self.f = h5py.File(out_name, "w")
        self.dataset_size_increment = 1000
        shape = (self.dataset_size_increment,)
        self.dset = self.f.create_dataset("event_buffers", shape, maxshape=(None,), dtype=dt)
        self.ts = self.f.create_dataset("event_buffers_start_times", shape, maxshape=(None,), dtype=dt2)
        self.dset.attrs["compression_backend"] = compression_backend
        self.dset.attrs["height"] = height
        self.dset.attrs["width"] = width

        compress_api = globals()[compression_backend]
        self.compress_api = compress_api
        self.index = 0
        self.is_close = False

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def write(self, events):
        """
        Writes event buffer into a compressed packet

        Args:
            events (ndarray): events of type EventCD
        """
        if not len(events):
            return
        if self.index >= len(self.dset):
            new_len = self.dset.shape[0] + self.dataset_size_increment
            self.dset.resize((new_len,))
            self.ts.resize((new_len,))
        self.ts[self.index] = events['t'][0]

        zipped_data = self.compress_api.compress(events)
        zipped_data = np.frombuffer(zipped_data, dtype="uint8")

        self.dset[self.index] = zipped_data
        self.index += 1

    def close(self):
        if not self.is_close:
            self.dset.resize((self.index,))
            self.ts.resize((self.index,))
            self.f.close()
            self.is_close = True

    def __del__(self):
        self.close()


class H5EventsReader(object):
    """
    Reads & Seeks into a h5 file of compressed event packets.

    Args:
        src_name (str): input path
    """

    def __init__(self, path):
        self.path = path
        dt = h5py.vlen_dtype(np.dtype("uint8"))
        dt2 = np.int64
        self.f = h5py.File(path, "r")
        self.len = len(self.f["event_buffers"])
        compression_backend = self.f["event_buffers"].attrs["compression_backend"]
        self.height = self.f["event_buffers"].attrs["height"]
        self.width = self.f["event_buffers"].attrs["width"]
        self.start_times = self.f['event_buffers_start_times'][...]
        self.compress_api = globals()[compression_backend]
        self.start_index = 0
        self.sub_start_index = 0

    def __len__(self):
        return len(self.start_times)

    def seek_in_buffers(self, ts):
        idx = np.searchsorted(self.start_times, ts, side='left')
        return idx

    def seek_time(self, ts):
        idx = np.searchsorted(self.start_times, ts, side='left')
        self.start_index = max(0, idx - 1)
        zipped_data = self.f["event_buffers"][self.start_index]
        unzipped_data = self.compress_api.decompress(zipped_data.data)
        events = np.frombuffer(unzipped_data, dtype=EventCD)
        if self.start_index > 0:
            assert events['t'][0] <= ts
        self.sub_start_index = np.searchsorted(events["t"], ts)
        self.sub_start_index = max(0, self.sub_start_index)

    def get_size(self):
        return (self.height, self.width)

    def __iter__(self):
        for i in range(self.start_index, len(self.f["event_buffers"])):
            zipped_data = self.f["event_buffers"][i]
            unzipped_data = self.compress_api.decompress(zipped_data.data)
            events = np.frombuffer(unzipped_data, dtype=EventCD)
            if i == self.start_index and self.sub_start_index > 0:
                events = events[self.sub_start_index:]
            yield events
