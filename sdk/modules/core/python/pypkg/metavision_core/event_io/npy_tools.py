# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Defines some tools to handle events, mimicking dat_tools.py.
In particular :
    -> defines functions to read events from binary .npy files using numpy
"""

import numpy as np


def stream_events(file_handle, buffer, dtype, ev_count=-1):
    """
    Streams data from opened file_handle

    Args :
        file_handle: File object, needs to be opened.
        buffer (events numpy array): Pre-allocated buffer to fill with events.
        dtype (numpy dtype): Expected fields.
        ev_count (int): Number of events.
    """
    dat = np.fromfile(file_handle, dtype=dtype, count=ev_count)
    count = len(dat)
    for name in dat.dtype.names:
        buffer[name][:count] = dat[name]


def parse_header(fhandle):
    """
    Parses the header of a .npy file

    Args:
        fhandle (file): File handle to a DAT file.

    Returns:
        int position of the file cursor after the header
        int type of event
        int size of event in bytes
        size (height, width) tuple of int or None
    """
    version = np.lib.format.read_magic(fhandle)
    shape, fortran, dtype = np.lib.format._read_array_header(fhandle, version)
    assert not fortran, "Fortran order arrays not supported"
    # Get the number of elements in one 'row' by taking
    # a product over all other dimensions.
    if len(shape) == 0:
        count = 1
    else:
        count = np.multiply.reduce(shape, dtype=np.int64)
    ev_size = dtype.itemsize
    assert ev_size != 0
    start = fhandle.tell()
    if 'ts' in dtype.names:
        # turn numpy.dtype into an iterable list
        ev_type = [(x, str(dtype.fields[x][0])) for x in dtype.names]
        # filter name to have only t and not ts
        ev_type = [(name if name != "ts" else "t", desc) for name, desc in ev_type]
        ev_type = [(name if name != "confidence" else "class_confidence", desc) for name, desc in ev_type]
    else:
        ev_type = dtype
    size = (None, None)

    return start, ev_type, ev_size, size
