# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Defines some tools to handle events.
In particular :
    -> defines events' types
    -> defines functions to read events from binary DAT files using numpy
    -> defines functions to write events to binary DAT files using numpy
"""

import os
import sys
import datetime
import numpy as np

EV_TYPES = {
    0: [('t', 'u4'), ('_', 'i4')],
    12: [('t', 'u4'), ('_', 'i4')],
    14: [("p", "i2"), ("t", "i8"), ("id", "i2")],
    40: [('t', 'u4'), ('_', 'i4'), ('vx', 'f4'), ('vy', 'f4'), ('center_x', 'f4'), ('center_y', 'f4'),
         ('id', 'u4')]
}

DECODE_DTYPES = {
    0: {'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'],
        'offsets': [0, 2, 4, 8], 'itemsize': 16},
    12: {'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'],
         'offsets': [0, 2, 4, 8], 'itemsize': 16},
    14: {'names': ['p', 't', 'id'], 'formats': ['<i2', '<i8', '<i2'], 'offsets': [0, 8, 16], 'itemsize': 24},
    40: {'names': ['x', 'y', 'p', 't', 'vx', 'vy', 'center_x', 'center_y', 'id'],
         'formats': ['<u2', '<u2', '<i2', '<i8', 'f4', 'f4', 'f4', 'f4', 'u4'],
         'offsets': [0, 2, 4, 8, 16, 20, 24, 28, 32], 'itemsize': 36}}

EV_STRINGS = {
    0: 'Event2D',
    12: 'EventCD',
    14: 'EventExtTrigger',
    40: 'EventOpticalFlow'
}

X_MASK = 2**14 - 1  # 18 zeros followed by 14 ones when formulated as a binary number.
Y_MASK = 2**28 - 2**14  # 4 zeros, 14 ones and then 14 zeros.
P_MASK = 2 ** 29 - 2**28  # 3 zeros a one and 28 zeros.


def load_events(filename, ev_count=-1, ev_start=0):
    """
    Loads event data from files.

    Args :
        path (string): Path to a DAT file.
        event_count (int): Number of events to load. (all events in the file we be loaded if set to the default -1).
        ev_start (int): Index of the first event.

    Returns :
        a numpy array behaving like a dictionary containing the fields ts, x, y, p
    """
    with open(filename, 'rb') as f:
        _, ev_type, ev_size, _ = parse_header(f)
        if ev_start > 0:
            f.seek(ev_start * ev_size, 1)

        dtype = EV_TYPES[ev_type]
        dat = np.fromfile(f, dtype=dtype, count=ev_count)
        xyp = None
        if ('_', 'i4') in dtype:
            x = np.bitwise_and(dat["_"], X_MASK)
            y = np.right_shift(
                np.bitwise_and(dat["_"], Y_MASK), 14)
            p = np.right_shift(np.bitwise_and(dat["_"], P_MASK), 28)
            xyp = (x, y, p)
        return _dat_transfer(dat, DECODE_DTYPES[ev_type], xyp=xyp)


def _dat_transfer(dat, decoded_dtype, xyp=None):
    """
    Transfers the fields present in dtype from an old data structure to a new data structure
    xyp should be passed as a tuple.

    Args :
        - dat vector as directly read from file
        - decoded_dtype _numpy dtype_ as a list of couple of field name/ type eg [('x','i4'), ('y','f2')]
        - xyp optional tuple containing x,y,p extracted from a field '_'and untangled by bitshift and masking
    """
    variables = []
    xyp_index = -1
    for i, name in enumerate(dat.dtype.names):
        if name == '_':
            xyp_index = i
            continue
        variables.append((name, dat[name]))
    if xyp is not None and xyp_index == -1:
        print("Error dat didn't contain a '_' field !")
        return
    new_dat = np.empty(dat.shape[0], dtype=decoded_dtype)
    if xyp:
        new_dat["x"] = xyp[0]
        new_dat["y"] = xyp[1]
        new_dat["p"] = xyp[2]
    for (name, arr) in variables:
        new_dat[name] = arr
    return new_dat


def stream_events(file_handle, buffer, dtype, ev_count=-1):
    """
    Streams data from opened file_handle.
    Args :
        file_handle: file object, needs to be opened.
        buffer (events numpy array): Pre-allocated buffer to fill with events
        dtype (numpy dtype):  expected fields
        ev_count (int): Number of events
    """
    dat = np.fromfile(file_handle, dtype=dtype, count=ev_count)
    count = len(dat)
    for name, _ in dtype:
        if name == '_':
            buffer['x'][:count] = np.bitwise_and(dat["_"], X_MASK)
            buffer['y'][:count] = np.right_shift(np.bitwise_and(dat["_"], Y_MASK), 14)
            buffer['p'][:count] = np.right_shift(np.bitwise_and(dat["_"], P_MASK), 28)
        else:
            buffer[name][:count] = dat[name]


def count_events(filename):
    """
    Returns the number of events in a DAT file.

    Args :
        filename (string): Path to a DAT file.
    """
    with open(filename, 'rb') as f:
        bod, _, ev_size, _ = parse_header(f)
        f.seek(0, os.SEEK_END)
        eod = f.tell()
        if (eod - bod) % ev_size != 0:
            raise Exception("unexpected format !")
        return (eod - bod) // ev_size


def parse_header(f):
    """
    Parses the header of a DAT file and put the file cursor at the beginning of the binary data part.

    Args:
        f (file): File handle to a DAT file.

    Returns:
        int position of the file cursor after the header
        int type of event
        int size of event in bytes
        size (height, width) tuple of int or None
    """
    f.seek(0, os.SEEK_SET)
    bod = None
    end_of_header = False
    header = []
    num_comment_line = 0
    size = [None, None]
    # parse header
    while not end_of_header:
        bod = f.tell()
        line = f.readline()
        if sys.version_info > (3, 0):
            first_item = line.decode("latin-1")[:2]
        else:
            first_item = line[:2]

        if first_item != '% ':
            end_of_header = True
        else:
            words = line.split()
            if len(words) > 1:
                if words[1] == 'Date':
                    header += ['Date', words[2] + ' ' + words[3]]
                if words[1] == 'Height' or words[1] == b'Height':
                    size[0] = int(words[2])
                    header += ['Height', words[2]]
                if words[1] == 'Width' or words[1] == b'Width':
                    size[1] = int(words[2])
                    header += ['Width', words[2]]
            else:
                header += words[1:3]
            num_comment_line += 1
    # parse data
    f.seek(bod, os.SEEK_SET)

    if num_comment_line > 0:  # Ensure compatibility with previous files.
        # Read event type
        ev_type = np.frombuffer(f.read(1), dtype=np.uint8)[0]
        # Read event size
        ev_size = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
    else:
        ev_type = 0
        ev_size = sum([int(n[-1]) for _, n in EV_TYPES[ev_type]])

    bod = f.tell()
    return bod, ev_type, ev_size, size


class DatWriter(object):
    """Convenience class used to write Event2D to a DAT file.

    The constructor writes the header for a DAT file.

    Args:
        filename (string): Path to the destination file
        height (int): Imager height in pixels
        width (int): Imager width in pixels

    Examples:
        >>> f = DatWriter("my_file_td.dat", height=480, width=640)
        >>> f.write(np.array([(3788, 283, 116, 0), (3791, 271, 158, 1)],
                             dtype=[('t', '<u4'), ('x', '<u2'), ('y', '<u2'), ('p', 'u1')]))
        >>> f.close()

    """

    def __init__(self, filename, height=240, width=320):
        if max(height, width) > 2**14 - 1:
            raise ValueError('Coordinates value exceed maximum range in'
                             ' binary DAT file format max({:d},{:d}) vs 2^14 - 1'.format(
                                 height, width))
        self.ev_type = 0
        self._path = filename
        self.file = open(filename, 'w')
        # write header
        self.file.write('% Data file containing {:s} events.\n'
                        '% Version 2\n'.format(EV_STRINGS[self.ev_type]))
        now = datetime.datetime.utcnow()
        self.file.write("% Date {}-{}-{} {}:{}:{}\n".format(now.year,
                                                            now.month, now.day, now.hour,
                                                            now.minute, now.second))

        self.file.write('% Height {:d}\n'
                        '% Width {:d}\n'.format(height, width))
        self.height = height
        self.width = width

        # write type and bit size
        ev_size = sum([int(b[-1]) for _, b in EV_TYPES[self.ev_type]])

        np.array([self.ev_type, ev_size], dtype=np.uint8).tofile(self.file)
        self.file.flush()

        self.ev_count = 0
        self.current_time = 0

    def __repr__(self):
        """String representation of a `DatWriter` object.

        Returns:
            string describing the DatWriter state and attributes
        """
        wrd = ''
        wrd += 'DatWriter: path {} \n'.format(self._path)
        wrd += 'Width {}, Height  {}\n'.format(self.width, self.height)
        wrd += 'events written : {}, last timestamp {}\n'.format(self.ev_count, self.current_time)
        return wrd

    def write(self, events):
        """
        Writes events of fields x,y,p,t into the file. Only Event2D events are supported

        Args:
            events (numpy array): Events to write
        """
        # if input is empty do nothing
        if not len(events):
            return

        assert events['t'][0] >= self.current_time, "events must be written in chronological order"
        # pack data as events
        dtype = EV_TYPES[0]
        data_to_write = np.empty(len(events['t']), dtype=dtype)

        for (name, typ) in events.dtype.fields.items():
            if name == 'x':
                x = events['x'].astype('i4')
            elif name == 'y':
                y = np.left_shift(events['y'].astype('i4'), 14)
            elif name == 'p':
                events['p'] = (events['p'] == 1).astype(events['p'].dtype)
                p = np.left_shift(events['p'].astype("i4"), 28)
            else:
                data_to_write[name] = events[name].astype(typ[0])

        data_to_write['_'] = x + y + p

        # write data
        data_to_write.tofile(self.file)
        self.file.flush()

        # update object state
        self.ev_count += len(events)
        self.current_time = events["t"][-1]

    def close(self):
        self.file.close()

    def __del__(self):
        self.file.close()
