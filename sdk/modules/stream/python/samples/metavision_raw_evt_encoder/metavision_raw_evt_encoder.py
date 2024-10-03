# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
This application demonstrates how to use Metavision SDK Stream module to decode an event recording, process it and
encode it back to RAW EVT2 format.
"""

import os
import sys
from metavision_sdk_base import EventCDBuffer
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import FlipYAlgorithm
from metavision_sdk_stream import RAWEvt2EventFileWriter


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision RAW EVT encoder sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-event-file', type=str, dest='event_file_path',
                        required=True, help="Path to input event file (RAW or HDF5)")
    parser.add_argument('-o', '--output-file', type=str, dest='output_file', default="",
                        help="Path to RAW output file. If not specified, will use a modified version of the input path.")
    parser.add_argument('--encode-triggers', action='store_true', dest='encode_triggers',
                        help="Flag to activate encoding of external trigger events.")
    parser.add_argument('--max-event-latency', type=int, dest='max_event_latency', default=-1,
                        help="Maximum latency in camera time for the reception of events, infinite by default.")
    parser.add_argument('-s', '--start-ts', type=int,
                        default=0, help="Start time in microsecond")
    parser.add_argument('-d', '--max-duration', type=int,
                        default=sys.maxsize, help="Maximum duration in microsecond")
    parser.add_argument('--delta-t', type=int, default=100000,
                        help="Duration of served event slice in us.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.isfile(args.event_file_path):
        raise TypeError(f'Fail to access file: {args.event_file_path}')
    if args.output_file == "":
        args.output_file = args.event_file_path[:-4] + "_evt_encoded.raw"

    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=args.delta_t, start_ts=args.start_ts,
                                 max_duration=args.max_duration)
    stream_height, stream_width = mv_iterator.get_size()
    yflipper = FlipYAlgorithm(stream_height-1)
    writer = RAWEvt2EventFileWriter(
        stream_width, stream_height, args.output_file, args.encode_triggers, {}, args.max_event_latency)

    print("Processing input file...")
    evs_processed_buf = EventCDBuffer()
    for evs in mv_iterator:
        yflipper.process_events(evs, evs_processed_buf)
        writer.add_cd_events(evs_processed_buf)
        if args.encode_triggers:
            writer.add_ext_trigger_events(
                mv_iterator.reader.get_ext_trigger_events())
            mv_iterator.reader.clear_ext_trigger_events()

    writer.flush()
    writer.close()
    print("Done!")


if __name__ == "__main__":
    main()
