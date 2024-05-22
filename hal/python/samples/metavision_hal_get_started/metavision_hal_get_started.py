# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Sample code that shows how to use HAL Python API to stream from a live camera or a RAW event file
"""

from metavision_hal import DeviceDiscovery
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision HAL Get Started Sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="",
        help="Path to input event RAW file. If not specified, the camera live stream is used.")
    args = parser.parse_args()
    return args


def print_cd_events(event_buffer):
    if event_buffer.size > 0:
        print(f"New buffer of size {event_buffer.size} with timestamp range: ("
              f"{event_buffer[0]['t']},{event_buffer[-1]['t']})")


def main():
    """ Main """
    args = parse_args()
    if args.event_file_path:
        device = DeviceDiscovery.open_raw_file(args.event_file_path)
    else:
        device = DeviceDiscovery.open("")
    i_cddecoder = device.get_i_event_cd_decoder()
    i_cddecoder.add_event_buffer_callback(print_cd_events)
    i_eventsstreamdecoder = device.get_i_events_stream_decoder()
    i_eventsstream = device.get_i_events_stream()
    i_eventsstream.start()
    while True:
        try:
            ret = i_eventsstream.poll_buffer()
            if ret < 0:
                break
            elif ret > 0:
                raw_data = i_eventsstream.get_latest_raw_data()
                if raw_data is not None:
                    i_eventsstreamdecoder.decode(raw_data)
        except KeyboardInterrupt:
            break
    i_eventsstream.stop()


if __name__ == "__main__":
    main()
