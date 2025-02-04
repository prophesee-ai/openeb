# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Metavision SDK Get Started.
"""


from metavision_sdk_stream import Camera, CameraStreamSlicer


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision SDK Get Started sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-event-file',
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.input_event_file:
        camera = Camera.from_file(args.input_event_file)
    else:
        camera = Camera.from_first_available()

    global_counter = 0  # This will track how many events we processed
    global_max_t = 0  # This will track the highest timestamp we processed

    slicer = CameraStreamSlicer(camera.move())
    for slice in slicer:
        print("----- New event slice! -----")
        if slice.events.size == 0:
            print("The current event slice is empty.")
        else:
            min_t = slice.events['t'][0]   # Get the timestamp of the first event of this slice
            max_t = slice.events['t'][-1]  # Get the timestamp of the last event of this callback
            global_max_t = max_t  # Events are ordered by timestamp, so the current last event has the highest timestamp

            counter = slice.events.size  # Local counter
            global_counter += counter  # Increase global counter

            print(f"There were {counter} events in this event slice.")
            print(f"There were {global_counter} total events up to now.")
            print(f"The current event slice included events from {min_t} to {max_t} microseconds.")
            print("----- End of the event slice! -----")


    # Print the global statistics
    duration_seconds = global_max_t / 1.0e6
    print(f"There were {global_counter} events in total.")
    print(f"The total duration was {duration_seconds:.2f} seconds.")
    if duration_seconds >= 1:  # No need to print this statistics if the total duration was too short
        print(f"There were {global_counter / duration_seconds :.2f} events per second on average.")

if __name__ == "__main__":
    main()
