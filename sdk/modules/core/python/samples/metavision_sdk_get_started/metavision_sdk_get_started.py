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


from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIAction, UIKeyEvent

accumulation_time_us = 10000


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision SDK Get Started sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-raw-file', dest='input_path', default="",
        help="Path to input RAW file. If not specified, the live stream of the first available camera is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    # Events iterator on Camera or RAW file
    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=1000)
    height, width = mv_iterator.get_size()  # Camera Geometry

    # Window - Graphical User Interface
    with Window(title="Metavision SDK Get Started", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        def keyboard_cb(key, scancode, action, mods):
            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        # Event Frame Generator
        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height,
                                                           accumulation_time_us=accumulation_time_us)

        def on_cd_frame_cb(ts, cd_frame):
            window.show(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        global_counter = 0  # This will track how many events we processed
        global_max_t = 0  # This will track the highest timestamp we processed

        # Process events
        for evs in mv_iterator:
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()

            event_frame_gen.process_events(evs)

            print("----- New event buffer! -----")
            if evs.size == 0:
                print("The current event buffer is empty.")
            else:
                min_t = evs['t'][0]   # Get the timestamp of the first event of this callback
                max_t = evs['t'][-1]  # Get the timestamp of the last event of this callback
                global_max_t = max_t  # Events are ordered by timestamp, so the current last event has the highest timestamp

                counter = evs.size  # Local counter
                global_counter += counter  # Increase global counter

                print(f"There were {counter} events in this event buffer.")
                print(f"There were {global_counter} total events up to now.")
                print(f"The current event buffer included events from {min_t} to {max_t} microseconds.")
                print("----- End of the event buffer! -----")

            if window.should_close():
                break

        # Print the global statistics
        duration_seconds = global_max_t / 1.0e6
        print(f"There were {global_counter} events in total.")
        print(f"The total duration was {duration_seconds:.2f} seconds.")
        if duration_seconds >= 1:  # No need to print this statistics if the video was too short
            print(f"There were {global_counter / duration_seconds :.2f} events per second on average.")


if __name__ == "__main__":
    main()
