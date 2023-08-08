# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Code sample showing how to create a simple application to filter and display events.
"""

from enum import Enum
from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, PolarityFilterAlgorithm, RoiFilterAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent


roi_crop_width = 150


class Polarity(Enum):
    ALL = -1,
    OFF = 0,
    ON = 1


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision Filtering sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-raw-file', dest='input_path', default="",
        help="Path to input RAW file. If not specified, the live stream of the first available camera is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    parser.add_argument(
        '-r', '--replay_factor', type=float, default=1,
        help="Replay Factor. If greater than 1.0 we replay with slow-motion, otherwise this is a speed-up over real-time.")
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    print("Code sample showing how to create a simple application to filter and display events.")
    print("Available keyboard options:\n"
          "  - R: Toggle the ROI filter algorithm\n"
          "  - P: Show only events of positive polarity\n"
          "  - N: Show only events of negative polarity\n"
          "  - A: Show all events\n"
          "  - Q/Escape: Quit the application\n")

    # Events iterator on Camera or RAW file
    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=1000)
    if args.replay_factor > 0 and not is_live_camera(args.input_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)
    height, width = mv_iterator.get_size()  # Camera Geometry

    polarity_filters = {Polarity.OFF: PolarityFilterAlgorithm(0), Polarity.ON: PolarityFilterAlgorithm(1)}
    roi_filter = RoiFilterAlgorithm(x0=roi_crop_width, y0=roi_crop_width,
                                    x1=width - roi_crop_width, y1=height - roi_crop_width)
    events_buf = RoiFilterAlgorithm.get_empty_output_buffer()
    use_roi_filter = False
    polarity = Polarity.ALL

    # Event Frame Generator
    event_frame_gen = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us=10000)

    # Window - Graphical User Interface (Display filtered events and process keyboard events)
    with MTWindow(title="Metavision Filtering", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        def on_cd_frame_cb(ts, cd_frame):
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        def keyboard_cb(key, scancode, action, mods):
            nonlocal use_roi_filter
            nonlocal polarity

            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()
            elif key == UIKeyEvent.KEY_A:
                # Show all events
                polarity = Polarity.ALL
            elif key == UIKeyEvent.KEY_N:
                # Show only negative events
                polarity = Polarity.OFF
            elif key == UIKeyEvent.KEY_P:
                # Show only positive events
                polarity = Polarity.ON
            elif key == UIKeyEvent.KEY_R:
                # Toggle ROI filter
                use_roi_filter = not use_roi_filter

        window.set_keyboard_callback(keyboard_cb)

        # Process events
        for evs in mv_iterator:
            if use_roi_filter:
                roi_filter.process_events(evs, events_buf)
                if polarity in polarity_filters:
                    polarity_filters[polarity].process_events_(events_buf)
                event_frame_gen.process_events(events_buf)
            elif polarity in polarity_filters:
                polarity_filters[polarity].process_events(evs, events_buf)
                event_frame_gen.process_events(events_buf)
            else:
                event_frame_gen.process_events(evs)

            if window.should_close():
                break


if __name__ == "__main__":
    main()
