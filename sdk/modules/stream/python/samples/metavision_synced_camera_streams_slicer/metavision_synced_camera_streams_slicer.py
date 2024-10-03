# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
This application demonstrates how to use Metavision SDK Stream module to slice events from synchronized cameras
"""

import numpy as np
from pathlib import Path
from typing import Optional

from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_stream import SyncedCameraSystemBuilder, SyncedCameraStreamsSlicer, FileConfigHints, \
    SliceCondition
from metavision_sdk_ui import MTWindow, BaseWindow, EventLoop, UIAction, UIKeyEvent


# [CAMERA_VIEW_BEGIN]
class CameraView:
    """
    Class to display a camera's events slice in a window
    """

    def __init__(self, camera, name):
        """
        Constructor

        Args:
            camera: Camera to display
            name: Name of the window
        """
        width = camera.width()
        height = camera.height()
        self.frame = np.zeros((height, width, 3), np.uint8)
        self.window = MTWindow(name, width, height, BaseWindow.RenderMode.BGR, True)

        def keyboard_cb(key, scancode, action, mods):
            """
            Keyboard callback

            Args:
                key: Key pressed
                scancode: Scancode
                action: Action (press, release)
                mods: Mods (shift, ctrl, alt)
            """
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                self.window.set_close_flag()

        self.window.set_keyboard_callback(keyboard_cb)

    def process(self, events):
        """
        Generates a frame from the events and displays it in the window

        Args:
            events: Events to display
        """
        BaseFrameGenerationAlgorithm.generate_frame(events, self.frame)
        self.window.show_async(self.frame)


# [CAMERA_VIEW_END]

def parse_args():
    """
    Parse command line arguments
    """
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=("Code sample showing how to use the Metavision SyncedCameraStreamsSlicer to slice events "
                     "from a master and slave cameras system into fixed slices")
    )

    # Base options
    parser.add_argument(
        '-i', '--input-event-files', nargs='+', default=[],
        help="Paths to input event files (first is master). If not specified, the camera live streams are used.")
    parser.add_argument('-s', '--camera-serial-numbers', nargs='+', default=[],
                        help="Serial numbers of the cameras to be used (first is master)")
    parser.add_argument('-r', '--real-time-playback', action='store_true',
                        help="Flag to play records at recording speed")
    parser.add_argument('--record', type=bool, default=False,
                        help="Flag to record the streams")
    parser.add_argument('--record-path', type=str, default="",
                        help="Path to save the recorded streams")
    parser.add_argument('--config-path', type=str, default="",
                        help="Path to load the configuration files for each live camera")

    # Slicing options
    parser.add_argument('-m', '--slicing-mode', type=str,
                        choices=['N_EVENTS', 'N_US', 'MIXED'],
                        default='N_US', help="Slicing mode (i.e. N_EVENTS, N_US, MIXED)")
    parser.add_argument('-t', '--delta-ts', type=int, default=10000,
                        help="Slice duration in microseconds (default=10000us)")
    parser.add_argument('-n', '--delta-n-events', type=int, default=100000,
                        help="Number of events in a slice (default=100000)")

    args = parser.parse_args()

    if args.slicing_mode == 'IDENTITY':
        args.slice_condition = SliceCondition.make_identity()
    elif args.slicing_mode == 'N_EVENTS':
        args.slice_condition = SliceCondition.make_n_events(args.delta_n_events)
    elif args.slicing_mode == 'N_US':
        args.slice_condition = SliceCondition.make_n_us(args.delta_ts)
    elif args.slicing_mode == 'MIXED':
        args.slice_condition = SliceCondition.make_mixed(args.delta_ts, args.delta_n_events)
    else:
        raise ValueError(f"Invalid slicing mode: {args.slicing_mode}")

    return args


def build_slicer(args):
    """
    Build the SyncedCameraStreamsSlicer from the command line arguments

    Args:
        args: Command line arguments

    Returns: The SyncedCameraStreamsSlicer instance
    """
    # [BUILD_CAMERA_SYSTEM_BEGIN]
    builder = SyncedCameraSystemBuilder()

    def get_settings_file_path(config_dir, serial_number) -> Optional[Path]:
        settings_file_path = Path(config_dir) / f"{serial_number}.json"
        if not settings_file_path.exists():
            return None
        return settings_file_path

    for sn in args.camera_serial_numbers:
        print(f"Adding camera with serial number {sn}")
        settings_file_path = get_settings_file_path(args.config_path, sn)
        builder.add_live_camera_parameters(serial_number=sn, settings_file_path=settings_file_path)

    builder.set_record(args.record)
    builder.set_record_dir(args.record_path)

    for record in args.input_event_files:
        builder.add_record_path(record)

    hints = FileConfigHints()
    hints.real_time_playback(args.real_time_playback)

    builder.set_file_config_hints(hints)

    [master, slaves] = builder.build()
    # [BUILD_CAMERA_SYSTEM_END]
    return SyncedCameraStreamsSlicer(master.move(), [slave.move() for slave in slaves], args.slice_condition)


def build_views(slicer):
    """
    Build the CameraView instances from the SyncedCameraStreamsSlicer

    Args:
        slicer: The SyncedCameraStreamsSlicer instance
    """
    # [BUILD_VIEWS_BEGIN]
    views = [CameraView(slicer.master(), "Master")]

    for i in range(slicer.slaves_count()):
        views.append(CameraView(slicer.slave(i), f"Slave {i}"))

    # [BUILD_VIEWS_END]
    return views


def should_exit(views):
    """
    Check if the program should exit. It happens if one of the windows is closed

    Args:
        views: The CameraView instances

    Returns: True if the program should exit
    """
    for view in views:
        if view.window.should_close():
            return True
    return False


def log_slice_info(slice):
    """
    Log information about the slice

    Args:
        slice: The synchronized slice of events to log
    """
    print(f"===== Slice =====")
    print(f"ts: {slice.t}")
    print(f"Master events: {slice.n_events}")
    for i, slave_slice in enumerate(slice.slave_events):
        print(f"Slave {i + 1} events: {len(slave_slice)}")
    print("=================\n")


def main():
    args = parse_args()
    slicer = build_slicer(args)
    views = build_views(slicer)

    # [SLICER_LOOP_BEGIN]
    for slice in slicer:
        EventLoop.poll_and_dispatch()

        log_slice_info(slice)

        views[0].process(slice.master_events)

        for i in range(slicer.slaves_count()):
            views[i + 1].process(slice.slave_events[i])

        if should_exit(views):
            break

    # [SLICER_LOOP_END]

    for view in views:
        view.window.destroy()


if __name__ == "__main__":
    main()
