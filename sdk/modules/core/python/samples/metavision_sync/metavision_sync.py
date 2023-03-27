# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Sample code that demonstrates how to use Metavision HAL Python API to synchronize two cameras.
"""

from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision camera synchronization sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-s', '--serial-number', dest='input_path', default="",
        help="Camera serial number. If not specified, the live stream of the first available camera is used.")

    parser.add_argument(
        '-m', '--mode-selection', dest='cam_mode', default='slave',
        help="Mode selected for camera: master or slave. If not specified, Camera will be in slave mode.")
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    # Creation of HAL device
    device = initiate_device(path=args.input_path)

    # then we use the facility i_device_control to set mode master/slave
    if device.get_i_camera_synchronization():
        if args.cam_mode == 'master':
            device.get_i_camera_synchronization().set_mode_master()
            print('Set mode master successful. Make sure to start slave camera first')
        else:
            device.get_i_camera_synchronization().set_mode_slave()
            print('Set mode slave successful. Start master camera to launch streaming ')

    # Events iterator on the device
    mv_iterator = EventsIterator.from_device(device=device)
    height, width = mv_iterator.get_size()  # Camera Geometry

    # Window - Graphical User Interface
    title = "Metavision Sync - Master" if args.cam_mode == 'master' else "Metavision Sync - Slave"
    with MTWindow(title=title, width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:
        def keyboard_cb(key, scancode, action, mods):
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        # Event Frame Generator
        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height, fps=25,
                                                           palette=ColorPalette.Dark)

        def on_cd_frame_cb(ts, cd_frame):
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        for evs in mv_iterator:
            # Dispatch system events to the window in order to catch keystrokes
            # This won't be available on the slave if the master is not streaming at the same time
            # (because in that case, the slave is in waiting mode with no event generated)
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)
            if window.should_close():
                break


if __name__ == "__main__":
    main()
