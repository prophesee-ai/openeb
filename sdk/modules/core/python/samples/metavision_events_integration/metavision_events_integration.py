# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Example of using Metavision SDK Core API for integrating events in a simple way into a grayscale-like image
"""

import numpy as np
import cv2
from skvideo.io import FFmpegWriter
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import ContrastMapGenerationAlgorithm, EventsIntegrationAlgorithm, OnDemandFrameGenerationAlgorithm
from metavision_sdk_ui import MTWindow, BaseWindow, EventLoop, UIAction, UIKeyEvent


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision Events Integration sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-file', dest='in_file_path', default="",
                        help="Path to input file. If not specified, the camera live stream is used.")
    parser.add_argument('-p', '--period', dest='period_us', type=int, default=30000,
                        help="Period for the generation of the integrated event frames, in us.")
    parser.add_argument(
        '-d', '--decay-time', dest='decay_time', type=int, default=100000,
        help="Decay time after which integrated frame tends back to neutral gray. This needs to be adapted to the scene dynamics.")
    parser.add_argument('-r', '--blur-radius', dest='integration_blur_radius', type=int, default=1,
                        help="Gaussian blur radius to be used to smooth integrated intensities.")
    parser.add_argument(
        '-w', '--diffusion-weight', dest='diffusion_weight', type=float, default=0,
        help="Weight used to diffuse neighboring intensities into each other to slowly smooth the image. Disabled if zero, cannot exceed 0.25f.")
    parser.add_argument('-c', '--contrast-on', dest='contrast_on', type=float,
                        default=1.2, help="Contrast associated to ON events.")
    parser.add_argument('--contrast-off', dest='contrast_off', type=float, default=-1,
                        help="Contrast associated to OFF events. If negative, the inverse of contrast-on is used.")
    parser.add_argument(
        '--tonemapping-count', dest='tonemapping_max_ev_count', type=float, default=5,
        help="Maximum event count to tonemap in 8-bit grayscale frame. This needs to be adapted to the scene dynamic range & sensor sensitivity.")
    parser.add_argument('-o', '--output-video', dest='output_video_path', type=str,
                        default="", help="Save display window in a .avi format")
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    print("Code sample showing how to integrate events in a simple way to reconstruct grayscale images.")

    # Events iterator on Camera or event file
    mv_iterator = EventsIterator(input_path=args.in_file_path, delta_t=args.period_us)
    if not is_live_camera(args.in_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)
    height, width = mv_iterator.get_size()  # Camera Geometry

    # Instantiate Event Frame Generator
    cmap_gen = ContrastMapGenerationAlgorithm(width, height, args.contrast_on, args.contrast_off)
    ev_int = EventsIntegrationAlgorithm(width, height, args.decay_time, args.contrast_on, args.contrast_off,
                                        args.tonemapping_max_ev_count, args.integration_blur_radius, args.diffusion_weight)
    frame_gen = OnDemandFrameGenerationAlgorithm(width, height, accumulation_time_us=args.period_us)

    show_cmap = False
    tonemapping_factor = 1 / (args.contrast_on ** (args.tonemapping_max_ev_count - 1) - 1)
    evs_frame_c3 = np.zeros((height, width, 3), dtype=np.uint8)
    int_frame_c1 = np.zeros((height, width), dtype=np.uint8)
    int_frame_c3 = np.zeros((height, width, 3), dtype=np.uint8)

    # Video writer
    if args.output_video_path:
        video_name = args.output_video_path + ".avi"
        video_writer = FFmpegWriter(video_name)

    # Window - Graphical User Interface (Display filtered events and process keyboard events)
    with MTWindow(title="Metavision Event Integration", width=width, height=2*height, mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()
            elif key == UIKeyEvent.KEY_T and action == UIAction.RELEASE:
                nonlocal show_cmap
                show_cmap = not show_cmap
        window.set_keyboard_callback(keyboard_cb)
        print("Press 'ESC' to quit, 'T' to toggle between integrated events and contrast map display.")

        # Processing loop
        for evs in mv_iterator:
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()
            if window.should_close():
                break
            # Process events
            if evs.size != 0:
                ts = mv_iterator.get_current_time()
                cmap_gen.process_events(evs)
                ev_int.process_events(evs)
                frame_gen.process_events(evs)

                frame_gen.generate(ts, evs_frame_c3)
                if show_cmap:
                    cmap_gen.generate(int_frame_c1, 128 * tonemapping_factor, 128 * (1 - tonemapping_factor))
                else:
                    ev_int.generate(int_frame_c1)
                int_frame_c3[:, :, 0] = int_frame_c1
                int_frame_c3[:, :, 1] = int_frame_c1
                int_frame_c3[:, :, 2] = int_frame_c1
                output_img_bgr = np.vstack((evs_frame_c3, int_frame_c3))
                window.show_async(output_img_bgr)
                if args.output_video_path:
                    output_img_rgb = cv2.cvtColor(output_img_bgr, cv2.COLOR_BGR2RGB)
                    video_writer.writeFrame(output_img_rgb)

        if args.output_video_path:
            video_writer.close()
            print("Video has been saved in " + video_name)


if __name__ == "__main__":
    main()
