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

    slicer = CameraStreamSlicer(camera.move())
    for _ in slicer:
        print("Events are available!")

if __name__ == "__main__":
    main()
