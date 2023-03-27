# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Metavision RAW or DAT to CSV python sample.
"""

from metavision_core.event_io import EventsIterator
import os
from tqdm import tqdm


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision RAW or DAT to CSV.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-file', dest='input_path', required=True, help="Path to input RAW/DAT file")
    parser.add_argument('-o', '--output-dir', required=True, help="Path to csv output directory")
    parser.add_argument('-s', '--start-ts', type=int, default=0, help="start time in microsecond")
    parser.add_argument('-d', '--max-duration', type=int, default=1e6 * 60, help="maximum duration in microsecond")
    parser.add_argument('--delta-t', type=int, default=1000000, help="Duration of served event slice in us.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if os.path.isfile(args.input_path):
        output_file = os.path.join(args.output_dir, os.path.basename(args.input_path)[:-4] + ".csv")
    else:
        raise TypeError(f'Fail to access file: {args.input_path}')

    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=args.delta_t, start_ts=args.start_ts,
                                 max_duration=args.max_duration)

    with open(output_file, 'w') as csv_file:
        for evs in tqdm(mv_iterator, total=args.max_duration // args.delta_t):
            for (x, y, p, t) in evs:
                csv_file.write("%d,%d,%d,%d\n" % (x, y, p, t))


if __name__ == "__main__":
    main()
