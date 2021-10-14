# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Metavision RAW to CSV python sample.
"""


from metavision_core.event_io import EventsIterator
import os


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision RAW to CSV.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i',
                        '--input-raw-file',
                        dest='input_path',
                        required=True,
                        help="Path to input RAW file")
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    if not os.path.isfile(args.input_path):
        print('Fail to access RAW file ' + args.input_path)
        return

    # Events iterator on Camera or RAW file
    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=1000)

    with open('cd.csv', 'w') as csv_file:

        # Read formatted CD events from EventsIterator & write to CSV
        for evs in mv_iterator:
            for (x, y, p, t) in evs:
                csv_file.write("%d,%d,%d,%d\n" % (x, y, p, t))


if __name__ == "__main__":
    main()
