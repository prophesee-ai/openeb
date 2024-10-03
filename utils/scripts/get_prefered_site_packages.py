#!/usr/bin/env python3

# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

import argparse
import functools
import site

def parse_args():
    """Utility function to define and input arguments

    Returns:
        argparse.ArgumentParser: to parse input arguments
    """

    parser = argparse.ArgumentParser(
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--site-type', type=str, required=True, choices=['system', 'local'],
                        help='Specifies the type of desired python site packages')
    parser.add_argument('--prefer-pattern', action='append', default=[],
                        help='Provide additional patterns to look for in candidate site packages paths')
    return parser.parse_args()


# select a suitable python module path by following ranking (i.e a path that meets
# criteria 1. is better than 2., and a path that meets criteria 1.+2. is better than 1.+3.)
# 1-a path that does not contain 'local'
# 2-a path that contains 'dist-packages'
# 3-a path that contains 'site-packages'
# 1-a path that does not contain '.local'
# 2-a path that does not contain 'home'
# 3-a path that does not contain '/opt/'
# 4-a path that does not contain '/local/'
# 5-a path that contains 'dist-packages'
# 6-a path that contains 'site-packages'
def system_site_score(site_path):
    return (('.local' not in site_path) * 100000.0 +
            ('home' not in site_path) * 10000.0 +
            ('/opt/' not in site_path) * 1000.0 +
            ('/local/' not in site_path) * 100.0 +
            ('dist-packages' in site_path) * 10.0 +
            ('site-packages' in site_path) * 1.0)


# select a suitable python module path by following ranking (i.e a path that meets
# criteria 1. is better than 2., and a path that meets criteria 1.+2.)
# 1-the cmake install prefix
# 2-a path that contains 'dist-packages'
def local_site_score(site_path):
    return (('dist-packages' in site_path) * 10.0 +
            ('site-packages' in site_path) * 1.0)


def prefered_patterns_score(site_path, prefer_patterns):
    return functools.reduce(lambda score, pattern: score + (1000000.0 if pattern in site_path else 0.0),
                            prefer_patterns, 0.0)


def get_sorted_site_packages(site_score, prefer_patterns):
    return sorted(site.getsitepackages(),
                  key=lambda path: prefered_patterns_score(path, prefer_patterns) + site_score(path),
                  reverse=True)


if __name__ == '__main__':
    args = parse_args()
    if args.site_type == 'system':
        site_score = system_site_score
    else:
        site_score = local_site_score
    print(get_sorted_site_packages(site_score, args.prefer_pattern)[0])
