# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
A collection of utilities for searching image/video files.
"""
import os
import glob

VIDEO_EXTENSIONS = [".mp4", ".mov", ".m4v", ".avi"]
IMAGE_EXTENSIONS = [".jpg", ".png"]


def is_image(path):
    """Checks if a path is an image

    Args:
        path: file path
    Returns:
        is_image (bool): True or False
    """
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def is_video(path):
    """Checks if a path is a video

    Args:
        path: file path

    Returns:
        is_video (bool): True or False
    """
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def grab_images_and_videos(adir, recursive=True):
    """Grabs image and video files

    Args:
        adir: directory with images

    Returns:
        files: image and video files
    """
    assert os.path.isdir(adir)
    return grab_images(adir, recursive=recursive) + grab_videos(adir, recursive=recursive)


def grab_images(adir, recursive=True):
    """Grabs image files

    Args:
        adir: directory with images

    Returns:
        files: image files
    """
    assert os.path.isdir(adir)
    return grab_files(adir, IMAGE_EXTENSIONS, recursive=recursive)


def grab_h5s(adir, recursive=True):
    """Grabs h5 files

    Args:
        adir: directory with h5 files
        recursive (boolean): search recursively

    Returns:
        files: h5 files
    """
    return grab_files(adir, [".h5"], recursive=recursive)


def grab_jsons(adir, recursive=True):
    """Grabs json files

    Args:
        adir: directory with json files
        recursive (boolean): search recursively

    Returns:
        files: json files
    """
    return grab_files(adir, [".json"], recursive=recursive)


def grab_videos(adir, recursive=True):
    """Grabs videos in a directory

    Args:
        adir (str): directory
        recursive (boolean): search recursively

    Returns:
        files: files with image/ video extension
    """
    assert os.path.isdir(adir)
    files = grab_files(adir, VIDEO_EXTENSIONS, recursive=recursive)
    return files


def grab_files(adir, extensions, recursive=True):
    """Grabs files with allowed extensions

    Args:
        adir (str): directory
        extensions (list): allowed extensions
        recursive (boolean): search recursively

    Returns:
        files
    """
    assert os.path.isdir(adir)
    all_files = []
    search_string = adir + os.sep + "*"
    if recursive:
        search_string += "*" + os.sep + "*"
    for ext in extensions:
        all_files += glob.glob(search_string + ext, recursive=recursive)
        all_files += glob.glob(search_string + ext.upper(), recursive=recursive)
    all_files = list(set(all_files))  # handle operating systems which are not case sensitive...
    return all_files
