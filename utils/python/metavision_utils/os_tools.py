#!/usr/bin/env python

# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Module with utility classes and functions to handle os related commmands
"""

import os
import datetime
import tempfile
import shutil
import platform
import math


def shorten_path(path, length=80):
    """Shortens path"""
    path = path.rstrip('/')
    if len(path) <= length:
        return path
    if length < 7:
        return path[-length:]

    num = math.floor((length - 5) / 2)
    return '{}/.../{}'.format(path[:num], path[-num:])


def which(program):
    """Tests if an executable program exists"""
    def is_exe(fpath):
        """Returns True if path is an executable"""
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def get_parent_directory(filepath, level=1):
    """Return parent directory of a file

    Args:
        filepath (str): path of the file
        level (int): parent directory level. 1 means parent directory, 2 is the parent directory of the parent
          directory, etc ... Defaults to 1

    Returns:
        str : full path to the directory asked
    """
    fullpath = os.path.abspath(filepath)
    dirname = os.path.dirname(fullpath)
    for _ in range(1, level):
        dirname = os.path.dirname(dirname)
    return dirname


def get_date():
    """Returns date"""
    return datetime.datetime.now().strftime("%Y-%m-%d")


def write_temp_and_get_filepath(content, **kwargs):
    """Write contents in temporary file and returns its path"""
    tmpf = tempfile.NamedTemporaryFile(
        'w', delete=False, **kwargs)
    tmpf.write(content)
    return tmpf.name


def is_python_script(filepath):
    """Returns True if a file is a python executable."""
    if os.path.isdir(filepath):
        return False
    if not os.access(filepath, os.X_OK):
        return False
    try:
        with open(filepath, 'r') as tmpf:
            return has_python_content(tmpf.read())
    except UnicodeDecodeError:
        return False


def has_python_extension(filename):
    """Returns true if filename is recognized as python file."""
    return filename.endswith('.py')


def has_python_content(content):
    """A function to check if some string contains python code."""
    first_line = content.split('\n')[0].strip()
    return "#!" in first_line and "python" in first_line


def has_cpp_extension(filename):
    """Returns true if filename is recognized as C++ file."""
    for ext in ['.c', '.cpp', '.cxx', '.cc', '.h', '.hpp']:
        if filename.endswith(ext):
            return True
    return False


def get_environment_with_added_paths(paths_to_add):
    """
    A function that will return an environment with paths_to_add added to PATH
    """
    env = os.environ

    for elt in paths_to_add:
        env["PATH"] += os.pathsep + os.path.join(elt)

    return env


def create_directory(directory_path, **kwargs):
    """Recursive directory creation function. If directory already exists, does nothing

    Args:
        directory_path (str): Path to the directory to create. Creates all intermediate-level
            directories needed to contain the leaf directory.
        **kwargs: Arbitrary keyword arguments.
            Allowed keys are :
                dry_run (bool): If True just prints what will be done, without actually doing it.
                    Defaults to False.
                verbose (bool): If True, prints both what will be done and its output
                    Defaults to False.

    Returns:
        bool : True if directory was successfully created, False otherwise. Remark that in case
            of a dry-run, it will return True.

    """

    dry_run = kwargs.pop('dry_run', False)
    verbose = kwargs.pop('verbose', False)

    if dry_run or verbose:
        print("Create directory '{}'".format(directory_path))
        if dry_run:
            return True

    if os.path.isdir(directory_path):
        if verbose:
            print("Nothing to do : directory '{}' already exists.".format(directory_path))
        return True

    try:
        os.makedirs(directory_path)
    except BaseException as exc:
        if verbose:
            print("Error occurred while creating directory '{}' :".format(directory_path))
            print(exc)
        return False

    if verbose:
        print("Directory '{}' created".format(directory_path))

    return True


def copy_directory(dir_to_copy_path, destination_directory, **kwargs):
    """Copy directory

    Args:
        dir_to_copy_path (str): Path to the directory to copy
        destination_directory (str): Path to the directory destination. If directory does not exist, it will be created
        **kwargs: Arbitrary keyword arguments.
            Allowed keys are :
                dry_run (bool): If True just prints what will be done, without actually doing it.
                    Defaults to False.
                verbose (bool): If True, prints both what will be done and its output
                    Defaults to False.
                rename (str): If provided, rename directory in destination directory
                ignore_patterns (list of str): list of patterns to ignore

    Returns:
        bool : True if file was successfully copied, False otherwise. Remark that in case
               of a dry-run, it will return True.

    """
    dry_run = kwargs.pop('dry_run', False)
    verbose = kwargs.pop('verbose', False)
    rename = kwargs.pop('rename', None)
    ignore_patterns = tuple(kwargs.pop('ignore_patterns', []))

    if (not os.path.isdir(dir_to_copy_path)) and (not dry_run):
        print("ERROR : directory '{}' does not exist".format(dir_to_copy_path))
        return False

    destination = os.path.join(destination_directory, os.path.basename(dir_to_copy_path))
    if rename:
        destination = os.path.join(destination_directory, rename)

    if dry_run or verbose:
        print("cp -R {} {}".format(dir_to_copy_path, destination))
        if dry_run:
            return True

    try:
        shutil.copytree(dir_to_copy_path, destination, ignore=shutil.ignore_patterns(*ignore_patterns))
    except BaseException as exc:
        if verbose:
            print("Error occurred :")
            print(exc)
        return False

    return True


def remove_directory(directory_path, **kwargs):
    """Remove directory

    Args:
        directory_path (str): Path to the directory to remove.
        **kwargs: Arbitrary keyword arguments.
            Allowed keys are :
                dry_run (bool): If True just prints what will be done, without actually doing it.
                    Defaults to False.
                verbose (bool): If True, prints both what will be done and its output
                    Defaults to False.

    Returns:
        bool : True if directory was successfully removed, False otherwise. Remark that in case
               of a dry-run, it will return True.


    """

    dry_run = kwargs.pop('dry_run', False)
    verbose = kwargs.pop('verbose', False)

    if not os.path.isdir(directory_path):
        if verbose:
            print("Nothing to do : directory '{}' does not exist.".format(directory_path))
        return True

    if dry_run or verbose:
        print("rm -r '{}'".format(directory_path))
        if dry_run:
            return True

    try:
        shutil.rmtree(directory_path)
    except BaseException as exc:
        if verbose:
            print("Error occurred while removing directory '{}' :".format(directory_path))
            print(exc)
        return False

    return True


def remove_file(filepath, **kwargs):
    """Remove directory

    Args:
        filepath (str): Path to the file to remove.
        **kwargs: Arbitrary keyword arguments.
            Allowed keys are :
                dry_run (bool): If True just prints what will be done, without actually doing it.
                    Defaults to False.
                verbose (bool): If True, prints both what will be done and its output
                    Defaults to False.

    Returns:
        bool : True if directory was successfully removed, False otherwise. Remark that in case
               of a dry-run, it will return True.


    """

    dry_run = kwargs.pop('dry_run', False)
    verbose = kwargs.pop('verbose', False)

    if not os.path.isfile(filepath):
        if verbose:
            print("Nothing to do : file '{}' does not exist.".format(filepath))
        return True

    if dry_run or verbose:
        print("rm '{}'".format(filepath))
        if dry_run:
            return True

    try:
        os.remove(filepath)
    except BaseException as exc:
        if verbose:
            print("Error occurred while removing file '{}' :".format(filepath))
            print(exc)
        return False

    return True


def copy_file_in_directory(file_to_copy_path, directory, **kwargs):
    """Copy file in directory

    Args:
        file_to_copy_path (str): Path to the file to copy
        directory (str): Path to the directory destination. If directory does not exist, it will be created
        **kwargs: Arbitrary keyword arguments.
            Allowed keys are :
                dry_run (bool): If True just prints what will be done, without actually doing it.
                    Defaults to False.
                verbose (bool): If True, prints both what will be done and its output
                    Defaults to False.
                rename (str): If provided, rename file in destination directory

    Returns:
        bool : True if file was successfully copied, False otherwise. Remark that in case
               of a dry-run, it will return True.

    """
    dry_run = kwargs.pop('dry_run', False)
    verbose = kwargs.pop('verbose', False)
    rename = kwargs.pop('rename', None)

    if not os.path.isdir(directory):
        if not create_directory(directory, dry_run=dry_run, verbose=verbose):
            return False

    destination = directory
    if rename:
        destination = os.path.join(directory, rename)

    if dry_run or verbose:
        print("cp {} {}".format(file_to_copy_path, destination))
        if dry_run:
            return True
    try:
        shutil.copy(file_to_copy_path, destination)
    except BaseException as exc:
        if verbose:
            print("Error occurred :")
            print(exc)
        return False

    return True


def move_file(file_to_move_path, directory_destination_path, **kwargs):
    """Move file in directory

    Args:
        file_to_move_path (str): Path to the file to move
        directory_destination_path (str): Path to the directory destination. If directory does not exist, it will be
            created
        **kwargs: Arbitrary keyword arguments.
            Allowed keys are :
                dry_run (bool): If True just prints what will be done, without actually doing it.
                    Defaults to False.
                verbose (bool): If True, prints both what will be done and its output
                    Defaults to False.
                rename (str): If provided, rename file in destination directory

    Returns:
        bool : True if file was successfully copied, False otherwise. Remark that in case
               of a dry-run, it will return True.

    """
    dry_run = kwargs.pop('dry_run', False)
    verbose = kwargs.pop('verbose', False)
    rename = kwargs.pop('rename', None)

    if not os.path.isdir(directory_destination_path):
        if not create_directory(directory_destination_path, dry_run=dry_run, verbose=verbose):
            return False

    destination = directory_destination_path
    if rename:
        destination = os.path.join(directory_destination_path, os.path.basename(file_to_move_path))

    if dry_run or verbose:
        print("mv {} {}".format(file_to_move_path, destination))
        if dry_run:
            return True
    try:
        shutil.move(file_to_move_path, destination)
    except BaseException as exc:
        if verbose:
            print("Error occurred :")
            print(exc)
        return False

    return True


def write_text_in_file(text, filepath, **kwargs):
    """Writes some text in a file

    Args:
        text (str): Text to write
        filepath (str): File to write. If the directory of the file does not exist, it will be created.
        **kwargs: Arbitrary keyword arguments.
            Allowed keys are :
                dry_run (bool): If True just prints what will be done, without actually doing it.
                    Defaults to False.
                verbose (bool): If True, prints both what will be done and its output
                    Defaults to False.
                append (bool): If True, append text in file rather than erasing previous contents of it.
                    Defaults to False.

    Returns:
        bool : True if file the operation was successful, False otherwise.
    """

    dry_run = kwargs.pop('dry_run', False)
    verbose = kwargs.pop('verbose', False)
    append = kwargs.pop('append', False)

    if dry_run or verbose:
        print("Write in file '{}' (mode : {})".format(filepath, "append" if append else "write"))

        if dry_run:
            return True

    if not create_directory(os.path.dirname(filepath), dry_run=dry_run, verbose=verbose):
        return False

    try:
        mode = 'a' if append else 'w'
        with open(filepath, mode) as file_opened:
            file_opened.write(text)
    except IOError as exc:
        if verbose:
            print("Error occurred :")
            print(exc)
        return False

    if verbose:
        print("File '{}' written".format(filepath))

    return True


class TemporaryDirectoryHandler(object):
    """Utility class to handle the creation of a temporary directory"""

    def __init__(
            self,
            create_dir_right_away=True,
            verbose_in_dtor=False,
            tmp_dir_root=None,
            tmp_dir_root_in_container=None):
        """Constructor

        Args:
            create_dir_right_away (bool): if True, a temporary directory will be created. If False, you need to call
                method call create_temporary_directory() in order to create the temporary directory
            verbose_in_dtor (bool): if True, the directory removal in the destructor will be with verbose print
                                    (for debug purposes)

        Raises:
            RuntimeError : if error occurred while trying to create temporary directory
        """
        self.__temporary_directory = None
        self.__temporary_directory_created = False
        self.__verbose_in_dtor = verbose_in_dtor
        self.__tmp_dir_root = tmp_dir_root or os.path.join(os.getcwd(), 'tmp')
        self.__tmp_dir_root_in_container = tmp_dir_root_in_container

        if not tmp_dir_root_in_container:
            self.__tmp_dir_root_in_container = tmp_dir_root

        if create_dir_right_away and not self.create_temporary_directory(tmp_dir_root=tmp_dir_root):
            raise RuntimeError("Could not create temporary directory")

    def __del__(self):
        """Destructor

        Erases temporary directory if it has not been done yet.
        """
        self.remove_temporary_directory(verobse=self.__verbose_in_dtor)

    def temporary_directory_root(self):
        """Returns tmp_dir_root"""
        return self.__tmp_dir_root

    def temporary_directory_root_in_container(self):
        """Returns tmp_dir_root_in_container"""
        return self.__tmp_dir_root_in_container

    def temporary_directory(self):
        """Returns temporary directory created"""
        return self.__temporary_directory

    def create_temporary_directory(self, **kwargs):
        """Creates temporary directory

        Args:
            **kwargs: Arbitrary keyword arguments.
                Allowed keys are :
                    dry_run (bool): If True just prints what will be done, without actually doing it.
                        Defaults to False.
                    verbose (bool): If True, prints both what will be done and its output
                        Defaults to False.

        Returns:
            bool : True if directory was successfully created, False otherwise. Remark that in case
                of a dry-run, it will return True.

        """

        dry_run = kwargs.pop('dry_run', False)
        verbose = kwargs.pop('verbose', False)

        if dry_run or verbose:
            print("Create temporary directory")  # FIX THIS
            if dry_run:
                if not self.__temporary_directory_created:  # not to overwrite the value of self.__temporary_directory
                    self.__temporary_directory = os.path.join(self.__tmp_dir_root, 'tmp_dir')
                return True

        if self.__temporary_directory_created:
            if verbose:
                print("Temporary directory '{}' already created.".format(self.__temporary_directory))
            return True

        try:
            create_directory(self.__tmp_dir_root)
            self.__temporary_directory = tempfile.mkdtemp(prefix=os.path.join(self.__tmp_dir_root, 'tmp_dir_'))
            self.__temporary_directory_created = True
        except BaseException as exc:
            if verbose:
                print("Error occurred :")
                print(exc)
            return False

        if verbose:
            print("Temporary directory '{}' created".format(self.__temporary_directory))

        return True

    def remove_temporary_directory(self, **kwargs):
        """Removes temporary directory, if created.

        Args:
            **kwargs: Arbitrary keyword arguments.
                Allowed keys are :
                    dry_run (bool): If True just prints what will be done, without actually doing it.
                        Defaults to False.
                    verbose (bool): If True, prints both what will be done and its output
                        Defaults to False.

        Returns:
            bool : True if directory was successfully removed (or if it was not created at all). False otherwise.
                Remark that in case of a dry-run, it will return True.

        """
        dry_run = kwargs.pop('dry_run', False)
        verbose = kwargs.pop('verbose', False)

        if dry_run or verbose:
            print("Remove temporary directory")
            if dry_run:
                if not self.__temporary_directory_created:  # not to overwrite the value of self.__temporary_directory
                    self.__temporary_directory = None
                return True

        if not self.__temporary_directory_created:
            if verbose:
                print("Temporary directory '{}' was not created.")
            return True

        if self.__temporary_directory:

            tmp_dir = self.__temporary_directory

            if remove_directory(self.__temporary_directory, dry_run=dry_run, verbose=verbose):
                self.__temporary_directory_created = False
                self.__temporary_directory = None
            else:
                return False

            if verbose:
                print("Temporary directory '{}' removed".format(tmp_dir))
            self.__temporary_directory = None
        return True

    def write_file_in_tmp_dir(self, file_basename, text, **kwargs):
        """Writes file in the temporary directory.

        If the temporary directory has not be created yet, it will be.

        Args:
            file_basename (str): Basename of the file to create (or write into if already created).
            text (str): Text to write.
            **kwargs: Arbitrary keyword arguments.
                Allowed keys are :
                    dry_run (bool): If True just prints what will be done, without actually doing it.
                        Defaults to False.
                    verbose (bool): If True, prints both what will be done and its output
                        Defaults to False.
                    append (bool): If True, append text in file rather than erasing previous contents of it.
                        Defaults to False.

        Returns :
            str : the full path of the file created. If an error occurred, returns None

        """

        dry_run = kwargs.pop('dry_run', False)
        verbose = kwargs.pop('verbose', False)
        append = kwargs.pop('append', False)

        if (not self.__temporary_directory_created) and (not dry_run):
            if not self.create_temporary_directory(dry_run=dry_run, verbose=verbose):
                return None

        file_fullpath = os.path.join(self.__temporary_directory, file_basename)

        if not write_text_in_file(text, file_fullpath, append=append, dry_run=dry_run, verbose=verbose):
            return None

        if verbose:
            print("File '{}' written".format(file_fullpath))

        return file_fullpath

    def copy_file_in_tmp_dir(self, file_to_copy, **kwargs):
        """Copy file in the temporary directory.

        If the temporary directory has not be created yet, it will be.

        Args:
            file_to_copy (str): File to copy
            **kwargs: Arbitrary keyword arguments.
                Allowed keys are :
                    dry_run (bool): If True just prints what will be done, without actually doing it.
                        Defaults to False.
                    verbose (bool): If True, prints both what will be done and its output
                        Defaults to False.
                    command_executer (function): a way to customise what/who executes a command

        Returns :
            str : the full path of destination file. If an error occurred, returns None

        """

        dry_run = kwargs.pop('dry_run', False)
        verbose = kwargs.pop('verbose', False)
        command_exec = kwargs.pop('command_executer', None)

        if (not self.__temporary_directory_created) and (not dry_run):
            if not self.create_temporary_directory(dry_run=dry_run, verbose=verbose):
                return None

        if (not command_exec) or platform.system() == 'Windows':
            if not copy_file_in_directory(file_to_copy, self.__temporary_directory, dry_run=dry_run, verbose=verbose):
                return None
        else:
            dest_dir = os.path.join(self.__tmp_dir_root_in_container, os.path.basename(self.__temporary_directory))
            cmd = "bash -c " + '"cp {} {}"'.format(file_to_copy, dest_dir)
            if not command_exec(
                    cmd,
                    verbose=verbose,
                    dry_run=dry_run):
                return None

        return os.path.join(self.__temporary_directory, os.path.basename(file_to_copy))

    def copy_directory_in_tmp_dir(self, source_directory, **kwargs):
        """Recursively copy directory in the temporary directory.

        If the temporary directory has not be created yet, it will be.

        Args:
            source_directory (str): Path to the directory to copy. A directory with the same basename
                will be created in the temporary directory.
            **kwargs: Arbitrary keyword arguments.
                Allowed keys are :
                    dry_run (bool): If True just prints what will be done, without actually doing it.
                        Defaults to False.
                    verbose (bool): If True, prints both what will be done and its output
                        Defaults to False.
                    command_executer (function): a way to customise what/who executes a command

        Returns :
            str : the full path of destination directory. If an error occurred, returns None

        """

        dry_run = kwargs.pop('dry_run', False)
        verbose = kwargs.pop('verbose', False)
        command_exec = kwargs.pop('command_executer', None)

        if (not self.__temporary_directory_created) and (not dry_run):
            if not self.create_temporary_directory(dry_run=dry_run, verbose=verbose):
                return None

        if (not command_exec) or platform.system() == 'Windows':
            if not copy_directory(source_directory, self.__temporary_directory, dry_run=dry_run, verbose=verbose):
                return None
        else:
            dest_dir = os.path.join(self.__tmp_dir_root_in_container, os.path.basename(self.__temporary_directory))
            cmd = "bash -c " + '"cp -r {} {}"'.format(source_directory, dest_dir)
            if not command_exec(
                    cmd,
                    verbose=verbose,
                    dry_run=dry_run):
                return None

        destination_directory = os.path.join(self.__temporary_directory,
                                             os.path.basename(os.path.basename(source_directory)))
        return destination_directory

    def create_directory_in_tmp_dir(self, directory_basename, **kwargs):
        """Creates directory in the temporary directory.

        If the temporary directory has not be created yet, it will be.

        Args:
            directory_basename (str): Basename of the directory to create.
            **kwargs: Arbitrary keyword arguments.
                Allowed keys are :
                    dry_run (bool): If True just prints what will be done, without actually doing it.
                        Defaults to False.
                    verbose (bool): If True, prints both what will be done and its output
                        Defaults to False.

        Returns :
            str : the full path of created directory. If an error occurred, returns None

        """

        dry_run = kwargs.pop('dry_run', False)
        verbose = kwargs.pop('verbose', False)

        if (not self.__temporary_directory_created) and (not dry_run):
            if not self.create_temporary_directory(dry_run=dry_run, verbose=verbose):
                return None

        directory_to_create = os.path.join(self.__temporary_directory, directory_basename)

        if not create_directory(directory_to_create, dry_run=dry_run, verbose=verbose):
            return None

        return directory_to_create
