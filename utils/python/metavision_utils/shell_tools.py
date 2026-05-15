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
Module defining utility function to execute commands
"""
from __future__ import print_function
from subprocess import Popen, PIPE
import shlex
import sys
import os
import platform
import shutil
from command_runner import command_runner


def print_to_stderr(msg):
    """Prints message to stderr

    Args :
        msg (str): message to print
    """
    sys.stderr.write(msg)


def execute_cmd(cmd, **kwargs):
    """Executes command

    Args :
        cmd (str): command to execute

    Keyword Args:
        dry_run (bool): if true, just prints the commands that would be executed, but
                  without actually executing them. (default = False)
        verbose (bool): if true, prints both the commands that are being executed
                        and their output. (default = False)
        working_directory (str): path to the directory where to execute the command
        shell (bool): If True, the specified command will be executed through the shell.
            WARNING : Using shell=True can be a security hazard
            (see https://docs.python.org/2/library/subprocess.html#frequently-used-arguments).
        env (os.environ) : If defined, enables to run the command in a customized environment

    Returns : a tuple containing output, error_output and error_code of the command executed

    """
    dry_run = kwargs.pop('dry_run', False)
    verbose = kwargs.pop('verbose', False)
    working_directory = kwargs.pop('working_directory', None)
    shell = kwargs.pop('shell', False)
    env = kwargs.pop('env', None)
    detached = kwargs.pop('detached', False)

    output, error, error_code = "", "", 0

    if dry_run or verbose:
        if working_directory:
            print("In directory '{}', execute '{}'".format(working_directory, cmd))
        else:
            print(cmd)

    if not dry_run:
        if working_directory:
            if not os.path.isdir(working_directory):
                error = "Specified working directory '{}' does not exist".format(working_directory)
                error_code = 1
                return output, error, error_code

        if detached:
            use_cmd_as_string = shell
            if platform.system() == 'Windows':
                use_cmd_as_string = True

            process = Popen(cmd if use_cmd_as_string else shlex.split(cmd),
                            cwd=working_directory,
                            stderr=None,
                            stdout=None,
                            shell=shell,
                            env=env,
                            close_fds=True)
            res = process.poll()
            if res is not None:
                return "", "", res
            else:
                return "", "", 0
        else:
            def _print_to_stdout(s):
                if verbose:
                    sys.stdout.write(s)
                    sys.stdout.flush()

            def _print_to_stderr(s):
                if verbose:
                    sys.stderr.write(s)
                    sys.stderr.flush()

            sys.stdout.flush()
            sys.stderr.flush()
            error_code, output, error = command_runner(cmd,
                                                       cwd=working_directory,
                                                       shell=shell,
                                                       env=env,
                                                       encoding="utf-8",
                                                       method="poller",
                                                       split_streams=True,
                                                       stdout=_print_to_stdout,
                                                       stderr=_print_to_stderr)
            # When there is nothing on stderr, command_runner returns None
            if error is None:
                error = ""

    return output, error, error_code


def update_env_from_string(env_string):
    """
    A function that returns an updated os.environ from env_string
    """
    excluded_keys = ["_", "SHLVL", "PWD", "OLDPWD"]
    env = os.environ
    for line in env_string.split("\n"):
        (key, _, value) = line.partition("=")
        if key and value and key not in excluded_keys:
            env[key] = value
    return env
