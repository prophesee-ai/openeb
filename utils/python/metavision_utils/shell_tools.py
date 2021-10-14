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
import pprint
if platform.system() != 'Windows':
    import fcntl


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

    output, error, error_code = "", "", 0

    if dry_run or verbose:
        if working_directory:
            print("In directory '{}', execute '{}'".format(working_directory, cmd))
        else:
            print(cmd)
        sys.stdout.flush()

    if not dry_run:
        if working_directory:
            if not os.path.isdir(working_directory):
                error = "Specified working directory '{}' does not exist".format(working_directory)
                error_code = 1
                return output, error, error_code

        use_cmd_as_string = shell
        if platform.system() == 'Windows':
            use_cmd_as_string = True

        process = Popen(cmd if use_cmd_as_string else shlex.split(cmd),
                        cwd=working_directory,
                        stderr=PIPE,
                        stdout=PIPE,
                        shell=shell,
                        env=env)

        if verbose:
            if platform.system() != 'Windows':
                fcntl.fcntl(process.stdout, fcntl.F_SETFL, fcntl.fcntl(process.stdout, fcntl.F_GETFL) | os.O_NONBLOCK)
                fcntl.fcntl(process.stderr, fcntl.F_SETFL, fcntl.fcntl(process.stderr, fcntl.F_GETFL) | os.O_NONBLOCK)
                while True:
                    exit_status = process.poll()
                    try:
                        output_contents = process.stdout.read()
                        if output_contents:
                            print(output_contents.decode(), end=' ')  # , to remove trailing new line
                            output += output_contents.decode()
                        sys.stdout.flush()
                    except BaseException:
                        pass
                    try:
                        error_contents = process.stderr.read()
                        error += error_contents.decode()
                    except BaseException:
                        pass
                    if exit_status is not None:
                        error_code = exit_status
                        break

                if error:
                    print_to_stderr(error)
                    sys.stderr.flush()

            else:
                from queue import Queue, Empty
                from threading import Thread

                def enqueue_output(out, queue):
                    for line in iter(out.readline, b''):
                        queue.put(line)

                # For a non blocking read take a look at :
                # https://stackoverflow.com/questions/375427/a-non-blocking-read-on-a-subprocess-pipe-in-python
                # The above solution has been adapted in this code to have live output
                # As well as to print the contents of stderr in case the command actually succeeds
                outputQ = Queue()
                outputT = Thread(target=enqueue_output, args=(process.stdout, outputQ))
                outputT.daemon = True
                outputT.start()

                while outputT.is_alive():
                    nextline = ''
                    try:
                        if not process.stdout.closed and process.stdout.readable():
                            bnextline = outputQ.get(timeout=2)
                    except Empty:
                        break
                    else:
                        nextline = bnextline.decode(errors='ignore')
                    if nextline == '' and process.poll() is not None:
                        break
                    output += nextline
                    sys.stdout.write(nextline)
                    sys.stdout.flush()

                _, berror = process.communicate()
                error = berror.decode()
                error_code = process.returncode
                if error != '' and error_code == 0:
                    sys.stdout.write(error)
                    sys.stderr.flush()
                if error_code != 0:
                    print("The command fails with code {} ".format(error_code))
                    print_to_stderr(error)
                    sys.stderr.flush()
                else:
                    print("The command succeeds ")

        else:
            boutput, berror = process.communicate()
            output = boutput.decode()
            error = berror.decode()
            error_code = process.returncode

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
