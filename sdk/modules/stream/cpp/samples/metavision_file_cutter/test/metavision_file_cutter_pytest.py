#!/usr/bin/env python

# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


import pytest
import os
import re
from metavision_utils import os_tools, pytest_tools


def cut_and_check_info(input, start, end, expected_output_info=None):
    """"Runs metavision_file_cutter on input file and checks the output

    Args:
        input (str): path of the input file
        start, end : cut range [s]
        expected_output_info : expected output on running metavision_file_info on the output file
                               If none is provided, we assume that we have to get the same info as
                               the input file (which is the case when the range covers all the input file)
    """

    # Before launching the app, check the dataset file exists
    assert os.path.exists(input)

    # Create temporary directory, where we'll put the output
    tmp_dir = os_tools.TemporaryDirectoryHandler()
    ext = os.path.splitext(input)[1]
    output_file_name = "cut_{}_{}{}".format(start, end, ext)
    output_file_path = os.path.join(tmp_dir.temporary_directory(), output_file_name)

    cmd = "./metavision_file_cutter -i \"{}\" --start {} --end {} -o {}".format(input, start, end,
                                                                                output_file_path)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check output file has been written
    assert os.path.exists(output_file_path)

    # Now, with the app metavision_file_info we check the information
    cmd = "./metavision_file_info -i {}".format(output_file_path)
    info_cut_file, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0

    # Now check output
    if not expected_output_info:
        # Then run metavision_file_info on input file, since we expect the same output
        cmd = "./metavision_file_info -i \"{}\"".format(input)
        info_input_file, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

        # Check no error occurred
        assert error_code == 0

        # Need to format the output
        expected_output_info = info_input_file.replace(input, "{}").replace(os.path.basename(input), "{}")

    expected_output_info_formatted = expected_output_info.format(
        output_file_name, re.escape(os.path.realpath(output_file_path)))

    output_strip = pytest_tools.get_mv_info_stripped_output(info_cut_file)
    expected_output_strip = pytest_tools.get_mv_info_stripped_output(expected_output_info_formatted)
    # Do not check plugin name, it may differ if the original plugin does not exist anymore
    output_strip = re.sub("Plugin name.*\n", "", output_strip)
    expected_output_strip = re.sub("Plugin name.*\n", "", expected_output_strip)
    assert re.search(expected_output_strip, output_strip)


def pytestcase_test_metavision_file_cutter_show_help():
    """
    Checks output of metavision_file_cutter when displaying help message
    """

    cmd = "./metavision_file_cutter --help"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check that the options shows in the output
    assert "Options:" in output, "******\nMissing options display in output :{}\n******".format(output)


def pytestcase_test_metavision_file_cutter_nonexistent_input_file():
    """
    Checks that metavision_file_cutter returns an error when passing an input file that doesn't exist
    """

    # Create temporary directory for nonexistent file
    tmp_dir = os_tools.TemporaryDirectoryHandler()
    input_file = os.path.join(tmp_dir.temporary_directory(), "nonexistent_in.raw")
    output_file = os.path.join(tmp_dir.temporary_directory(), "nonexistent_out.raw")

    cmd = "./metavision_file_cutter -i {} --start {} --end {} -o {}".format(input_file, 0, 2, output_file)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited with error
    assert error_code != 0

    # And now check that the error came from the fact that the input file could not be read
    assert "No such file or directory" in output


def pytestcase_test_metavision_file_cutter_missing_input_args():
    """
    Checks that metavision_file_cutter returns an error when not passing required input args
    """

    cmd = "./metavision_file_cutter"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited with error
    assert error_code != 0

    # And now check that the error came from the fact that the input file arg is missing
    assert re.search("Parsing error: the option (.+) is required but missing", output)


def pytestcase_test_metavision_file_cutter_invalid_range(dataset_dir):
    """
    Checks that metavision_file_cutter returns an error when passing inconsistent values for start and stop
    """

    # To be sure the error isn't thrown because the input file doesn't exist, use one from the datasets
    input_file = os.path.join(dataset_dir, "openeb", "gen31_timer.raw")
    assert os.path.exists(input_file)

    # Create temporary directory for output file
    tmp_dir = os_tools.TemporaryDirectoryHandler()
    output_file = os.path.join(tmp_dir.temporary_directory(), "data_out.raw")

    start = 4
    end = 2
    cmd = "./metavision_file_cutter -i {} --start {} --end {} -o {}".format(input_file, start, end, output_file)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited with error
    assert error_code != 0

    # And now check that the error came from the fact that the input range was invalid
    assert re.search("End time {} is less than or equal to start {}".format(end, start), output)


def pytestcase_test_metavision_file_cutter_on_raw_gen31_recording_full_cut(dataset_dir):
    """
    Checks output of metavision_file_cutter application when the range given spans through the whole file
    """

    filename = "gen31_timer.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 0
    end = 15  # This recording is ~13s, so 15 is well after its end

    cut_and_check_info(filename_full, start, end)


def pytestcase_test_metavision_file_cutter_on_raw_gen31_recording_from_0s_to_6s(dataset_dir):
    """
    Checks output of metavision_file_cutter on dataset gen31_timer.raw, cutting from 0s to 6s
    """

    filename = "gen31_timer.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 0
    end = 6

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            5s 999ms 281us
Integrator          Prophesee
Plugin name         hal_plugin_gen31_fx3
Data encoding       EVT2
Camera generation   3.1
Camera serial       00001621

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  14066479            16                  5999281             2.3 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


def pytestcase_test_metavision_file_cutter_on_raw_gen31_recording_from_8s_to_11s(dataset_dir):
    """
    Checks output of metavision_file_cutter on dataset gen31_timer.raw, cutting from 8s to 11s
    """

    filename = "gen31_timer.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 8
    end = 11

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            3s 0ms 200us
Integrator          Prophesee
Plugin name         hal_plugin_gen31_fx3
Data encoding       EVT2
Camera generation   3.1
Camera serial       00001621

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  5590889             48                  3000200             1.9 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


def pytestcase_test_metavision_file_cutter_on_raw_gen4_evt2_recording_full_cut(dataset_dir):
    """
    Checks output of metavision_file_cutter application when the range given spans throws all the file
    """

    filename = "gen4_evt2_hand.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 0
    end = 11

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            10s 442ms 743us
Integrator          Prophesee
Plugin name         hal_plugin_gen41_evk3
Data encoding       EVT2
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  17025195            49                  10442743            1.6 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


def pytestcase_test_metavision_file_cutter_on_raw_gen4_evt2_recording_from_2s_to_3s(dataset_dir):
    """
    Checks output of metavision_file_cutter on dataset gen4_evt2_hand.raw, cutting from 2s to 3s
    """

    filename = "gen4_evt2_hand.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 2
    end = 3

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            999ms 995us
Integrator          Prophesee
Plugin name         hal_plugin_gen41_evk3
Data encoding       EVT2
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  1985443             32                  999995              2.0 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


def pytestcase_test_metavision_file_cutter_on_raw_gen4_evt2_recording_from_4s_to_10s(dataset_dir):
    """
    Checks output of metavision_file_cutter on dataset gen4_evt2_hand.raw, cutting from 4s to 10s
    """

    filename = "gen4_evt2_hand.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 4
    end = 10

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            5s 999ms 686us
Integrator          Prophesee
Plugin name         hal_plugin_gen41_evk3
Data encoding       EVT2
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  9468485             48                  5999686             1.6 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


def pytestcase_test_metavision_file_cutter_on_raw_gen4_evt3_recording_full_cut(dataset_dir):
    """
    Checks output of metavision_file_cutter application when the range given spans throws all the file
    """

    filename = "gen4_evt3_hand.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 0
    end = 16

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            15s 441ms 920us
Integrator          Prophesee
Plugin name         hal_plugin_gen41_evk3
Data encoding       EVT3
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  18094969            5714                15000125            1.2 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


def pytestcase_test_metavision_file_cutter_on_raw_gen4_evt3_recording_from_3s_to_7s(dataset_dir):
    """
    Checks output of metavision_file_cutter on dataset gen4_evt3_hand.raw, cutting from 3s to 7s
    """

    filename = "gen4_evt3_hand.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 3
    end = 7

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            4s 5ms 779us
Integrator          Prophesee
Plugin name         hal_plugin_gen41_evk3
Data encoding       EVT3
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  4884780             5424                4005779             1.2 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


def pytestcase_test_metavision_file_cutter_on_raw_gen4_evt3_recording_from_8s_to_9s(dataset_dir):
    """
    Checks output of metavision_file_cutter on dataset gen4_evt3_hand.raw, cutting from 8s to 9s
    """

    filename = "gen4_evt3_hand.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 8
    end = 9

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            1s 4ms 384us
Integrator          Prophesee
Plugin name         hal_plugin_gen41_evk3
Data encoding       EVT3
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  1319849             4432                1004384             1.3 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


def pytestcase_test_metavision_file_cutter_on_raw_gen4_evt3_recording_from_4s_to_15s(dataset_dir):
    """
    Checks output of metavision_file_cutter on dataset gen4_evt3_hand.raw, cutting from 4s to 15s
    """

    filename = "gen4_evt3_hand.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 4
    end = 15

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            11s 6ms 181us
Integrator          Prophesee
Plugin name         hal_plugin_gen41_evk3
Data encoding       EVT3
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  12759075            6064                11006181            1.2 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_cutter_on_hdf5_gen31_recording_full_cut(dataset_dir):
    """
    Checks output of metavision_file_cutter application when the range given spans through the whole file
    """

    filename = "gen31_timer.hdf5"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 0
    end = 15  # This recording is ~13s, so 15 is well after its end

    cut_and_check_info(filename_full, start, end)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_cutter_on_hdf5_gen31_recording_from_0s_to_6s(dataset_dir):
    """
    Checks output of metavision_file_cutter on dataset gen31_timer.hdf5, cutting from 0s to 6s
    """

    filename = "gen31_timer.hdf5"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 0
    end = 6

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            5s 999ms 996us
Integrator          Prophesee
Data encoding       ECF
Camera generation   3.1
Camera serial       00001621

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  14067262            16                  5999996             2.3 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_cutter_on_hdf5_gen31_recording_from_8s_to_11s(dataset_dir):
    """
    Checks output of metavision_file_cutter on dataset gen31_timer.hdf5, cutting from 8s to 11s
    """

    filename = "gen31_timer.hdf5"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 8
    end = 11

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            2s 999ms 999us
Integrator          Prophesee
Data encoding       ECF
Camera generation   3.1
Camera serial       00001621

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  5590599             2                   2999999             1.9 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_cutter_on_hdf5_gen4_evt2_recording_full_cut(dataset_dir):
    """
    Checks output of metavision_file_cutter application when the range given spans throws all the file
    """

    filename = "gen4_evt2_hand.hdf5"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 0
    end = 11

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            10s 442ms 743us
Integrator          Prophesee
Data encoding       ECF
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  17025195            49                  10442743            1.6 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_cutter_on_hdf5_gen4_evt2_recording_from_2s_to_3s(dataset_dir):
    """
    Checks output of metavision_file_cutter on dataset gen4_evt2_hand.hdf5, cutting from 2s to 3s
    """

    filename = "gen4_evt2_hand.hdf5"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 2
    end = 3

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            999ms 999us
Integrator          Prophesee
Data encoding       ECF
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  1985546             0                   999999              2.0 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_cutter_on_hdf5_gen4_evt2_recording_from_4s_to_10s(dataset_dir):
    """
    Checks output of metavision_file_cutter on dataset gen4_evt2_hand.hdf5, cutting from 4s to 10s
    """

    filename = "gen4_evt2_hand.hdf5"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 4
    end = 10

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            5s 999ms 998us
Integrator          Prophesee
Data encoding       ECF
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  9468511             0                   5999998             1.6 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_cutter_on_hdf5_gen4_evt3_recording_full_cut(dataset_dir):
    """
    Checks output of metavision_file_cutter application when the range given spans throws all the file
    """

    filename = "gen4_evt3_hand.hdf5"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 0
    end = 16

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            15s 0ms 125us
Integrator          Prophesee
Data encoding       ECF
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  18094969            5714                15000125            1.2 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_cutter_on_hdf5_gen4_evt3_recording_from_3s_to_7s(dataset_dir):
    """
    Checks output of metavision_file_cutter on dataset gen4_evt3_hand.hdf5, cutting from 3s to 7s
    """

    filename = "gen4_evt3_hand.hdf5"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 3
    end = 7

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            3s 999ms 998us
Integrator          Prophesee
Data encoding       ECF
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  4884485             0                   3999998             1.2 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_cutter_on_hdf5_gen4_evt3_recording_from_8s_to_9s(dataset_dir):
    """
    Checks output of metavision_file_cutter on dataset gen4_evt3_hand.hdf5, cutting from 8s to 9s
    """

    filename = "gen4_evt3_hand.hdf5"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 8
    end = 9

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            999ms 999us
Integrator          Prophesee
Data encoding       ECF
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  1319961             0                   999999              1.3 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_cutter_on_hdf5_gen4_evt3_recording_from_4s_to_15s(dataset_dir):
    """
    Checks output of metavision_file_cutter on dataset gen4_evt3_hand.hdf5, cutting from 4s to 15s
    """

    filename = "gen4_evt3_hand.hdf5"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 4
    end = 15

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            10s 999ms 997us
Integrator          Prophesee
Data encoding       ECF
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  12759018            0                   10999997            1.2 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)
