/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

#include <iostream>
#include <fstream>
#include <memory>

#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/utils/gtest/gtest_custom.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/raw_file_header.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"

using namespace Metavision;

class DeviceDiscovery_GTest : public GTestWithTmpDir {
protected:
    virtual void SetUp() override {
        static int raw_counter = 1;
        rawfile_to_log_path_   = tmpdir_handler_->get_full_path("rawfile_" + std::to_string(++raw_counter) + ".raw");
    }

    void write_header(RawFileHeader header_to_write = RawFileHeader()) {
        std::ofstream rawfile_to_log(rawfile_to_log_path_, std::ios::out | std::ios::binary);
        if (!rawfile_to_log.is_open()) {
            std::cerr << "Could not open file for writing at " << rawfile_to_log_path_ << std::endl;
            FAIL();
        }

        rawfile_to_log << header_to_write;
        rawfile_to_log.close();
    }

    std::string rawfile_to_log_path_;
};

TEST_F(DeviceDiscovery_GTest, open_rawfile_fails_if_unknown_file) {
    std::unique_ptr<Device> device;

    ASSERT_THROW(device = DeviceDiscovery::open_raw_file("unknown_file.raw"), HalException);
}

TEST_F(DeviceDiscovery_GTest, open_rawfile_success_with_dummy_test_plugin) {
    const std::string dummy_plugin_test_path(HAL_DUMMY_TEST_PLUGIN);
    const char *env = getenv("MV_HAL_PLUGIN_PATH");

#ifdef _WIN32
    std::string s("MV_HAL_PLUGIN_PATH=");
    s += std::string(env ? env : "") + ";" + dummy_plugin_test_path;
    _putenv(s.c_str());
#else
    std::string s(env ? env : "");
    s += ":" + dummy_plugin_test_path;
    setenv("MV_HAL_PLUGIN_PATH", s.c_str(), 1);
#endif

    std::unique_ptr<Device> device;

    RawFileHeader header;
    const std::string plugin_integrator_name("__DummyTestPlugin__");
    const std::string plugin_name("hal_dummy_test_plugin");
    const std::string camera_integrator_name("__DummyTestCamera__");
    header.set_plugin_integrator_name(plugin_integrator_name);
    header.set_plugin_name(plugin_name);
    header.set_camera_integrator_name(camera_integrator_name);
    write_header(header);
    ASSERT_NO_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_));

    I_HW_Identification *hw_id = device->get_facility<I_HW_Identification>();
    ASSERT_EQ(camera_integrator_name, hw_id->get_integrator());

    I_PluginSoftwareInfo *plugin_soft_info = device->get_facility<I_PluginSoftwareInfo>();
    ASSERT_EQ(plugin_integrator_name, plugin_soft_info->get_plugin_integrator_name());
    ASSERT_EQ(plugin_name, plugin_soft_info->get_plugin_name());

#ifdef _WIN32
    s = "MV_HAL_PLUGIN_PATH=" + std::string(env ? env : "");
    _putenv(s.c_str());
#else
    setenv("MV_HAL_PLUGIN_PATH", env ? env : "", 1);
#endif
}

TEST_F(DeviceDiscovery_GTest, open_rawfile_success_with_dummy_test_plugin_and_default_search_mode) {
    const std::string dummy_plugin_test_path(HAL_DUMMY_TEST_PLUGIN);
    const char *env = getenv("MV_HAL_PLUGIN_PATH");

#ifdef _WIN32
    std::string s("MV_HAL_PLUGIN_PATH=");
    s += std::string(env ? env : "") + ";" + dummy_plugin_test_path;
    _putenv(s.c_str());
    _putenv("MV_HAL_PLUGIN_SEARCH_MODE=DEFAULT");
#else
    std::string s(env ? env : "");
    s += ":" + dummy_plugin_test_path;
    setenv("MV_HAL_PLUGIN_PATH", s.c_str(), 1);
    setenv("MV_HAL_PLUGIN_SEARCH_MODE", "DEFAULT", 1);
#endif

    std::unique_ptr<Device> device;

    RawFileHeader header;
    const std::string plugin_integrator_name("__DummyTestPlugin__");
    const std::string plugin_name("hal_dummy_test_plugin");
    const std::string camera_integrator_name("__DummyTestCamera__");
    header.set_plugin_integrator_name(plugin_integrator_name);
    header.set_plugin_name(plugin_name);
    header.set_camera_integrator_name(camera_integrator_name);
    write_header(header);
    ASSERT_NO_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_));
    I_HW_Identification *hw_id = device->get_facility<I_HW_Identification>();
    ASSERT_EQ(camera_integrator_name, hw_id->get_integrator());

    I_PluginSoftwareInfo *plugin_soft_info = device->get_facility<I_PluginSoftwareInfo>();
    ASSERT_EQ(plugin_integrator_name, plugin_soft_info->get_plugin_integrator_name());
    ASSERT_EQ(plugin_name, plugin_soft_info->get_plugin_name());

#ifdef _WIN32
    s = "MV_HAL_PLUGIN_PATH=" + std::string(env ? env : "");
    _putenv(s.c_str());
    _putenv("MV_HAL_PLUGIN_SEARCH_MODE=");
#else
    setenv("MV_HAL_PLUGIN_PATH", env ? env : "", 1);
    setenv("MV_HAL_PLUGIN_SEARCH_MODE", "", 1);
#endif
}

TEST_F(DeviceDiscovery_GTest, open_rawfile_success_with_dummy_test_plugin_and_plugin_path_search_mode) {
    const std::string dummy_plugin_test_path(HAL_DUMMY_TEST_PLUGIN);
    const char *env = getenv("MV_HAL_PLUGIN_PATH");

#ifdef _WIN32
    std::string s("MV_HAL_PLUGIN_PATH=");
    s += std::string(env ? env : "") + ";" + dummy_plugin_test_path;
    _putenv(s.c_str());
    _putenv("MV_HAL_PLUGIN_SEARCH_MODE=PLUGIN_PATH_ONLY");
#else
    std::string s(env ? env : "");
    s += ":" + dummy_plugin_test_path;
    setenv("MV_HAL_PLUGIN_PATH", s.c_str(), 1);
    setenv("MV_HAL_PLUGIN_SEARCH_MODE", "PLUGIN_PATH_ONLY", 1);
#endif

    std::unique_ptr<Device> device;

    RawFileHeader header;
    const std::string plugin_integrator_name("__DummyTestPlugin__");
    const std::string plugin_name("hal_dummy_test_plugin");
    const std::string camera_integrator_name("__DummyTestCamera__");
    header.set_plugin_integrator_name(plugin_integrator_name);
    header.set_plugin_name(plugin_name);
    header.set_camera_integrator_name(camera_integrator_name);
    write_header(header);
    ASSERT_NO_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_));
    I_HW_Identification *hw_id = device->get_facility<I_HW_Identification>();
    ASSERT_EQ(camera_integrator_name, hw_id->get_integrator());

    I_PluginSoftwareInfo *plugin_soft_info = device->get_facility<I_PluginSoftwareInfo>();
    ASSERT_EQ(plugin_integrator_name, plugin_soft_info->get_plugin_integrator_name());
    ASSERT_EQ(plugin_name, plugin_soft_info->get_plugin_name());

#ifdef _WIN32
    s = "MV_HAL_PLUGIN_PATH=" + std::string(env ? env : "");
    _putenv(s.c_str());
    _putenv("MV_HAL_PLUGIN_SEARCH_MODE=");
#else
    setenv("MV_HAL_PLUGIN_PATH", env ? env : "", 1);
    setenv("MV_HAL_PLUGIN_SEARCH_MODE", "", 1);
#endif
}

TEST_WITHOUT_CAMERA(DeviceDiscoveryNoF_GTest, open_camera_fails_if_no_camera_plugged) {
    std::unique_ptr<Device> device;
    ASSERT_NO_THROW(device = DeviceDiscovery::open(""));
    ASSERT_EQ(nullptr, device.get());
}
