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

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>

#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/utils/gtest/gtest_custom.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/facilities/i_camera_synchronization.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_geometry.h"
#include "metavision/hal/facilities/i_trigger_in.h"
#include "metavision/hal/facilities/i_trigger_out.h"
#include "metavision/hal/facilities/i_ll_biases.h"
#include "metavision/hal/facilities/i_event_rate_activity_filter_module.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_hw_register.h"
#include "metavision/hal/facilities/i_roi.h"
#include "metavision/hal/facilities/i_monitoring.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "devices/utils/device_system_id.h"
#include "geometries/vga_geometry.h"
#include "geometries/hd_geometry.h"
#include "metavision/psee_hw_layer/boards/rawfile/psee_raw_file_header.h"

using namespace Metavision;

class DeviceDiscoveryPseePlugins_GTest : public GTestWithTmpDir {
protected:
    virtual void SetUp() override {
        static int raw_counter = 1;
        rawfile_to_log_path_   = tmpdir_handler_->get_full_path("rawfile_" + std::to_string(++raw_counter) + ".raw");
    }

    void write_header(RawFileHeader header_to_write) {
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

static const std::vector<SystemId> offline_supported_system_ids{
    {SystemId::SYSTEM_CCAM3_GEN3, SystemId::SYSTEM_CCAM3_GEN31, SystemId::SYSTEM_CCAM5_GEN31,
     SystemId::SYSTEM_EVK3_GEN31_EVT3, SystemId::SYSTEM_EVK2_GEN31, SystemId::SYSTEM_CCAM4_GEN3,
     SystemId::SYSTEM_CCAM4_GEN3_EVK, SystemId::SYSTEM_CCAM4_GEN3_REV_B, SystemId::SYSTEM_CCAM4_GEN3_REV_B_EVK,
     SystemId::SYSTEM_CCAM4_GEN3_REV_B_EVK_BRIDGE, SystemId::SYSTEM_VISIONCAM_GEN3, SystemId::SYSTEM_VISIONCAM_GEN3_EVK,
     SystemId::SYSTEM_CCAM3_GEN4, SystemId::SYSTEM_EVK2_GEN4, SystemId::SYSTEM_VISIONCAM_GEN31,
     SystemId::SYSTEM_VISIONCAM_GEN31_EVK, SystemId::SYSTEM_EVK2_GEN41}};

static const std::vector<SystemId> offline_unsupported_system_ids{
    {SystemId::SYSTEM_CCAM3_GEN2, SystemId::SYSTEM_CCAM2_STEREO, SystemId::SYSTEM_CCAM2_STEREO_MAPPING,
     SystemId::SYSTEM_STEREO_DEMO, SystemId::SYSTEM_CCAM3_STEREO_LEFT_GTP, SystemId::SYSTEM_CCAM3_STEREO_LEFT,
     SystemId::SYSTEM_CCAM2_STEREO_MERGE_IMU, SystemId::SYSTEM_CCAM3_GEN2, SystemId::SYSTEM_CCAM4_GEN4_EVK,
     SystemId::SYSTEM_CCAM5_GEN4_EVK_BRIDGE, SystemId::SYSTEM_INVALID_NO_FPGA}};

TEST_F(DeviceDiscoveryPseePlugins_GTest, offline_supported_system_id_are_not_unsupported_in_the_gtest) {
    for (const auto system_id : offline_supported_system_ids) {
        ASSERT_TRUE(std::find(offline_unsupported_system_ids.cbegin(), offline_unsupported_system_ids.cend(),
                              system_id) == offline_unsupported_system_ids.cend());
    }
}

TEST_F(DeviceDiscoveryPseePlugins_GTest, open_rawfile_succeeds_without_integrator_and_plugin_name) {
    std::unique_ptr<Device> device;
    for (const long system_id : offline_supported_system_ids) {
        auto header = std::stringstream();
        header << "% Date 2014-02-28 13:37:42" << std::endl
               << "% system_ID " << system_id << std::endl
               << "% serial_number 00001337" << std::endl;
        write_header(RawFileHeader(header));
        ASSERT_NO_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_));

        // If no info is provided, we assume that it's an old RAW file that only
        // psee plugins can open

        I_HW_Identification *hw_id = device->get_facility<I_HW_Identification>();
        ASSERT_EQ("Prophesee", hw_id->get_integrator());
    }
}

TEST_F(DeviceDiscoveryPseePlugins_GTest, open_rawfile_succeeds_with_integrator_and_no_plugin_name) {
    std::unique_ptr<Device> device;
    for (const long system_id : offline_supported_system_ids) {
        auto header = std::stringstream();
        header << "% Date 2014-02-28 13:37:42" << std::endl
               << "% system_ID " << system_id << std::endl
               << "% integrator_name Prophesee" << std::endl
               << "% serial_number 00001337" << std::endl;
        write_header(RawFileHeader(header));
        ASSERT_NO_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_));
        I_HW_Identification *hw_id = device->get_facility<I_HW_Identification>();
        ASSERT_EQ("Prophesee", hw_id->get_integrator());
    }
}

TEST_F(DeviceDiscoveryPseePlugins_GTest, open_rawfile_succeeds_with_no_integrator_and_plugin_name) {
    std::unique_ptr<Device> device;
    for (const long system_id : offline_supported_system_ids) {
        std::string plugin_name;
        switch (system_id) {
        case SystemId::SYSTEM_CCAM3_GEN3:
        case SystemId::SYSTEM_CCAM4_GEN3:
        case SystemId::SYSTEM_CCAM4_GEN3_EVK:
        case SystemId::SYSTEM_CCAM4_GEN3_REV_B:
        case SystemId::SYSTEM_CCAM4_GEN3_REV_B_EVK:
        case SystemId::SYSTEM_CCAM4_GEN3_REV_B_EVK_BRIDGE:
        case SystemId::SYSTEM_VISIONCAM_GEN3:
        case SystemId::SYSTEM_VISIONCAM_GEN3_EVK:
            plugin_name = "hal_plugin_gen3_fx3";
            break;
        case SystemId::SYSTEM_CCAM3_GEN31:
        case SystemId::SYSTEM_VISIONCAM_GEN31:
        case SystemId::SYSTEM_VISIONCAM_GEN31_EVK:
            plugin_name = "hal_plugin_gen31_fx3";
            break;
        case SystemId::SYSTEM_CCAM5_GEN31:
        case SystemId::SYSTEM_EVK3_GEN31_EVT3:
            plugin_name = "hal_plugin_gen31_evk3";
            break;
        case SystemId::SYSTEM_EVK2_GEN31:
            plugin_name = "hal_plugin_gen31_evk2";
            break;
        case SystemId::SYSTEM_CCAM3_GEN4:
            plugin_name = "hal_plugin_gen4_fx3";
            break;
        case SystemId::SYSTEM_EVK2_GEN4:
            plugin_name = "hal_plugin_gen4_evk2";
            break;
        case SystemId::SYSTEM_EVK2_GEN41:
            plugin_name = "hal_plugin_gen41_evk2";
            break;
        default:
            std::cerr
                << "This is an enum fallback that should not have been reached. Is the input camera supported but not "
                   "handled in this test ?"
                << std::endl;
            FAIL();
        }

        auto header = std::stringstream();
        header << "% Date 2014-02-28 13:37:42" << std::endl
               << "% system_ID " << system_id << std::endl
               << "% plugin_name " << plugin_name << std::endl
               << "% serial_number 00001337" << std::endl;
        write_header(RawFileHeader(header));
        ASSERT_NO_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_));
    }
}

TEST_F(DeviceDiscoveryPseePlugins_GTest, open_rawfile_fails_with_bad_integrator_name) {
    std::unique_ptr<Device> device;
    for (const long system_id : offline_supported_system_ids) {
        auto header = std::stringstream();
        header << "% Date 2014-02-28 13:37:42" << std::endl
               << "% system_ID " << system_id << std::endl
               << "% integrator_name _aZ0$fooBar@%!" << std::endl
               << "% serial_number 00001337" << std::endl;
        write_header(RawFileHeader(header));
        ASSERT_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_), HalException);
    }
}

TEST_F(DeviceDiscoveryPseePlugins_GTest, open_rawfile_fails_with_bad_integrator_and_plugin_name) {
    std::unique_ptr<Device> device;
    // This test shall fail because we can't infer features with a system_ID from an unknown
    // integrator, and the header doesn't explicitely specify the data format
    for (const long system_id : offline_supported_system_ids) {
        auto header = std::stringstream();
        header << "% Date 2014-02-28 13:37:42" << std::endl
               << "% system_ID " << system_id << std::endl
               << "% integrator_name _aZ0$fooBar@%!" << std::endl
               << "% plugin_name _aZ0$fooBar@%!" << std::endl
               << "% serial_number 00001337" << std::endl;
        write_header(RawFileHeader(header));
        ASSERT_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_), HalException);
    }
}

TEST_F(DeviceDiscoveryPseePlugins_GTest, open_rawfile_success_with_supported_system_ids) {
    for (const long system_id : offline_supported_system_ids) {
        auto header = std::stringstream();
        header << "% Date 2014-02-28 13:37:42" << std::endl
               << "% system_ID " << system_id << std::endl
               << "% serial_number 00001337" << std::endl;
        write_header(RawFileHeader(header));

        std::unique_ptr<Device> device;
        ASSERT_NO_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_));

        // Check hw identification
        I_HW_Identification *hw_id = device->get_facility<I_HW_Identification>();
        ASSERT_NE(nullptr, hw_id);
        ASSERT_EQ("File", hw_id->get_connection_type());

        // Check decoder
        I_EventsStreamDecoder *decoder = device->get_facility<I_EventsStreamDecoder>();
        ASSERT_NE(nullptr, decoder);

        // Check geometry
        I_Geometry *geometry = device->get_facility<I_Geometry>();
        ASSERT_NE(nullptr, geometry);

        switch (system_id) {
        case SystemId::SYSTEM_CCAM3_GEN3:
        case SystemId::SYSTEM_CCAM4_GEN3:
        case SystemId::SYSTEM_CCAM4_GEN3_EVK:
        case SystemId::SYSTEM_CCAM4_GEN3_REV_B:
        case SystemId::SYSTEM_CCAM4_GEN3_REV_B_EVK:
        case SystemId::SYSTEM_CCAM4_GEN3_REV_B_EVK_BRIDGE:
        case SystemId::SYSTEM_VISIONCAM_GEN3:
        case SystemId::SYSTEM_VISIONCAM_GEN3_EVK:
            ASSERT_EQ("Gen3.0", hw_id->get_sensor_info().name_);
            ASSERT_EQ(VGAGeometry::width_, geometry->get_width());
            ASSERT_EQ(VGAGeometry::height_, geometry->get_height());
            ASSERT_EQ(4, decoder->get_raw_event_size_bytes());
            break;
        case SystemId::SYSTEM_CCAM3_GEN31:
        case SystemId::SYSTEM_CCAM5_GEN31:
        case SystemId::SYSTEM_EVK2_GEN31:
        case SystemId::SYSTEM_VISIONCAM_GEN31:
        case SystemId::SYSTEM_VISIONCAM_GEN31_EVK:
            ASSERT_EQ("Gen3.1", hw_id->get_sensor_info().name_);
            ASSERT_EQ(VGAGeometry::width_, geometry->get_width());
            ASSERT_EQ(VGAGeometry::height_, geometry->get_height());
            ASSERT_EQ(4, decoder->get_raw_event_size_bytes());
            break;
        case SystemId::SYSTEM_EVK3_GEN31_EVT3:
            ASSERT_EQ("Gen3.1", hw_id->get_sensor_info().name_);
            ASSERT_EQ(VGAGeometry::width_, geometry->get_width());
            ASSERT_EQ(VGAGeometry::height_, geometry->get_height());
            ASSERT_EQ(2, decoder->get_raw_event_size_bytes());
            break;
        case SystemId::SYSTEM_CCAM3_GEN4:
        case SystemId::SYSTEM_EVK2_GEN4:
            ASSERT_EQ("Gen4.0", hw_id->get_sensor_info().name_);
            ASSERT_EQ(HDGeometry::width_, geometry->get_width());
            ASSERT_EQ(HDGeometry::height_, geometry->get_height());
            ASSERT_EQ(4,
                      decoder->get_raw_event_size_bytes()); // Default fallback when no evt format is indicated in the
                                                            // rawfile header
            break;
        case SystemId::SYSTEM_EVK2_GEN41:
            ASSERT_EQ("Gen4.1", hw_id->get_sensor_info().name_);
            ASSERT_EQ(HDGeometry::width_, geometry->get_width());
            ASSERT_EQ(HDGeometry::height_, geometry->get_height());
            ASSERT_EQ(4,
                      decoder->get_raw_event_size_bytes()); // Default fallback when no evt format is indicated in the
                                                            // rawfile header
            break;
        default:
            std::cerr
                << "This is an enum fallback that should not have been reached. Is the input camera supported but not "
                   "handled in this test ?"
                << std::endl;
            FAIL();
        }

        // Check other facilities presence
        ASSERT_NE(nullptr, device->get_facility<I_PluginSoftwareInfo>());
        ASSERT_NE(nullptr, device->get_facility<I_EventsStreamDecoder>());
        ASSERT_NE(nullptr, device->get_facility<I_EventDecoder<EventCD>>());
        ASSERT_NE(nullptr, device->get_facility<I_EventDecoder<EventExtTrigger>>());
    }
}

TEST_F(DeviceDiscoveryPseePlugins_GTest, open_rawfile_with_geometry_and_format) {
    // Just test VGA EVT2 and HD EVT3, should have a reasonable code coverage
    {
        auto header = std::stringstream();
        header << "% format EVT3" << std::endl << "% geometry 1280x720" << std::endl;
        write_header(RawFileHeader(header));
        std::unique_ptr<Device> device;
        ASSERT_NO_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_));

        // Check decoder
        I_EventsStreamDecoder *decoder = device->get_facility<I_EventsStreamDecoder>();
        ASSERT_NE(nullptr, decoder);

        // Check geometry
        I_Geometry *geometry = device->get_facility<I_Geometry>();
        ASSERT_NE(nullptr, geometry);

        ASSERT_EQ(HDGeometry::width_, geometry->get_width());
        ASSERT_EQ(HDGeometry::height_, geometry->get_height());
        ASSERT_EQ(2, decoder->get_raw_event_size_bytes());

        // Check facilites presence
        ASSERT_NE(nullptr, device->get_facility<I_PluginSoftwareInfo>());
        ASSERT_NE(nullptr, device->get_facility<I_EventsStreamDecoder>());
        ASSERT_NE(nullptr, device->get_facility<I_EventDecoder<EventCD>>());
        ASSERT_NE(nullptr, device->get_facility<I_EventDecoder<EventExtTrigger>>());
    }
    {
        auto header = std::stringstream();
        header << "% format EVT2" << std::endl << "% geometry 640x480" << std::endl;
        write_header(RawFileHeader(header));
        std::unique_ptr<Device> device;
        ASSERT_NO_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_));

        // Check decoder
        I_EventsStreamDecoder *decoder = device->get_facility<I_EventsStreamDecoder>();
        ASSERT_NE(nullptr, decoder);

        // Check geometry
        I_Geometry *geometry = device->get_facility<I_Geometry>();
        ASSERT_NE(nullptr, geometry);

        ASSERT_EQ(VGAGeometry::width_, geometry->get_width());
        ASSERT_EQ(VGAGeometry::height_, geometry->get_height());
        ASSERT_EQ(4, decoder->get_raw_event_size_bytes());

        // Check facilites presence
        ASSERT_NE(nullptr, device->get_facility<I_PluginSoftwareInfo>());
        ASSERT_NE(nullptr, device->get_facility<I_EventsStreamDecoder>());
        ASSERT_NE(nullptr, device->get_facility<I_EventDecoder<EventCD>>());
        ASSERT_NE(nullptr, device->get_facility<I_EventDecoder<EventExtTrigger>>());
    }
}

TEST_F(DeviceDiscoveryPseePlugins_GTest, open_rawfile_with_unknown_format) {
    auto header = std::stringstream();
    header << "% format UNSUPPORTED_FORMAT" << std::endl << "% geometry 1280x720" << std::endl;
    write_header(RawFileHeader(header));
    std::unique_ptr<Device> device;
    ASSERT_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_), HalException);
}

TEST_F(DeviceDiscoveryPseePlugins_GTest, open_rawfile_doesnt_have_board_facilities) {
    for (const long system_id : offline_supported_system_ids) {
        auto header = std::stringstream();
        header << "% Date 2014-02-28 13:37:42" << std::endl
               << "% system_ID " << system_id << std::endl
               << "% serial_number 00001337" << std::endl;
        write_header(RawFileHeader(header));

        std::unique_ptr<Device> device;
        ASSERT_NO_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_));

        // Check hw identification
        ASSERT_EQ(nullptr, device->get_facility<I_LL_Biases>());
        ASSERT_EQ(nullptr, device->get_facility<I_CameraSynchronization>());
        ASSERT_EQ(nullptr, device->get_facility<I_HW_Register>());
        ASSERT_EQ(nullptr, device->get_facility<I_TriggerIn>());
        ASSERT_EQ(nullptr, device->get_facility<I_TriggerOut>());
        ASSERT_EQ(nullptr, device->get_facility<I_ROI>());
        ASSERT_EQ(nullptr, device->get_facility<I_Monitoring>());
    }
}

TEST_F(DeviceDiscoveryPseePlugins_GTest, open_rawfile_succeeds_with_evt_converter_output) {
    // MetavisionESP::evt_converter removed redundant information, but kept a unique set of fields
    std::unique_ptr<Device> device;
    auto header = std::stringstream();
    header << "% format EVT3" << std::endl
           << "% integrator_name Prophesee" << std::endl
           << "% plugin_name hal_plugin_gen41_evk2" << std::endl
           << "% system_ID 39" << std::endl;
    write_header(RawFileHeader(header));
    ASSERT_NO_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_));
    I_HW_Identification *hw_id = device->get_facility<I_HW_Identification>();
    ASSERT_EQ("Prophesee", hw_id->get_integrator());

    // Check facilites presence
    ASSERT_NE(nullptr, device->get_facility<I_PluginSoftwareInfo>());
    ASSERT_NE(nullptr, device->get_facility<I_EventsStreamDecoder>());
    ASSERT_NE(nullptr, device->get_facility<I_EventDecoder<EventCD>>());
    ASSERT_NE(nullptr, device->get_facility<I_EventDecoder<EventExtTrigger>>());
    I_EventsStreamDecoder *decoder = device->get_facility<I_EventsStreamDecoder>();
    ASSERT_NE(nullptr, decoder);
    ASSERT_EQ(2, decoder->get_raw_event_size_bytes());
    I_Geometry *geometry = device->get_facility<I_Geometry>();
    ASSERT_NE(nullptr, geometry);
    ASSERT_EQ(1280, geometry->get_width());
    ASSERT_EQ(720, geometry->get_height());
}

TEST_F(DeviceDiscoveryPseePlugins_GTest, open_rawfile_succeeds_with_saphir_bringup) {
    std::unique_ptr<Device> device;
    auto header = std::stringstream();
    header << "% date 2022-07-14 13:37:00" << std::endl
           << "% firmware_version 0.0.0" << std::endl
           << "% format EVT21" << std::endl
           << "% geometry 1792x1792" << std::endl
           << "% integrator_name Prophesee" << std::endl
           << "% plugin_name hal_plugin_sensorlib_tz" << std::endl
           << "% sensor_generation 0.0" << std::endl
           << "% system_ID 0" << std::endl;
    write_header(RawFileHeader(header));
    ASSERT_NO_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_));
    I_HW_Identification *hw_id = device->get_facility<I_HW_Identification>();
    ASSERT_EQ("Prophesee", hw_id->get_integrator());

    // We should have absurd geometry and Evt2.1 decoder
    ASSERT_NE(nullptr, device->get_facility<I_PluginSoftwareInfo>());
    ASSERT_NE(nullptr, device->get_facility<I_EventsStreamDecoder>());
    ASSERT_NE(nullptr, device->get_facility<I_EventDecoder<EventCD>>());
    ASSERT_NE(nullptr, device->get_facility<I_EventDecoder<EventExtTrigger>>());
    I_EventsStreamDecoder *decoder = device->get_facility<I_EventsStreamDecoder>();
    ASSERT_NE(nullptr, decoder);
    ASSERT_EQ(8, decoder->get_raw_event_size_bytes());
    I_Geometry *geometry = device->get_facility<I_Geometry>();
    ASSERT_NE(nullptr, geometry);
    ASSERT_EQ(1792, geometry->get_width());
    ASSERT_EQ(1792, geometry->get_height());
}

TEST_F(DeviceDiscoveryPseePlugins_GTest, open_rawfile_succeeds_with_hal_info_and_format) {
    std::unique_ptr<Device> device;
    auto header = std::stringstream();
    header << "% camera_integrator_name Prophesee" << std::endl
           << "% format EVT3;width=1920;height=1200" << std::endl
           << "% plugin_integrator_name Prophesee" << std::endl
           << "% plugin_name hal_plugin_prophesee" << std::endl;
    write_header(RawFileHeader(header));
    ASSERT_NO_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_));

    // We should have FullHD 16:10 geometry and Evt3 decoder
    ASSERT_NE(nullptr, device->get_facility<I_PluginSoftwareInfo>());
    ASSERT_NE(nullptr, device->get_facility<I_EventsStreamDecoder>());
    ASSERT_NE(nullptr, device->get_facility<I_EventDecoder<EventCD>>());
    ASSERT_NE(nullptr, device->get_facility<I_EventDecoder<EventExtTrigger>>());
    I_EventsStreamDecoder *decoder = device->get_facility<I_EventsStreamDecoder>();
    ASSERT_NE(nullptr, decoder);
    ASSERT_EQ(2, decoder->get_raw_event_size_bytes());
    I_Geometry *geometry = device->get_facility<I_Geometry>();
    ASSERT_NE(nullptr, geometry);
    ASSERT_EQ(1920, geometry->get_width());
    ASSERT_EQ(1200, geometry->get_height());
}

TEST_F(DeviceDiscoveryPseePlugins_GTest, open_rawfile_does_not_work_with_unsupported_id) {
    for (const long system_id : offline_unsupported_system_ids) {
        auto header = std::stringstream();
        header << "% Date 2014-02-28 13:37:42" << std::endl
               << "% system_ID " << system_id << std::endl
               << "% serial_number 00001337" << std::endl;
        write_header(RawFileHeader(header));

        std::unique_ptr<Device> device;
        EXPECT_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path_), HalException);
    }
}

TEST_WITH_CAMERA(DeviceDiscoveryRepositoryNoF_GTest, open_camera_check_facilities_existence,
                 camera_params(camera_param().integrator("Prophesee").generation("3.0"),
                               camera_param().integrator("Prophesee").generation("3.1").board("fx3"),
                               camera_param().integrator("Prophesee").generation("3.1").board("cx3"),
                               camera_param().generation("4.0"))) {
    std::unique_ptr<Device> device;
    try {
        device = DeviceDiscovery::open("");
    } catch (const HalException &) {
        std::cerr << "Plug a camera to run this test." << std::endl;
        FAIL();
    }

    ASSERT_NE(nullptr, device.get());

    // Check board facilities presence
    ASSERT_NE(nullptr, device->get_facility<I_HW_Identification>());
    ASSERT_NE(nullptr, device->get_facility<I_EventsStream>());
    ASSERT_NE(nullptr, device->get_facility<I_Geometry>());
    ASSERT_NE(nullptr, device->get_facility<I_LL_Biases>());
    ASSERT_NE(nullptr, device->get_facility<I_CameraSynchronization>());
    ASSERT_NE(nullptr, device->get_facility<I_TriggerIn>());
    ASSERT_NE(nullptr, device->get_facility<I_TriggerOut>());
    ASSERT_NE(nullptr, device->get_facility<I_ROI>());

    // check others facilities presence
    ASSERT_NE(nullptr, device->get_facility<I_PluginSoftwareInfo>());
    ASSERT_NE(nullptr, device->get_facility<I_EventsStream>());
    ASSERT_NE(nullptr, device->get_facility<I_EventsStreamDecoder>());
    ASSERT_NE(nullptr, device->get_facility<I_EventDecoder<EventCD>>());
    ASSERT_NE(nullptr, device->get_facility<I_EventDecoder<EventExtTrigger>>());
}

TEST_WITH_CAMERA(DeviceDiscoveryRepositoryNoF_GTest, open_camera_check_facilities_existence_no_triggers,
                 camera_params(camera_param().integrator("Prophesee").generation("3.1").board("evk2"),
                               camera_param().integrator("Prophesee").generation("3.1").board("evk3"))) {
    std::unique_ptr<Device> device;
    try {
        device = DeviceDiscovery::open("");
    } catch (const HalException &) {
        std::cerr << "Plug a camera to run this test." << std::endl;
        FAIL();
    }

    ASSERT_NE(nullptr, device.get());

    // Check board facilities presence
    ASSERT_NE(nullptr, device->get_facility<I_HW_Identification>());
    ASSERT_NE(nullptr, device->get_facility<I_EventsStream>());
    ASSERT_NE(nullptr, device->get_facility<I_Geometry>());
    ASSERT_NE(nullptr, device->get_facility<I_LL_Biases>());
    ASSERT_NE(nullptr, device->get_facility<I_CameraSynchronization>());
    ASSERT_NE(nullptr, device->get_facility<I_ROI>());

    // check others facilities presence
    ASSERT_NE(nullptr, device->get_facility<I_PluginSoftwareInfo>());
    ASSERT_NE(nullptr, device->get_facility<I_EventsStreamDecoder>());
    ASSERT_NE(nullptr, device->get_facility<I_EventDecoder<EventCD>>());
    ASSERT_NE(nullptr, device->get_facility<I_EventDecoder<EventExtTrigger>>());
}

TEST_WITH_CAMERA(DeviceDiscoveryRepositoryNoF_GTest, open_camera_build_gen3,
                 camera_params(camera_param().integrator("Prophesee").generation("3.0"))) {
    std::unique_ptr<Device> device;
    try {
        device = DeviceDiscovery::open("");
    } catch (const HalException &) {
        std::cerr << "Plug a camera to run this test." << std::endl;
        FAIL();
    }

    ASSERT_NE(nullptr, device.get());

    // Assert that the needed facilities for this test exist
    I_EventsStreamDecoder *decoder = device->get_facility<I_EventsStreamDecoder>();
    ASSERT_NE(nullptr, decoder);
    I_Geometry *geometry = device->get_facility<I_Geometry>();
    ASSERT_NE(nullptr, geometry);

    // Now check the information on the device
    ASSERT_EQ(VGAGeometry::width_, geometry->get_width());
    ASSERT_EQ(VGAGeometry::height_, geometry->get_height());
    ASSERT_EQ(4, decoder->get_raw_event_size_bytes());
}

TEST_WITH_CAMERA(DeviceDiscoveryRepositoryNoF_GTest, open_camera_build_gen31,
                 camera_params(camera_param().integrator("Prophesee").generation("3.1"))) {
    std::unique_ptr<Device> device;
    try {
        device = DeviceDiscovery::open("");
    } catch (const HalException &) {
        std::cerr << "Plug a camera to run this test." << std::endl;
        FAIL();
    }

    ASSERT_NE(nullptr, device.get());

    // Assert that the needed facilities for this test exist
    I_EventsStreamDecoder *decoder = device->get_facility<I_EventsStreamDecoder>();
    ASSERT_NE(nullptr, decoder);
    I_Geometry *geometry = device->get_facility<I_Geometry>();
    ASSERT_NE(nullptr, geometry);
    I_HW_Identification *hw_id_ = device->get_facility<Metavision::I_HW_Identification>();
    ASSERT_NE(nullptr, hw_id_);

    // Now check the information on the device
    ASSERT_NE(nullptr, device->get_facility<I_HW_Register>());
    ASSERT_NE(nullptr, device->get_facility<I_Monitoring>());
    ASSERT_NE(nullptr, device->get_facility<I_EventRateActivityFilterModule>());
    ASSERT_EQ(VGAGeometry::width_, geometry->get_width());
    ASSERT_EQ(VGAGeometry::height_, geometry->get_height());
    // The format may be either evt 2.0 or 3.0, this information was already used to spawn
    // a decoder, and it worked, there is no point re-checking it
}

TEST_WITH_CAMERA(DeviceDiscoveryRepositoryNoF_GTest, open_camera_build_gen4,
                 camera_params(camera_param().integrator("Prophesee").generation("4.0"))) {
    std::unique_ptr<Device> device;
    try {
        device = DeviceDiscovery::open("");
    } catch (const HalException &) {
        std::cerr << "Plug a camera to run this test." << std::endl;
        FAIL();
    }

    ASSERT_NE(nullptr, device.get());

    // Assert that the needed facilities for this test exist
    I_EventsStreamDecoder *decoder = device->get_facility<I_EventsStreamDecoder>();
    ASSERT_NE(nullptr, decoder);
    I_Geometry *geometry = device->get_facility<I_Geometry>();
    ASSERT_NE(nullptr, geometry);

    // Now check the information on the device
    ASSERT_NE(nullptr, device->get_facility<I_HW_Register>());
    ASSERT_NE(nullptr, device->get_facility<I_Monitoring>());
    ASSERT_EQ(HDGeometry::width_, geometry->get_width());
    ASSERT_EQ(HDGeometry::height_, geometry->get_height());
    ASSERT_EQ(2, decoder->get_raw_event_size_bytes());
}
