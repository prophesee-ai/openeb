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

#include <memory>
#include <atomic>

#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/utils/gtest/gtest_custom.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/facilities/i_camera_synchronization.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_geometry.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "sample_hw_identification.h"
#include "sample_geometry.h"

using namespace Metavision;

class HalToySamplePlugin_GTest : public GTestWithTmpDir {
public:
    // Check facilities that should be present both online and offline
    void check_common_facilities(Metavision::Device *device, bool offline) {
        // I_HW_Identification
        Metavision::I_HW_Identification *i_hw_identification = device->get_facility<Metavision::I_HW_Identification>();
        ASSERT_NE(nullptr, i_hw_identification);
        ASSERT_EQ(SampleHWIdentification::SAMPLE_SERIAL, i_hw_identification->get_serial());
        ASSERT_EQ("Gen1.0", i_hw_identification->get_sensor_info().name_);
        std::vector<std::string> available_formats = i_hw_identification->get_available_data_encoding_formats();
        ASSERT_EQ(1, available_formats.size());
        ASSERT_EQ("SAMPLE-FORMAT-1.0", available_formats[0]);
        ASSERT_EQ("SAMPLE-FORMAT-1.0", i_hw_identification->get_current_data_encoding_format());
        ASSERT_EQ(SampleHWIdentification::SAMPLE_INTEGRATOR, i_hw_identification->get_integrator());
        if (offline) {
            ASSERT_EQ("File", i_hw_identification->get_connection_type());
        } else {
            ASSERT_EQ("USB", i_hw_identification->get_connection_type());
        }

        // I_Geometry
        Metavision::I_Geometry *i_geometry = device->get_facility<Metavision::I_Geometry>();
        ASSERT_NE(nullptr, i_geometry);
        ASSERT_EQ(SampleGeometry::WIDTH_, i_geometry->get_width());
        ASSERT_EQ(SampleGeometry::HEIGHT_, i_geometry->get_height());

        // I_EventsStreamDecoder
        Metavision::I_EventsStreamDecoder *i_eventsstreamdecoder =
            device->get_facility<Metavision::I_EventsStreamDecoder>();
        ASSERT_NE(nullptr, i_eventsstreamdecoder);
        if (offline) {
            ASSERT_TRUE(i_eventsstreamdecoder->is_time_shifting_enabled());
        } else {
            ASSERT_FALSE(i_eventsstreamdecoder->is_time_shifting_enabled());
        }

        // I_EventDecoder<Metavision::EventCD>
        Metavision::I_EventDecoder<Metavision::EventCD> *i_cd_decoder =
            device->get_facility<Metavision::I_EventDecoder<Metavision::EventCD>>();
        ASSERT_NE(nullptr, i_cd_decoder);

        // I_EventsStream
        Metavision::I_EventsStream *i_events_stream = device->get_facility<Metavision::I_EventsStream>();
        ASSERT_NE(nullptr, i_events_stream);
    }
};

TEST_F_WITHOUT_CAMERA(HalToySamplePlugin_GTest, list_sources_and_open_it) {
    // GIVEN the sample plugin library
    // WHEN we get the list of available sources
    auto v = Metavision::DeviceDiscovery::list();

    // THEN we get exactly one source, with specific serial
    ASSERT_EQ(1, v.size());
    std::string full_serial_expected = std::string(SampleHWIdentification::SAMPLE_INTEGRATOR) +
                                       ":hal_toy_sample_plugin:" + std::string(SampleHWIdentification::SAMPLE_SERIAL);
    ASSERT_EQ(full_serial_expected, v.front());

    // GIVEN the sample plugin library
    // WHEN we create a device with the given serial
    std::unique_ptr<Metavision::Device> device;
    ASSERT_NO_THROW(device = DeviceDiscovery::open(v.front()));

    // THEN a valid device is built
    ASSERT_NE(nullptr, device);
}

TEST_F(HalToySamplePlugin_GTest, open_first_available_live_source_and_check_facilities) {
    // GIVEN the sample plugin library
    // WHEN we create a device from first available
    std::unique_ptr<Metavision::Device> device;
    ASSERT_NO_THROW(device = DeviceDiscovery::open(""));

    // THEN a valid device is built, and it has the expected facilities
    ASSERT_NE(nullptr, device);

    check_common_facilities(device.get(), false);

    // I_DeviceControl
    Metavision::I_CameraSynchronization *i_camera_synchronization =
        device->get_facility<Metavision::I_CameraSynchronization>();
    ASSERT_NE(nullptr, i_camera_synchronization);
}

TEST_F(HalToySamplePlugin_GTest, record_and_read_back) {
    // GIVEN the sample plugin library

    // WHEN we record from live source
    std::unique_ptr<Metavision::Device> device(DeviceDiscovery::open(""));

    Metavision::I_EventsStream *i_events_stream = device->get_facility<Metavision::I_EventsStream>();
    Metavision::I_EventsStreamDecoder *i_eventsstreamdecoder =
        device->get_facility<Metavision::I_EventsStreamDecoder>();
    Metavision::I_EventDecoder<Metavision::EventCD> *i_cd_decoder =
        device->get_facility<Metavision::I_EventDecoder<Metavision::EventCD>>();

    std::atomic<int> n_cd_events_decoded(0);
    i_cd_decoder->add_event_buffer_callback(
        [&n_cd_events_decoded](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
            n_cd_events_decoded += std::distance(begin, end);
        });

    std::string rawfile_to_log_path = tmpdir_handler_->get_full_path("rawfile_sample_plugin.raw");
    i_events_stream->log_raw_data(rawfile_to_log_path);
    i_events_stream->start();
    while (n_cd_events_decoded < 1000) { // To be sure to record something
        short ret = i_events_stream->wait_next_buffer();
        ASSERT_LE(0, ret);

        auto raw_data = i_events_stream->get_latest_raw_data();
        i_eventsstreamdecoder->decode(raw_data);
    }
    Metavision::timestamp last_time = i_eventsstreamdecoder->get_last_timestamp();

    // THEN when reading back the recording, we get the same events as the first run
    device.reset(nullptr);
    ASSERT_NO_THROW(device = DeviceDiscovery::open_raw_file(rawfile_to_log_path));
    ASSERT_NE(nullptr, device);

    // Check the facilities
    check_common_facilities(device.get(), true);

    // Reset counter and facilities
    int number_cd_expected = n_cd_events_decoded;
    n_cd_events_decoded    = 0;

    i_events_stream       = device->get_facility<Metavision::I_EventsStream>();
    i_eventsstreamdecoder = device->get_facility<Metavision::I_EventsStreamDecoder>();
    i_cd_decoder          = device->get_facility<Metavision::I_EventDecoder<Metavision::EventCD>>();
    i_cd_decoder->add_event_buffer_callback(
        [&n_cd_events_decoded](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
            n_cd_events_decoded += std::distance(begin, end);
        });
    i_events_stream->start();
    short ret = i_events_stream->wait_next_buffer();
    while (ret > 0) { // To be sure to record something
        auto raw_data = i_events_stream->get_latest_raw_data();
        i_eventsstreamdecoder->decode(raw_data);
        ret = i_events_stream->wait_next_buffer();
    }
    ASSERT_EQ(number_cd_expected, n_cd_events_decoded);
    ASSERT_EQ(last_time, i_eventsstreamdecoder->get_last_timestamp());
}
