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

#include <string>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "metavision/hal/device/device.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/facilities/i_antiflicker_module.h"
#include "metavision/hal/facilities/i_camera_synchronization.h"
#include "metavision/hal/facilities/i_erc_module.h"
#include "metavision/hal/facilities/i_event_rate_activity_filter_module.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/facilities/i_digital_crop.h"
#include "metavision/hal/facilities/i_digital_event_mask.h"
#include "metavision/hal/facilities/i_event_trail_filter_module.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_hw_register.h"
#include "metavision/hal/facilities/i_ll_biases.h"
#include "metavision/hal/facilities/i_monitoring.h"
#include "metavision/hal/facilities/i_roi.h"
#include "metavision/hal/facilities/i_trigger_in.h"
#include "metavision/hal/facilities/i_trigger_out.h"

#include "dummy_test_plugin_facilities.h"

using namespace Metavision;
using namespace ::testing;

const std::string dummy_test_plugin_name = "__DummyTest__";

TEST(DummyTestPlugin, should_device_discovery_list_dummytest_plugin) {
    EXPECT_THAT(DeviceDiscovery::list(), Contains(HasSubstr(dummy_test_plugin_name)));
}

class DummyTestPluginTest : public Test {
public:
    void SetUp() override {
        dummy_device = DeviceDiscovery::open(dummy_test_plugin_name);
        ASSERT_THAT(dummy_device.get(), NotNull());
    }

    std::unique_ptr<Device> dummy_device = nullptr;
};

TEST_F(DummyTestPluginTest, should_have_facilities) {
    EXPECT_THAT(dummy_device->get_facility<I_AntiFlickerModule>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<I_CameraSynchronization>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<I_DigitalEventMask>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<I_DigitalCrop>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<I_ErcModule>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<I_EventRateActivityFilterModule>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<I_EventTrailFilterModule>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<I_EventsStream>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<I_EventsStreamDecoder>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<I_HW_Identification>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<I_HW_Register>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<I_LL_Biases>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<I_Monitoring>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<I_ROI>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<I_TriggerIn>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<I_TriggerOut>(), NotNull());
}

TEST_F(DummyTestPluginTest, should_have_facilities_multi_version_facility) {
    EXPECT_THAT(dummy_device->get_facility<DummyFacilityV1>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<DummyFacilityV2>(), NotNull());
    EXPECT_THAT(dummy_device->get_facility<DummyFacilityV3>(), NotNull());
    EXPECT_EQ(dummy_device->get_facility<DummyFacilityV1>(), dummy_device->get_facility<DummyFacilityV2>());
    EXPECT_EQ(dummy_device->get_facility<DummyFacilityV2>(), dummy_device->get_facility<DummyFacilityV3>());
}

TEST_F(DummyTestPluginTest, should_stream) {
    // DummyRawDataProducer generates an incrementing pattern
    int counter       = 0;
    auto event_stream = dummy_device->get_facility<I_EventsStream>();
    EXPECT_THAT(event_stream, NotNull());

    event_stream->start();

    do {
        // DummyRawDataProducer sends 8 buffers of up to 128 bytes
        // it is assumed that this test will make DummyRawDataProducer run out of buffers
        // (it has a bounded pool and can't drop)
        // but this depends on runner's performances
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto buffer = event_stream->get_latest_raw_data();
        for (auto &data : buffer) {
            EXPECT_EQ(data, counter++);
        }
    } while (event_stream->poll_buffer() > 0);

    // DummyRawDataProducer sends 255 values
    EXPECT_EQ(counter, 255);
    event_stream->stop();
}

TEST_F(DummyTestPluginTest, can_copy_buffers_outside_of_the_pool) {
    auto event_stream = dummy_device->get_facility<I_EventsStream>();
    EXPECT_THAT(event_stream, NotNull());

    event_stream->start();

    event_stream->wait_next_buffer();
    auto buffer = event_stream->get_latest_raw_data();
    event_stream->stop();

    DataTransfer::BufferPtr outsider = buffer.clone();
    // A copy of a buffer shall be equal
    EXPECT_EQ(buffer, outsider);
}
