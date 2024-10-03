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

#ifndef METAVISION_HAL_TEST_UTILS_DEVICE_TEST_H
#define METAVISION_HAL_TEST_UTILS_DEVICE_TEST_H

#include <gtest/gtest.h>

#include "metavision/hal/device/device.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/utils/gtest/gtest_with_tmp_dir.h"

namespace Metavision {
namespace testing {

class DeviceTest : public GTestWithTmpDir {
protected:
    virtual void on_opened_device(Metavision::Device &device) = 0;

    // Sets up the test fixture.
    void SetUp() {
        try {
            auto serial_list = Metavision::DeviceDiscovery::list();
            if (serial_list.empty()) {
                std::cerr << "No Device Found" << std::endl;
                FAIL();
            } else if (serial_list.size() > 1) {
                std::cerr << "WARNING: Several Cameras Plugged In" << std::endl;
            }

            device_ = Metavision::DeviceDiscovery::open("");
        } catch (const Metavision::HalException &) {
            std::cerr << "Plug a camera to run this test." << std::endl;
            FAIL();
        }

        ASSERT_TRUE(device_);

        events_stream         = device_->get_facility<I_EventsStream>();
        events_stream_decoder = device_->get_facility<I_EventsStreamDecoder>();
        event_cd_decoder      = device_->get_facility<I_EventDecoder<EventCD>>();

        on_opened_device(*device_);
    }

    void TearDown() {}

    void stream_n_buffers(int nb_buffers_to_process, const I_EventDecoder<EventCD>::EventBufferCallback_t &cb) {
        ASSERT_NE(event_cd_decoder, nullptr);
        ASSERT_NE(events_stream, nullptr);
        ASSERT_NE(events_stream_decoder, nullptr);

        auto cb_id = event_cd_decoder->add_event_buffer_callback(cb);
        events_stream->start();

        while (nb_buffers_to_process > 0) {
            if (events_stream->wait_next_buffer()) {
                auto raw_data = events_stream->get_latest_raw_data();
                events_stream_decoder->decode(raw_data);
                nb_buffers_to_process--;
            }
        }

        events_stream->stop();
        event_cd_decoder->remove_callback(cb_id);
    }

    std::unique_ptr<Metavision::Device> device_;
    I_EventsStream *events_stream                = nullptr;
    I_EventsStreamDecoder *events_stream_decoder = nullptr;
    I_EventDecoder<EventCD> *event_cd_decoder    = nullptr;
};

} // namespace testing
} // namespace Metavision

#endif // METAVISION_HAL_TEST_UTILS_DEVICE_TEST_H
