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

#include <vector>
#include <numeric>
#include <atomic>
#include <chrono>
#include <condition_variable>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/utils/gtest/gtest_custom.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/facilities/i_geometry.h"
#include "metavision/hal/facilities/i_device_control.h"
#include "metavision/hal/facilities/i_roi.h"
#include "metavision/hal/facilities/i_decoder.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"

using namespace Metavision;

class I_ROI_GTest : public GTestWithTmpDir {};

TEST_F_WITH_CAMERA(I_ROI_GTest, roi_columns_lines_with_camera) {
    std::unique_ptr<Device> device;
    try {
        device = DeviceDiscovery::open("");
    } catch (const HalException &e) {
        std::cerr << "Plug a camera to run this test." << std::endl;
        FAIL();
    }

    ASSERT_NE(nullptr, device.get());

    // Check geometry
    I_Geometry *geometry = device->get_facility<I_Geometry>();
    ASSERT_NE(nullptr, geometry);

    // Check board facilities presence
    auto device_control = device->get_facility<I_DeviceControl>();
    auto roi            = device->get_facility<I_ROI>();
    auto es             = device->get_facility<I_EventsStream>();
    auto decoder        = device->get_facility<I_Decoder>();
    auto cd_decoder     = device->get_facility<I_EventDecoder<EventCD>>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, device_control);
    ASSERT_NE(nullptr, roi);
    ASSERT_NE(nullptr, es);

    std::vector<bool> rows_to_enable(geometry->get_height(), true), cols_to_enable(geometry->get_width(), true);

    std::vector<size_t> rows_disabled(100, 0);
    std::vector<size_t> cols_disabled(100, 0);

    std::iota(rows_disabled.begin(), rows_disabled.begin() + 50, 20);  // disabled line from 20 to 69
    std::iota(rows_disabled.begin() + 50, rows_disabled.end(), 124);   // disabled line from 124 to 173
    std::iota(cols_disabled.begin(), cols_disabled.begin() + 50, 100); // disabled cols from 100 to 149
    std::iota(cols_disabled.begin() + 50, cols_disabled.end(), 200);   // disabled cols from 200 to 249

    for (auto row_to_disable : rows_disabled) {
        rows_to_enable[row_to_disable] = false; // disable the line
    }

    for (auto col_to_disable : cols_disabled) {
        cols_to_enable[col_to_disable] = false; // disable the line
    }

    std::atomic<size_t> n_cd_counts{0};
    static constexpr size_t max_cd_count = 1000;
    bool roi_set                         = false;
    cd_decoder->add_event_buffer_callback([&](const EventCD *it_begin, const EventCD *it_end) {
        n_cd_counts += (it_end - it_begin);
        for (; it_begin != it_end; ++it_begin) {
            for (auto row_to_disable : rows_disabled) {
                EXPECT_TRUE((it_begin->y != row_to_disable));
            }

            for (auto col_to_disable : cols_disabled) {
                EXPECT_TRUE((it_begin->x != col_to_disable));
            }
        }
    });

    roi->set_ROIs(cols_to_enable, rows_to_enable);

    es->start();
    device_control->start();

    const auto tnow = std::chrono::system_clock::now();
    while (n_cd_counts < max_cd_count) {
        if (es->poll_buffer() < 0) {
            std::cerr << "Test failed prematurely: events stream failed to poll data." << std::endl;
            FAIL();
        }

        long n_bytes;
        auto data = es->get_latest_raw_data(n_bytes);

        if (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - tnow).count() <
            1000000) {
            // Flush first events
            continue;
        }
        decoder->decode(data, data + n_bytes);
    }
}

TEST_F_WITH_CAMERA(I_ROI_GTest, roi_rectangle_with_camera) {
    std::unique_ptr<Device> device;
    try {
        device = DeviceDiscovery::open("");
    } catch (const HalException &e) {
        std::cerr << "Plug a camera to run this test." << std::endl;
        FAIL();
    }

    ASSERT_NE(nullptr, device.get());

    // Check geometry
    I_Geometry *geometry = device->get_facility<I_Geometry>();
    ASSERT_NE(nullptr, geometry);

    // Check board facilities presence
    auto device_control = device->get_facility<I_DeviceControl>();
    auto roi            = device->get_facility<I_ROI>();
    auto es             = device->get_facility<I_EventsStream>();
    auto decoder        = device->get_facility<I_Decoder>();
    auto cd_decoder     = device->get_facility<I_EventDecoder<EventCD>>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, device_control);
    ASSERT_NE(nullptr, roi);
    ASSERT_NE(nullptr, es);

    DeviceRoi roi_to_set(10, 100, 10, 100);

    std::atomic<size_t> n_cd_counts{0};
    static constexpr size_t max_cd_count = 1000;
    bool roi_set                         = false;
    cd_decoder->add_event_buffer_callback([&](const EventCD *it_begin, const EventCD *it_end) {
        n_cd_counts += (it_end - it_begin);
        for (; it_begin != it_end; ++it_begin) {
            EXPECT_GE(it_begin->x, roi_to_set.x_);
            EXPECT_LT(it_begin->x, roi_to_set.x_ + roi_to_set.width_);
            EXPECT_GE(it_begin->y, roi_to_set.y_);
            EXPECT_LT(it_begin->y, roi_to_set.y_ + roi_to_set.height_);
        }
    });

    roi->set_ROI(roi_to_set);

    es->start();
    device_control->start();

    const auto tnow = std::chrono::system_clock::now();
    while (n_cd_counts < max_cd_count) {
        if (es->poll_buffer() < 0) {
            std::cerr << "Test failed prematurely: events stream failed to poll data." << std::endl;
            FAIL();
        }

        long n_bytes;
        auto data = es->get_latest_raw_data(n_bytes);

        if (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - tnow).count() <
            1000000) {
            // Flush first events
            continue;
        }
        decoder->decode(data, data + n_bytes);
    }
}

TEST_F_WITH_CAMERA(I_ROI_GTest, several_roi_rectangle_with_camera) {
    std::unique_ptr<Device> device;
    try {
        device = DeviceDiscovery::open("");
    } catch (const HalException &e) {
        std::cerr << "Plug a camera to run this test." << std::endl;
        FAIL();
    }

    ASSERT_NE(nullptr, device.get());

    // Check geometry
    I_Geometry *geometry = device->get_facility<I_Geometry>();
    ASSERT_NE(nullptr, geometry);

    // Check board facilities presence
    auto device_control = device->get_facility<I_DeviceControl>();
    auto roi            = device->get_facility<I_ROI>();
    auto es             = device->get_facility<I_EventsStream>();
    auto decoder        = device->get_facility<I_Decoder>();
    auto cd_decoder     = device->get_facility<I_EventDecoder<EventCD>>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, device_control);
    ASSERT_NE(nullptr, roi);
    ASSERT_NE(nullptr, es);

    std::vector<DeviceRoi> rois_to_set{{10, 10, 20, 20},
                                       {100, 100, 20, 20}}; // 2 ROIs creates 4 rois on the sensor (AND operator)

    std::atomic<size_t> n_cd_counts{0};
    static constexpr size_t max_cd_count = 1000;
    bool roi_set                         = false;
    cd_decoder->add_event_buffer_callback([&](const EventCD *it_begin, const EventCD *it_end) {
        n_cd_counts += (it_end - it_begin);
        for (; it_begin != it_end; ++it_begin) {
            const bool x_good =
                (it_begin->x >= rois_to_set[0].x_ && it_begin->x < rois_to_set[0].x_ + rois_to_set[0].width_) ||
                (it_begin->x >= rois_to_set[1].x_ && it_begin->x < rois_to_set[0].x_ + rois_to_set[0].width_) ||
                (it_begin->x >= rois_to_set[0].x_ && it_begin->x < rois_to_set[1].x_ + rois_to_set[1].width_) ||
                (it_begin->x >= rois_to_set[1].x_ && it_begin->x < rois_to_set[1].x_ + rois_to_set[1].width_);
            const bool y_good =
                (it_begin->y >= rois_to_set[0].y_ && it_begin->y < rois_to_set[0].y_ + rois_to_set[0].height_) ||
                (it_begin->y >= rois_to_set[1].y_ && it_begin->y < rois_to_set[0].y_ + rois_to_set[0].height_) ||
                (it_begin->y >= rois_to_set[0].y_ && it_begin->y < rois_to_set[1].y_ + rois_to_set[1].height_) ||
                (it_begin->y >= rois_to_set[1].y_ && it_begin->y < rois_to_set[1].y_ + rois_to_set[1].height_);

            EXPECT_TRUE(x_good && y_good);
        }
    });

    roi->set_ROIs(rois_to_set);

    es->start();
    device_control->start();

    const auto tnow = std::chrono::system_clock::now();
    while (n_cd_counts < max_cd_count) {
        if (es->poll_buffer() < 0) {
            std::cerr << "Test failed prematurely: events stream failed to poll data." << std::endl;
            FAIL();
        }

        long n_bytes;
        auto data = es->get_latest_raw_data(n_bytes);

        if (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - tnow).count() <
            1000000) {
            // Flush first events
            continue;
        }
        decoder->decode(data, data + n_bytes);
    }
}
