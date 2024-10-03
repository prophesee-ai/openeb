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

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/utils/gtest/gtest_custom.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/facilities/i_camera_synchronization.h"
#include "metavision/hal/facilities/i_geometry.h"
#include "metavision/hal/facilities/i_roi.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"

#include "utils/device_test.h"

using namespace Metavision;

class I_ROI_GTest : public Metavision::testing::DeviceTest {
protected:
    void on_opened_device(Metavision::Device &device) override {
        ASSERT_NE(nullptr, device_.get());

        geometry               = device_->get_facility<I_Geometry>();
        es                     = device_->get_facility<I_EventsStream>();
        decoder                = device_->get_facility<I_EventsStreamDecoder>();
        cd_decoder             = device_->get_facility<I_EventDecoder<EventCD>>();
        camera_synchronization = device_->get_facility<I_CameraSynchronization>();
        roi                    = device_->get_facility<I_ROI>();

        // Check geometry
        ASSERT_NE(nullptr, geometry);

        // Check board facilities presence
        ASSERT_NE(nullptr, decoder);
        ASSERT_NE(nullptr, cd_decoder);
        ASSERT_NE(nullptr, camera_synchronization);
        ASSERT_NE(nullptr, roi);
        ASSERT_NE(nullptr, es);
    }

    I_Geometry *geometry                            = nullptr;
    I_EventsStream *es                              = nullptr;
    I_EventsStreamDecoder *decoder                  = nullptr;
    I_EventDecoder<EventCD> *cd_decoder             = nullptr;
    I_CameraSynchronization *camera_synchronization = nullptr;
    I_ROI *roi                                      = nullptr;
};

TEST_F_WITH_CAMERA(I_ROI_GTest, roi_columns_lines_with_camera) {
    std::vector<bool> rows(geometry->get_height(), true), cols(geometry->get_width(), true);

    std::vector<size_t> rows_disabled(100, 0);
    std::vector<size_t> cols_disabled(100, 0);

    std::iota(rows_disabled.begin(), rows_disabled.begin() + 50, 20);  // disabled line from 20 to 69
    std::iota(rows_disabled.begin() + 50, rows_disabled.end(), 124);   // disabled line from 124 to 173
    std::iota(cols_disabled.begin(), cols_disabled.begin() + 50, 100); // disabled cols from 100 to 149
    std::iota(cols_disabled.begin() + 50, cols_disabled.end(), 200);   // disabled cols from 200 to 249

    for (auto row_to_disable : rows_disabled) {
        rows[row_to_disable] = false; // disable the line
    }

    for (auto col_to_disable : cols_disabled) {
        cols[col_to_disable] = false; // disable the line
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

    roi->set_lines(cols, rows);

    es->start();

    const auto tnow = std::chrono::system_clock::now();
    while (n_cd_counts < max_cd_count) {
        if (es->poll_buffer() < 0) {
            std::cerr << "Test failed prematurely: events stream failed to poll data." << std::endl;
            FAIL();
        }

        auto buffer = es->get_latest_raw_data();

        if (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - tnow).count() <
            1000000) {
            // Flush first events
            continue;
        }
        decoder->decode(buffer);
    }
}

TEST_F_WITH_CAMERA(I_ROI_GTest, roi_rectangle_with_camera) {
    I_ROI::Window roi_to_set(10, 100, 10, 100);

    std::atomic<size_t> n_cd_counts{0};
    static constexpr size_t max_cd_count = 1000;
    bool roi_set                         = false;
    cd_decoder->add_event_buffer_callback([&](const EventCD *it_begin, const EventCD *it_end) {
        n_cd_counts += (it_end - it_begin);
        for (; it_begin != it_end; ++it_begin) {
            EXPECT_GE(it_begin->x, roi_to_set.x);
            EXPECT_LT(it_begin->x, roi_to_set.x + roi_to_set.width);
            EXPECT_GE(it_begin->y, roi_to_set.y);
            EXPECT_LT(it_begin->y, roi_to_set.y + roi_to_set.height);
        }
    });

    roi->set_window(roi_to_set);

    es->start();

    const auto tnow = std::chrono::system_clock::now();
    while (n_cd_counts < max_cd_count) {
        if (es->poll_buffer() < 0) {
            std::cerr << "Test failed prematurely: events stream failed to poll data." << std::endl;
            FAIL();
        }

        auto buffer = es->get_latest_raw_data();

        if (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - tnow).count() <
            1000000) {
            // Flush first events
            continue;
        }
        decoder->decode(buffer);
    }
}

TEST_F_WITH_CAMERA(I_ROI_GTest, several_roi_rectangle_with_camera) {
    std::vector<I_ROI::Window> rois_to_set{{10, 10, 20, 20},
                                           {100, 100, 20, 20}}; // 2 ROIs creates 4 rois on the sensor (AND operator)

    std::atomic<size_t> n_cd_counts{0};
    static constexpr size_t max_cd_count = 1000;
    bool roi_set                         = false;
    cd_decoder->add_event_buffer_callback([&](const EventCD *it_begin, const EventCD *it_end) {
        n_cd_counts += (it_end - it_begin);
        for (; it_begin != it_end; ++it_begin) {
            const bool x_good =
                (it_begin->x >= rois_to_set[0].x && it_begin->x < rois_to_set[0].x + rois_to_set[0].width) ||
                (it_begin->x >= rois_to_set[1].x && it_begin->x < rois_to_set[0].x + rois_to_set[0].width) ||
                (it_begin->x >= rois_to_set[0].x && it_begin->x < rois_to_set[1].x + rois_to_set[1].width) ||
                (it_begin->x >= rois_to_set[1].x && it_begin->x < rois_to_set[1].x + rois_to_set[1].width);
            const bool y_good =
                (it_begin->y >= rois_to_set[0].y && it_begin->y < rois_to_set[0].y + rois_to_set[0].height) ||
                (it_begin->y >= rois_to_set[1].y && it_begin->y < rois_to_set[0].y + rois_to_set[0].height) ||
                (it_begin->y >= rois_to_set[0].y && it_begin->y < rois_to_set[1].y + rois_to_set[1].height) ||
                (it_begin->y >= rois_to_set[1].y && it_begin->y < rois_to_set[1].y + rois_to_set[1].height);

            EXPECT_TRUE(x_good && y_good);
        }
    });

    roi->set_windows(rois_to_set);

    es->start();

    const auto tnow = std::chrono::system_clock::now();
    while (n_cd_counts < max_cd_count) {
        if (es->poll_buffer() < 0) {
            std::cerr << "Test failed prematurely: events stream failed to poll data." << std::endl;
            FAIL();
        }

        auto buffer = es->get_latest_raw_data();

        if (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - tnow).count() <
            1000000) {
            // Flush first events
            continue;
        }
        decoder->decode(buffer);
    }
}
