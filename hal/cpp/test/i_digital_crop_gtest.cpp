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

#include "metavision/hal/facilities/i_digital_crop.h"
#include "metavision/hal/facilities/i_geometry.h"
#include "metavision/utils/gtest/gtest_custom.h"

#include "utils/device_test.h"
#include <gtest/gtest.h>

using namespace Metavision;
using Metavision::testing::DeviceTest;

class I_DigitalCrop_GTest : public DeviceTest {
protected:
    I_DigitalCrop *digital_crop = nullptr;
    I_Geometry *geom            = nullptr;

    void on_opened_device(Device &device) {
        digital_crop = device.get_facility<I_DigitalCrop>();
        geom         = device_->get_facility<I_Geometry>();
    }
};

TEST_F_WITH_CAMERA(I_DigitalCrop_GTest, should_NOT_have_digital_crop_implemented,
                   camera_params(camera_param().generation("3.0"), camera_param().generation("3.1"),
                                 camera_param().generation("4.0"))) {
    ASSERT_FALSE(digital_crop);
}

TEST_F_WITH_CAMERA(I_DigitalCrop_GTest, should_be_enabling_digital_crop,
                   camera_params(camera_param().generation("4.1"), camera_param().generation("4.2"))) {
    ASSERT_TRUE(digital_crop);
    EXPECT_FALSE(digital_crop->is_enabled());

    EXPECT_TRUE(digital_crop->enable(true));
    EXPECT_TRUE(digital_crop->is_enabled());

    EXPECT_TRUE(digital_crop->enable(false));
    EXPECT_FALSE(digital_crop->is_enabled());
}

TEST_F_WITH_CAMERA(I_DigitalCrop_GTest, should_be_retreiving_configured_region,
                   camera_params(camera_param().generation("4.1"), camera_param().generation("4.2"))) {
    ASSERT_TRUE(digital_crop);

    EXPECT_TRUE(digital_crop->set_window_region({12, 23, 34, 45}, true));

    uint32_t start_x = 0, start_y = 0, end_x = 0, end_y = 0;
    std::tie(start_x, start_y, end_x, end_y) = digital_crop->get_window_region();

    EXPECT_EQ(start_x, 12);
    EXPECT_EQ(start_y, 23);
    EXPECT_EQ(end_x, 34);
    EXPECT_EQ(end_y, 45);
}

TEST_F_WITH_CAMERA(I_DigitalCrop_GTest, should_have_digital_crop_events_when_enabled,
                   camera_params(camera_param().generation("4.1"), camera_param().generation("4.2"))) {
    ASSERT_TRUE(geom);
    ASSERT_TRUE(digital_crop);
    auto height = geom->get_height();

    digital_crop->enable(true);
    digital_crop->set_window_region({32, 0, 64, height});

    EXPECT_TRUE(digital_crop->is_enabled());

    std::vector<EventCD> evt_inside_crop_region;
    std::vector<EventCD> evt_outside_crop_region;

    stream_n_buffers(10, [&](const EventCD *evt_beg, const EventCD *evt_end) {
        EXPECT_GT(std::distance(evt_beg, evt_end), 0)
            << "No events from the sensor. Whether scene is pitch black, or feature is broken";

        while (evt_beg != evt_end) {
            auto &evt_vector = evt_beg->x >= 32 && evt_beg->x <= 64 ? evt_inside_crop_region : evt_outside_crop_region;
            evt_vector.emplace_back(*evt_beg);
            evt_beg++;
        }
    });

    EXPECT_GT(evt_inside_crop_region.size(), 0) << "Crop region should contain all events";
    EXPECT_EQ(evt_outside_crop_region.size(), 0) << "Outside crop region should contain no event";
}

TEST_F_WITH_CAMERA(I_DigitalCrop_GTest, should_have_origin_reset_on_digital_crop_events_when_enabled,
                   camera_params(camera_param().generation("4.1"), camera_param().generation("4.2"))) {
    ASSERT_TRUE(geom);
    ASSERT_TRUE(digital_crop);
    auto height = geom->get_height();

    std::vector<EventCD> evt_inside_crop_region;
    std::vector<EventCD> evt_outside_crop_region;

    digital_crop->enable(true);
    digital_crop->set_window_region({64, 0, 64 + 32, height}, true);

    stream_n_buffers(100, [&](const EventCD *evt_beg, const EventCD *evt_end) {
        EXPECT_GT(std::distance(evt_beg, evt_end), 0)
            << "No events from the sensor. Whether scene is pitch black, or feature is broken";

        while (evt_beg != evt_end) {
            auto &evt_vector = evt_beg->x <= 32 ? evt_inside_crop_region : evt_outside_crop_region;
            evt_vector.emplace_back(*evt_beg);
            evt_beg++;
        }
    });

    EXPECT_GT(evt_inside_crop_region.size(), 0) << "Crop region should contain all events";
    EXPECT_EQ(evt_outside_crop_region.size(), 0) << "Outside crop region should contain no event";
}

TEST_F_WITH_CAMERA(I_DigitalCrop_GTest, should_have_digital_NOT_crop_events_when_disabled,
                   camera_params(camera_param().generation("4.1"), camera_param().generation("4.2"))) {
    ASSERT_TRUE(geom);
    ASSERT_TRUE(digital_crop);
    auto height = geom->get_height();

    digital_crop->enable(false);
    digital_crop->set_window_region({0, 0, 32, height});

    std::vector<EventCD> evt_inside_crop_region;
    std::vector<EventCD> evt_outside_crop_region;

    stream_n_buffers(100, [&](auto *evt_beg, const EventCD *evt_end) {
        EXPECT_GT(std::distance(evt_beg, evt_end), 0)
            << "No events from the sensor. Whether scene is pitch black, or feature is broken";

        while (evt_beg != evt_end) {
            auto &evt_vector = evt_beg->x < 32 ? evt_inside_crop_region : evt_outside_crop_region;
            evt_vector.emplace_back(*evt_beg);
            evt_beg++;
        }
    });

    EXPECT_GT(evt_inside_crop_region.size(), 0) << "Crop region isn't enabled, thus should contain some events";
    EXPECT_GT(evt_outside_crop_region.size(), 0) << "Outside crop region should contain some events";
}

TEST_F_WITH_CAMERA(I_DigitalCrop_GTest, should_raise_on_wrong_window_region,
                   camera_params(camera_param().generation("4.1"), camera_param().generation("4.2"))) {
    ASSERT_TRUE(digital_crop);

    EXPECT_ANY_THROW(digital_crop->set_window_region({200, 200, 0, 300}, true));
    EXPECT_ANY_THROW(digital_crop->set_window_region({200, 200, 300, 0}, true));
}
