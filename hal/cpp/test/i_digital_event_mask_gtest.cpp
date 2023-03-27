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

#include "metavision/hal/facilities/i_digital_event_mask.h"
#include "metavision/utils/gtest/gtest_custom.h"

#include "utils/device_test.h"
#include <gtest/gtest.h>

using namespace Metavision;
using namespace ::testing;
using Metavision::testing::DeviceTest;

class I_DigitalEventMask_Gtest : public DeviceTest {
protected:
    I_DigitalEventMask *digital_event_mask = nullptr;

    void on_opened_device(Device &device) override {
        digital_event_mask = device.get_facility<I_DigitalEventMask>();
    }
};

TEST_F_WITH_CAMERA(I_DigitalEventMask_Gtest, should_NOT_have_digital_event_mask_implemented,
                   camera_params(camera_param().generation("3.0"), camera_param().generation("3.1"),
                                 camera_param().generation("4.0"))) {
    ASSERT_FALSE(digital_event_mask);
}

TEST_F_WITH_CAMERA(I_DigitalEventMask_Gtest, should_have_digital_event_mask_implemented,
                   camera_params(camera_param().generation("4.1"), camera_param().generation("4.2"))) {
    ASSERT_TRUE(digital_event_mask);

    auto pixel_masks = digital_event_mask->get_pixel_masks();
    EXPECT_EQ(pixel_masks.size(), 64);

    // We mask the first 64 pixels of the first line (with y = 0)
    size_t i = 0;
    for (auto &pixel_mask : pixel_masks) {
        pixel_mask->set_mask(i++, 0, true);
    }

    // We expect that none of the masked pixels will be decoded
    stream_n_buffers(100, [](const EventCD *event_beg, const EventCD *event_end) {
        while (event_beg != event_end) {
            const EventCD &event_cd = *event_beg;
            EXPECT_FALSE(event_cd.x < 64 && event_cd.y == 0) << "Event should be masked : " << event_cd;
            event_beg++;
        }
    });
}
