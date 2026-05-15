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

#include "metavision/hal/facilities/i_ll_biases.h"
#include "metavision/hal/facilities/i_monitoring.h"
#include "metavision/utils/gtest/gtest_custom.h"

#include "utils/device_test.h"
#include <gtest/gtest.h>

using namespace Metavision;
using namespace ::testing;
using Metavision::testing::DeviceTest;

class I_Monitoring_GTest : public DeviceTest {
protected:
    I_Monitoring *monitoring = nullptr;
    I_LL_Biases *ll_biases   = nullptr;

    void on_opened_device(Device &device) override {
        monitoring = device.get_facility<I_Monitoring>();
        ASSERT_NE(nullptr, monitoring);

        ll_biases = device.get_facility<I_LL_Biases>();
        ASSERT_NE(nullptr, ll_biases);
    }
};

TEST_F_WITH_CAMERA(I_Monitoring_GTest, should_NOT_have_monitoring_implemented,
                   camera_params(camera_param().generation("3.0"))) {
    EXPECT_FALSE(monitoring);
}

TEST_F_WITH_CAMERA(I_Monitoring_GTest, should_have_monitoring_implemented,
                   camera_params(camera_param().generation("3.1"), camera_param().generation("4.0"),
                                 camera_param().generation("4.1"), camera_param().generation("4.2"))) {
    EXPECT_TRUE(monitoring);
}

TEST_F_WITH_CAMERA(I_Monitoring_GTest, should_gen31_get_temperature_from_monitoring,
                   camera_params(camera_param().generation("3.1"))) {
    ASSERT_TRUE(monitoring);

    EXPECT_GE(monitoring->get_temperature(), 15);
    EXPECT_LT(monitoring->get_temperature(), 120);
}

TEST_F_WITH_CAMERA(I_Monitoring_GTest, should_gen41_get_temperature_from_monitoring,
                   camera_params(camera_param().generation("4.1"), camera_param().generation("4.2"))) {
    ASSERT_TRUE(monitoring);

    EXPECT_GE(monitoring->get_temperature(), -20);
    EXPECT_LT(monitoring->get_temperature(), 60);
}

TEST_F_WITH_CAMERA(I_Monitoring_GTest, should_gen31_get_illumination_from_monitoring,
                   camera_params(camera_param().generation("3.1"))) {
    ASSERT_TRUE(monitoring);

    // First measure gives invalid values. Looks like an unitiliazed value within the fpga or
    // something.
    monitoring->get_illumination();

    for (uint32_t test = 0; test < 5; ++test) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        EXPECT_GE(monitoring->get_illumination(), 0); // Can be 0 in dark scene
        EXPECT_LT(monitoring->get_illumination(), 100);
    }
}

TEST_F_WITH_CAMERA(I_Monitoring_GTest, should_gen41_get_illumination_from_monitoring,
                   camera_params(camera_param().generation("4.1"), camera_param().generation("4.2"))) {
    ASSERT_TRUE(monitoring);

    EXPECT_GE(monitoring->get_illumination(), 0);
    EXPECT_LT(monitoring->get_illumination(), 10'000);
}

TEST_F_WITH_CAMERA(I_Monitoring_GTest, should_get_pixel_dead_time_from_monitoring,
                   camera_params(camera_param().generation("4.2"))) {
    ASSERT_TRUE(monitoring);

    EXPECT_GE(monitoring->get_pixel_dead_time(), 0);
    EXPECT_LT(monitoring->get_pixel_dead_time(), 100'000);
}

TEST_F_WITH_CAMERA(I_Monitoring_GTest, should_get_pixel_dead_time_inline_with_bias_refr,
                   camera_params(camera_param().generation("4.2"))) {
    ASSERT_TRUE(monitoring);
    ASSERT_TRUE(ll_biases);

    ll_biases->set("bias_refr", -20);
    int minus_20_refr_dead_time = monitoring->get_pixel_dead_time();

    ll_biases->set("bias_refr", 0);
    int _0_refr_dead_time = monitoring->get_pixel_dead_time();

    ll_biases->set("bias_refr", 100);
    int _100_refr_dead_time = monitoring->get_pixel_dead_time();

    // The smaller the refr, the smaller the pixel dead time
    EXPECT_GT(minus_20_refr_dead_time, _0_refr_dead_time);
    EXPECT_GT(_0_refr_dead_time, _100_refr_dead_time);
}
