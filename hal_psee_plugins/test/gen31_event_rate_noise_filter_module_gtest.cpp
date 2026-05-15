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

#include <cmath>
#include <mutex>
#include <chrono>
#include <thread>

#include "metavision/utils/gtest/gtest_custom.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_event_rate_activity_filter_module.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/decoders/base/event_base.h"
#include "metavision/hal/decoders/evt2/evt2_event_types.h"
#include "metavision/psee_hw_layer/devices/gen31/gen31_event_rate_noise_filter_module.h"

using namespace Metavision;

class Gen31EventRateNoiseFilterModule_GTest : public ::testing::Test {
public:
    void TearDown() override {
        device_.reset(nullptr);
        ev_rate_ = nullptr;
    }

    void open() {
        try {
            device_ = DeviceDiscovery::open("");
        } catch (const HalException &) {
            std::cerr << "Plug a camera to run this test." << std::endl;
            FAIL();
            return;
        }

        ASSERT_NE(nullptr, device_.get());

        ev_rate_ = device_->get_facility<I_EventRateActivityFilterModule>();
        ASSERT_NE(nullptr, ev_rate_);
    }

protected:
    std::unique_ptr<Device> device_;
    I_EventRateActivityFilterModule *ev_rate_{nullptr};

    static constexpr uint32_t min_event_rate_threshold_kev_s_ =
        Gen31_EventRateNoiseFilterModule::min_event_rate_threshold_kev_s;
    static constexpr uint32_t max_event_rate_threshold_kev_s_ =
        Gen31_EventRateNoiseFilterModule::max_event_rate_threshold_kev_s;
    static constexpr uint32_t in_range_event_rate_threshold_kev_s_ =
        (max_event_rate_threshold_kev_s_ - min_event_rate_threshold_kev_s_) / 2;
};

constexpr uint32_t Gen31EventRateNoiseFilterModule_GTest::min_event_rate_threshold_kev_s_;
constexpr uint32_t Gen31EventRateNoiseFilterModule_GTest::max_event_rate_threshold_kev_s_;
constexpr uint32_t Gen31EventRateNoiseFilterModule_GTest::in_range_event_rate_threshold_kev_s_;

TEST_F_WITH_CAMERA(Gen31EventRateNoiseFilterModule_GTest, i_event_rate_gen31_event_rate_threshold,
                   camera_params(camera_param().integrator("Prophesee").generation("3.1"))) {
    open();

    constexpr uint32_t lower_out_of_range_time_window   = 0;
    constexpr uint32_t greater_out_of_range_time_window = max_event_rate_threshold_kev_s_ + 1;

    // Compute expected error as we converting kev/s in ev/us and the latter one is rounded
    uint32_t expected_max_error_kev_s = std::abs(in_range_event_rate_threshold_kev_s_ -
                                                 1000 * std::round(in_range_event_rate_threshold_kev_s_ / 1000.));

    // GIVEN Valid values for the event rate threshold
    // WHEN setting each of them
    // THEN The value set in the sensor is the correct one
    ASSERT_TRUE(ev_rate_->set_thresholds({min_event_rate_threshold_kev_s_, 0, 0, 0}));
    ASSERT_NEAR(min_event_rate_threshold_kev_s_, ev_rate_->get_thresholds().lower_bound_start,
                expected_max_error_kev_s);
    ASSERT_TRUE(ev_rate_->set_thresholds({max_event_rate_threshold_kev_s_, 0, 0, 0}));
    ASSERT_NEAR(max_event_rate_threshold_kev_s_, ev_rate_->get_thresholds().lower_bound_start,
                expected_max_error_kev_s);
    ASSERT_TRUE(ev_rate_->set_thresholds({in_range_event_rate_threshold_kev_s_, 0, 0, 0}));
    ASSERT_NEAR(in_range_event_rate_threshold_kev_s_, ev_rate_->get_thresholds().lower_bound_start,
                expected_max_error_kev_s);

    // GIVEN Invalid values for the event rate threshold
    // WHEN setting each of them
    // THEN The value set in the sensor is not changed (equal to the last set)
    const auto currently_time_window_new = ev_rate_->get_thresholds().lower_bound_start;
    ASSERT_FALSE(ev_rate_->set_thresholds({lower_out_of_range_time_window, 0, 0, 0}));
    ASSERT_EQ(currently_time_window_new, ev_rate_->get_thresholds().lower_bound_start);
    ASSERT_FALSE(ev_rate_->set_thresholds({greater_out_of_range_time_window, 0, 0, 0}));
    ASSERT_EQ(currently_time_window_new, ev_rate_->get_thresholds().lower_bound_start);
}
