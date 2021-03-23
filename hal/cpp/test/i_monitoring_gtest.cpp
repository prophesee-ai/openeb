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
#include <thread>
#include <chrono>

#include "metavision/utils/gtest/gtest_custom.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_monitoring.h"
#include "metavision/hal/facilities/i_device_control.h"

using namespace Metavision;

class I_Monitoring_GTest : public ::testing::Test {
public:
    void TearDown() override {
        device_.reset(nullptr);
        hw_id_      = nullptr;
        monitoring_ = nullptr;
    }

    void open(const std::string &sensor_info_version) {
        try {
            device_ = DeviceDiscovery::open("");
        } catch (const HalException &e) {
            std::cerr << "Plug a camera to run this test." << std::endl;
            EXPECT_EQ(0, 1);
            return;
        }

        EXPECT_NE(nullptr, device_.get());

        // Check hw identification
        hw_id_ = device_->get_facility<I_HW_Identification>();
        EXPECT_NE(nullptr, hw_id_);

        // This test is made for gen31 Prophesee cameras :
        if (hw_id_->get_integrator() != "Prophesee") {
            std::cerr << "The plugged camera must be a Prophesee Gen" << sensor_info_version << " for this test."
                      << std::endl;
            FAIL();
            return;
        }
        auto sensor_info = hw_id_->get_sensor_info();
        if (sensor_info.as_string() != sensor_info_version) {
            std::cerr << "The plugged camera must be a Prophesee Gen" << sensor_info_version << " for this test."
                      << std::endl;
            FAIL();
            return;
        }
        monitoring_ = device_->get_facility<I_Monitoring>();
        EXPECT_NE(nullptr, monitoring_);
    }

protected:
    std::unique_ptr<Device> device_;
    I_HW_Identification *hw_id_{nullptr};
    I_Monitoring *monitoring_{nullptr};
};

TEST_F_WITH_CAMERA(I_Monitoring_GTest, i_monitoring_gen31_temperature,
                   camera_params(camera_param().integrator("Prophesee").generation("3.1"))) {
    open("3.1");
    auto device_control = device_->get_facility<I_DeviceControl>();
    ASSERT_NE(nullptr, device_control);
    device_control->start();

    ASSERT_GT(monitoring_->get_temperature(), 15);
    ASSERT_LT(monitoring_->get_temperature(), 120);
}

TEST_F_WITH_CAMERA(I_Monitoring_GTest, i_monitoring_gen31_illumination,
                   camera_params(camera_param().integrator("Prophesee").generation("3.1"))) {
    open("3.1");
    auto device_control = device_->get_facility<I_DeviceControl>();
    ASSERT_NE(nullptr, device_control);
    device_control->start();

    // First measure gives invalid values. Looks like an unitiliazed value within the fpga or
    // something.
    monitoring_->get_illumination();

    uint32_t test = 0;
    for (; test < 5; ++test) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        EXPECT_GE(monitoring_->get_illumination(), 0); // Can be 0 in dark scene
        EXPECT_LT(monitoring_->get_illumination(), 100);
    }
}
