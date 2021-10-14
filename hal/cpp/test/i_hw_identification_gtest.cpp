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

#include "metavision/hal/device/device.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/utils/gtest/gtest_custom.h"

class I_HW_Identification_GTest : public ::testing::Test {
public:
    void open() {
        try {
            auto serial_list = Metavision::DeviceDiscovery::list();
            if (serial_list.empty()) {
                std::cerr << "No Device Found" << std::endl;
                FAIL();
            } else if (serial_list.size() > 1) {
                std::cerr << "WARNING: Several Cameras Plugged In" << std::endl;
            }

            device_ = Metavision::DeviceDiscovery::open("");

        } catch (const Metavision::HalException &e) {
            std::cerr << "Plug a camera to run this test." << std::endl;
            FAIL();
        }

        ASSERT_NE(nullptr, device_.get());

        hw_id_ = device_->get_facility<Metavision::I_HW_Identification>();
        ASSERT_NE(nullptr, hw_id_);
    }

protected:
    std::unique_ptr<Metavision::Device> device_;
    Metavision::I_HW_Identification *hw_id_{nullptr};
};

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_sensor_info_major_version_gen3_or_gen31,
                   camera_params(camera_param().generation("3.0"), camera_param().generation("3.1"))) {
    open();
    ASSERT_EQ(3, hw_id_->get_sensor_info().major_version_);
}

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_get_integrator_psee,
                   camera_params(camera_param().integrator("Prophesee"))) {
    open();
    ASSERT_EQ("Prophesee", hw_id_->get_integrator());
}

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_get_available_raw_format_psee_gen3,
                   camera_params(camera_param().integrator("Prophesee").generation("3.0"))) {
    open();
    ASSERT_EQ(1, hw_id_->get_available_raw_format().size());
    ASSERT_EQ("EVT2", hw_id_->get_available_raw_format()[0]);
}

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_get_available_raw_format_psee_gen31,
                   camera_params(camera_param().integrator("Prophesee").generation("3.1"))) {
    open();
    ASSERT_EQ(1, hw_id_->get_available_raw_format().size());
    if (hw_id_->get_system_id() == 0x28) {
        ASSERT_EQ("EVT3", hw_id_->get_available_raw_format()[0]);
    } else {
        ASSERT_EQ("EVT2", hw_id_->get_available_raw_format()[0]);
    }
}

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_get_available_raw_format_psee_gen4,
                   camera_params(camera_param().integrator("Prophesee").generation("4.0"),
                                 camera_param().integrator("Prophesee").generation("4.1"))) {
    open();
    ASSERT_EQ(2, hw_id_->get_available_raw_format().size());
    ASSERT_EQ("EVT2", hw_id_->get_available_raw_format()[0]);
    ASSERT_EQ("EVT3", hw_id_->get_available_raw_format()[1]);
}

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_get_sensor_info_gen3,
                   camera_params(camera_param().generation("3.0"))) {
    open();
    ASSERT_EQ("3.0", hw_id_->get_sensor_info().as_string());
}

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_get_sensor_info_gen31,
                   camera_params(camera_param().generation("3.1"))) {
    open();
    ASSERT_EQ("3.1", hw_id_->get_sensor_info().as_string());
}

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_get_sensor_info_gen4,
                   camera_params(camera_param().generation("4.0"))) {
    open();
    ASSERT_EQ("4.0", hw_id_->get_sensor_info().as_string());
}

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_get_sensor_info_gen41,
                   camera_params(camera_param().generation("4.1"))) {
    open();
    ASSERT_EQ("4.1", hw_id_->get_sensor_info().as_string());
}
