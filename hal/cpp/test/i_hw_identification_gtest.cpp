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

#include "utils/device_test.h"

class I_HW_Identification_GTest : public Metavision::testing::DeviceTest {
public:
    void on_opened_device(Metavision::Device &device) override {
        hw_id_ = device.get_facility<Metavision::I_HW_Identification>();
        ASSERT_NE(nullptr, hw_id_);
    }

protected:
    Metavision::I_HW_Identification *hw_id_{nullptr};
};

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_sensor_info_major_version_gen3_or_gen31,
                   camera_params(camera_param().generation("3.0"), camera_param().generation("3.1"))) {
    ASSERT_EQ(3, hw_id_->get_sensor_info().major_version_);
}

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_get_integrator_psee,
                   camera_params(camera_param().integrator("Prophesee"))) {
    ASSERT_EQ("Prophesee", hw_id_->get_integrator());
}

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_get_available_data_encoding_formats_psee_gen3,
                   camera_params(camera_param().integrator("Prophesee").generation("3.0"))) {
    ASSERT_EQ(1, hw_id_->get_available_data_encoding_formats().size());
    ASSERT_EQ("EVT2", hw_id_->get_available_data_encoding_formats()[0]);
    ASSERT_EQ("EVT2", hw_id_->get_available_data_encoding_formats()[0]);
    ASSERT_EQ("EVT2", hw_id_->get_current_data_encoding_format());
}

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_get_available_data_encoding_formats_psee_gen31,
                   camera_params(camera_param().integrator("Prophesee").generation("3.1"))) {
    ASSERT_EQ(1, hw_id_->get_available_data_encoding_formats().size());
    auto available = hw_id_->get_available_data_encoding_formats()[0];
    auto current = hw_id_->get_current_data_encoding_format();

    // Gen3.1 systems exist with either of these encodings
    ASSERT_TRUE((available == "EVT2") || (available == "EVT3"));
    ASSERT_TRUE((current == "EVT2") || (current == "EVT3"));
    ASSERT_EQ(available, current);
}

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_get_available_data_encoding_formats_psee_gen4,
                   camera_params(camera_param().integrator("Prophesee").generation("4.0"),
                                 camera_param().integrator("Prophesee").generation("4.1"))) {
    ASSERT_EQ(2, hw_id_->get_available_data_encoding_formats().size());
    ASSERT_EQ("EVT2", hw_id_->get_available_data_encoding_formats()[0]);
    ASSERT_EQ("EVT3", hw_id_->get_available_data_encoding_formats()[1]);
    ASSERT_EQ("EVT3", hw_id_->get_current_data_encoding_format());
}

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_get_sensor_info_gen3,
                   camera_params(camera_param().generation("3.0"))) {
    ASSERT_EQ("Gen3.0", hw_id_->get_sensor_info().name_);
}

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_get_sensor_info_gen31,
                   camera_params(camera_param().generation("3.1"))) {
    ASSERT_EQ("Gen3.1", hw_id_->get_sensor_info().name_);
}

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_get_sensor_info_gen4,
                   camera_params(camera_param().generation("4.0"))) {
    ASSERT_EQ("Gen4.0", hw_id_->get_sensor_info().name_);
}

TEST_F_WITH_CAMERA(I_HW_Identification_GTest, hd_get_sensor_info_gen41,
                   camera_params(camera_param().generation("4.1"))) {
    ASSERT_EQ("Gen4.1", hw_id_->get_sensor_info().name_);
}
