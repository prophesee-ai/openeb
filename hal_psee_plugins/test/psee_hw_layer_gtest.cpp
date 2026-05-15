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

#include <filesystem>
#include <gtest/gtest.h>

#include "metavision/utils/gtest/gtest_custom.h"

#include "metavision/hal/device/device.h"
#include "metavision/hal/device/device_discovery.h"

#include <metavision/hal/facilities/i_hw_identification.h>
#include <metavision/psee_hw_layer/boards/rawfile/file_hw_identification.h>

using namespace Metavision;

class PseeBase_Gtest : virtual public ::testing::Test {
public:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_F_WITH_DATASET(PseeBase_Gtest, cast_facility) {
    // Read the dataset provided
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt3_hand.raw").string();

    auto device = DeviceDiscovery::open_raw_file(dataset_file_path);
    ASSERT_TRUE(device != nullptr);

    I_HW_Identification *i_hw_identification = device->get_facility<I_HW_Identification>();
    ASSERT_TRUE(i_hw_identification != nullptr);

    FileHWIdentification *f_hw_identification = dynamic_cast<FileHWIdentification *>(i_hw_identification);
    ASSERT_TRUE(f_hw_identification != nullptr);

    ASSERT_EQ(f_hw_identification->get_integrator(), "Prophesee");
}
