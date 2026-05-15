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

#include <memory>
#include <fstream>

#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/utils/gtest/gtest_custom.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/raw_file_header.h"

class RawFileHeader_Gtest : public Metavision::GTestWithTmpDir {};

TEST_F_WITH_CAMERA(RawFileHeader_Gtest, check_evt_format_is_in_psee_files_header,
                   camera_params(camera_param().integrator("Prophesee"))) {
    std::string raw_file_name = tmpdir_handler_->get_full_path("recording.raw");
    {
        std::unique_ptr<Metavision::Device> device;
        try {
            device = Metavision::DeviceDiscovery::open("");
        } catch (const Metavision::HalException &e) {
            std::cerr << e.what() << std::endl;
            FAIL();
        }

        // Call log raw data, so that we write the header
        bool log_success = device->get_facility<Metavision::I_EventsStream>()->log_raw_data(raw_file_name);
        ASSERT_TRUE(log_success);
    }

    std::ifstream ifs(raw_file_name);
    ASSERT_TRUE(ifs.is_open());

    Metavision::RawFileHeader header(ifs);

    ASSERT_FALSE(header.empty());
    ASSERT_FALSE(header.get_field("evt").empty());
}
