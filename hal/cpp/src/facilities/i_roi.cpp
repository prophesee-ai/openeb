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
#include <iostream>
#include <algorithm>
#include <iterator>

#include "metavision/hal/facilities/i_roi.h"
#include "metavision/hal/utils/device_roi.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

std::vector<uint32_t> I_ROI::create_ROI(const DeviceRoi &roi) {
    return create_ROIs({roi});
}

void I_ROI::set_ROI(const DeviceRoi &roi, bool enable) {
    set_ROIs_from_bitword(create_ROIs({roi}), enable);
}

void I_ROI::set_ROIs(const std::vector<DeviceRoi> &vroi, bool enable) {
    set_ROIs_from_bitword(create_ROIs(vroi), enable);
}

void I_ROI::set_ROIs_from_file(std::string const &file_path, bool enable) {
    std::ifstream roi_file(file_path.c_str(), std::ios::in);
    if (!roi_file.is_open()) {
        MV_HAL_LOG_WARNING() << "Could not open file at" << file_path << "ROI not set.";
        return;
    }

    std::vector<DeviceRoi> vroi;
    std::copy(std::istream_iterator<DeviceRoi>(roi_file), std::istream_iterator<DeviceRoi>(), std::back_inserter(vroi));

    set_ROIs_from_bitword(create_ROIs(vroi), enable);
}

} // namespace Metavision
