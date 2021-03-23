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

#include "metavision/sdk/driver/roi.h"
#include "metavision/hal/utils/device_roi.h"
#include "metavision/sdk/driver/camera_error_code.h"
#include "metavision/sdk/driver/camera_exception.h"

namespace Metavision {

Roi::Roi(I_ROI *roi) : pimpl_(roi) {}

Roi::~Roi() {}

void Roi::set(Rectangle roi) {
    auto roi_to_set = DeviceRoi(roi.x, roi.y, roi.width, roi.height);
    pimpl_->set_ROI(roi_to_set, true);
}

void Roi::set(const std::vector<bool> &cols_to_enable, const std::vector<bool> &rows_to_enable) {
    if (!pimpl_->set_ROIs(cols_to_enable, rows_to_enable, true)) {
        throw(CameraException(
            CameraErrorCode::RoiError,
            "When trying to set advanced ROI: input binary map must be of the same size of the sensor's dimension."));
    }
}

void Roi::set(const std::vector<Roi::Rectangle> &rois_to_set) {
    std::vector<DeviceRoi> to_set;
    for (auto roi : rois_to_set) {
        to_set.push_back({roi.x, roi.y, roi.width, roi.height});
    }
    pimpl_->set_ROIs(to_set);
}

void Roi::unset() {
    pimpl_->enable(false);
}

I_ROI *Roi::get_facility() const {
    return pimpl_;
}

} // namespace Metavision
