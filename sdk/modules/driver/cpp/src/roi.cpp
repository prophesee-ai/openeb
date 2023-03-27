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
#include "metavision/sdk/driver/camera_error_code.h"
#include "metavision/sdk/driver/camera_exception.h"

namespace Metavision {

Roi::Roi(I_ROI *roi) : pimpl_(roi) {}

Roi::~Roi() {}

void Roi::set(Window roi) {
    auto roi_to_set = I_ROI::Window(roi.x, roi.y, roi.width, roi.height);
    pimpl_->set_window(roi_to_set);
}

void Roi::set(const std::vector<bool> &cols, const std::vector<bool> &rows) {
    if (!pimpl_->set_lines(cols, rows)) {
        throw(CameraException(
            CameraErrorCode::RoiError,
            "When trying to set advanced ROI: input binary map must be of the same size of the sensor's dimension."));
    }
}

void Roi::set(const std::vector<Roi::Window> &rois) {
    std::vector<I_ROI::Window> windows;
    for (auto roi : rois) {
        windows.push_back({roi.x, roi.y, roi.width, roi.height});
    }
    pimpl_->set_windows(windows);
}

void Roi::unset() {
    pimpl_->enable(false);
}

I_ROI *Roi::get_facility() const {
    return pimpl_;
}

} // namespace Metavision
