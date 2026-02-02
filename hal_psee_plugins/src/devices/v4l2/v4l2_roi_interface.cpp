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

#include "metavision/psee_hw_layer/devices/v4l2/v4l2_roi_interface.h"

namespace Metavision {

void raise_error(const std::string &str);

V4L2RoiInterface::V4L2RoiInterface(std::shared_ptr<V4L2DeviceControl> device) : device_(device), enabled_(false) {
    auto controls = device_->get_controls();
    auto ctrl     = controls->get("roi_reset");
    if (ctrl.push()) {
        MV_HAL_LOG_ERROR() << "Failed to reset roi";
    }
}

int V4L2RoiInterface::device_width() const {
    return device_->get_width();
}

int V4L2RoiInterface::device_height() const {
    return device_->get_height();
}

bool V4L2RoiInterface::enable(bool state) {
    enabled_ = state;
    return true;
}

bool V4L2RoiInterface::is_enabled() const {
    return enabled_;
}

bool V4L2RoiInterface::set_mode(const Mode &mode) {
    auto controls = device_->get_controls();
    auto ctrl     = controls->get("roi_roni");
    switch (mode) {
    case I_ROI::Mode::ROI:
        return ctrl.set_bool(false) < 0 ? false : true;
    case I_ROI::Mode::RONI:
        return ctrl.set_bool(true) < 0 ? false : true;
    default:
        return false;
    }
}

I_ROI::Mode V4L2RoiInterface::get_mode() const {
    auto controls = device_->get_controls();
    auto ctrl     = controls->get("roi_roni");
    auto is_roni  = ctrl.get_bool().value();
    return is_roni ? I_ROI::Mode::RONI : I_ROI::Mode::ROI;
}

size_t V4L2RoiInterface::get_max_supported_windows_count() const {
    // update roi_set to provide max number of windows
    // for now supoprt both genx and imx with only 1 allowed window
    return 18;
}

bool V4L2RoiInterface::set_lines(const std::vector<bool> &cols, const std::vector<bool> &rows) {
    throw std::runtime_error("V4L2RoiInterface::set_lines() not implemented");
}

struct roi {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
};

// TODO:
// add size and use flexible array
// struct roi_set {
//    uint32_t n;       // number of set rois
//    uint32_t size;     // number of possible rois
//    struct roi rois[]; // flexible array of rois
// }

struct roi_set {
    uint32_t n;
    struct roi rois[18];
};

bool V4L2RoiInterface::set_windows_impl(const std::vector<Window> &windows) {
    auto controls        = device_->get_controls();
    auto ctrl            = controls->get("roi_set");
    struct roi_set *rois = ctrl.get_compound<struct roi_set>().value();

    if (windows.size() > get_max_supported_windows_count()) {
        MV_HAL_LOG_ERROR() << "Too many windows";
        return false;
    }

    rois->n = 0;
    for (const auto &window : windows) {
        if (window.width <= 0 || window.height <= 0) {
            continue;
        }

        auto &roi  = rois->rois[rois->n++];
        roi.x      = std::max(0, window.x);
        roi.y      = std::max(0, window.y);
        roi.width  = std::min(device_width() - roi.x, static_cast<uint32_t>(window.width));
        roi.height = std::min(device_width() - roi.y, static_cast<uint32_t>(window.height));
    }

    if (ctrl.set_compound(rois)) {
        MV_HAL_LOG_ERROR() << "Failed to set windows";
        return false;
    }

    return true;
}

std::vector<I_ROI::Window> V4L2RoiInterface::get_windows() const {
    return {};
}

bool V4L2RoiInterface::get_lines(std::vector<bool> &cols, std::vector<bool> &rows) const {
    raise_error("V4L2RoiInterface::get_lines() not implemented");
    return false;
}

} // namespace Metavision
