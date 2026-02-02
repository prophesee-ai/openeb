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

#include "metavision/psee_hw_layer/devices/v4l2/v4l2_crop.h"
#include "boards/v4l2/v4l2_device.h"
#include <cmath>

namespace Metavision {

void raise_error(const std::string &str);

V4L2Crop::V4L2Crop(std::shared_ptr<V4L2DeviceControl> device) : device_(device), enabled_(false) {
    // set default crop to native size
    device_->get_native_size(device_->get_sensor_entity()->fd, rect_);
    native_rect_ = rect_;
    // reset
    device->set_crop(device->get_sensor_entity()->fd, native_rect_);
}

int V4L2Crop::device_width() const {
    return native_rect_.width;
}

int V4L2Crop::device_height() const {
    return native_rect_.height;
}

bool V4L2Crop::enable(bool state) {
    enabled_ = state;
    if (enabled_) {
        device_->set_crop(device_->get_sensor_entity()->fd, rect_);
    } else {
        device_->set_crop(device_->get_sensor_entity()->fd, native_rect_);
    }
    return true;
}

bool V4L2Crop::is_enabled() const {
    return enabled_;
}

bool V4L2Crop::set_mode(const Mode &mode) {
    raise_error("V4L2Crop::set_mode() not supported by device");
    return false;
}

I_ROI::Mode V4L2Crop::get_mode() const {
    return I_ROI::Mode::ROI;
}

size_t V4L2Crop::get_max_supported_windows_count() const {
    return 1;
}

bool V4L2Crop::set_lines(const std::vector<bool> &cols, const std::vector<bool> &rows) {
    throw std::runtime_error("V4L2Crop::set_lines() not implemented");
}

bool V4L2Crop::set_windows_impl(const std::vector<Window> &windows) {
    if (windows.size() > 1) {
        MV_HAL_LOG_ERROR() << "Only one window is supported";
        return false;
    }

    const auto &window = windows[0];

    if (window.width <= 0 || window.height <= 0) {
        return false;
    }

    rect_.left   = std::max(0, window.x);
    rect_.top    = std::max(0, window.y);
    rect_.width  = std::min(device_width() - rect_.left, window.width);
    rect_.height = std::min(device_width() - rect_.top, window.height);

    if (enabled_) {
        device_->set_crop(device_->get_sensor_entity()->fd, rect_);
    }

    return true;
}

std::vector<I_ROI::Window> V4L2Crop::get_windows() const {
    Window window;
    window.x      = rect_.left;
    window.y      = rect_.top;
    window.width  = rect_.width;
    window.height = rect_.height;
    return {window};
}

bool V4L2Crop::get_lines(std::vector<bool> &cols, std::vector<bool> &rows) const {
    throw std::runtime_error("V4L2Crop::set_lines() not implemented");
}

} // namespace Metavision
