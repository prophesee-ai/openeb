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

#include "metavision/psee_hw_layer/devices/genx320/genx320_roi_interface.h"

namespace Metavision {

GenX320RoiInterface::GenX320RoiInterface(const std::shared_ptr<GenX320RoiDriver> &driver) :
    driver_(driver), enabled_(false) {}

int GenX320RoiInterface::device_width() const {
    return 320;
}

int GenX320RoiInterface::device_height() const {
    return 320;
}

bool GenX320RoiInterface::enable(bool state) {
    enabled_ = state;
    return driver_->enable(state);
}

bool GenX320RoiInterface::is_enabled() const {
    return enabled_;
}

bool GenX320RoiInterface::set_mode(const Mode &mode) {
    return driver_->set_roi_mode(mode);
}

I_ROI::Mode GenX320RoiInterface::get_mode() const {
    return driver_->get_roi_mode();
}

size_t GenX320RoiInterface::get_max_supported_windows_count() const {
    return driver_->get_max_windows_count();
}

bool GenX320RoiInterface::set_lines(const std::vector<bool> &cols, const std::vector<bool> &rows) {
    driver_->set_driver_mode(GenX320RoiDriver::DriverMode::LATCH);
    return driver_->set_lines(cols, rows);
}

bool GenX320RoiInterface::set_windows_impl(const std::vector<Window> &windows) {
    driver_->set_driver_mode(GenX320RoiDriver::DriverMode::MASTER);
    return driver_->set_windows(windows);
}

std::vector<I_ROI::Window> GenX320RoiInterface::get_windows() const {
    return driver_->get_windows();
}

bool GenX320RoiInterface::get_lines(std::vector<bool> &cols, std::vector<bool> &rows) const {
    return driver_->get_lines(cols, rows);
}

} // namespace Metavision
