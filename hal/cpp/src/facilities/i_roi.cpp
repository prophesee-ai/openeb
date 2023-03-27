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
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

I_ROI::Window::Window() = default;

I_ROI::Window::Window(int x, int y, int width, int height) : x(x), y(y), width(width), height(height) {}

bool I_ROI::Window::operator==(const I_ROI::Window &roi) const {
    return x == roi.x && y == roi.y && width == roi.width && height == roi.height;
}

std::string I_ROI::Window::to_string() const {
    std::string out = "[" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(width) + "x" +
                      std::to_string(height) + "]";
    return out;
}

std::istream &operator>>(std::istream &is, I_ROI::Window &rhs) {
    is >> rhs.x;
    is >> rhs.y;
    is >> rhs.width;
    is >> rhs.height;
    return is;
}

std::ostream &operator<<(std::ostream &lhs, I_ROI::Window &rhs) {
    lhs << rhs.to_string();
    return lhs;
}

bool I_ROI::set_window(const Window &window) {
    return set_windows({window});
}

bool I_ROI::set_windows(const std::vector<Window> &windows) {
    if (windows.size() > get_max_supported_windows_count()) {
        throw HalException(HalErrorCode::ValueOutOfRange,
                           "Too many windows provided to I_ROI::set_windows, maximum number of windows supported is " +
                               std::to_string(get_max_supported_windows_count()));
    }
    return set_windows_impl(windows);
}

} // namespace Metavision
