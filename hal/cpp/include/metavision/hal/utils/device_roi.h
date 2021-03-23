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

#ifndef METAVISION_HAL_DEVICE_ROI_H
#define METAVISION_HAL_DEVICE_ROI_H

#include <istream>
#include <string>

namespace Metavision {

/// @brief Defines a simple rectangular region
class DeviceRoi {
public:
    DeviceRoi() = default;

    /// @brief Destructor
    ~DeviceRoi() = default;

    /// @brief Creates a rectangle defined by the corner {(x, y), (x + width, y + height)}
    DeviceRoi(int x, int y, int width, int height) : x_(x), y_(y), width_(width), height_(height) {}

    // defined for python bindings
    bool operator==(const DeviceRoi &roi) const {
        return x_ == roi.x_ && y_ == roi.y_ && width_ == roi.width_ && height_ == roi.height_;
    }

    /// @brief Returns the ROI as a string
    /// @return Human readable string representation of an ROI
    std::string to_string() {
        std::string out = "[" + std::to_string(x_) + "," + std::to_string(y_) + "," + std::to_string(width_) + "x" +
                          std::to_string(height_) + "]";
        return out;
    }

    // defined to read from a string buffer
    friend std::istream &operator>>(std::istream &is, DeviceRoi &rhs) {
        is >> rhs.x_;
        is >> rhs.y_;
        is >> rhs.width_;
        is >> rhs.height_;
        return is;
    }

    // defined to output a string
    friend std::ostream &operator<<(std::ostream &lhs, DeviceRoi &rhs) {
        lhs << rhs.to_string();
        return lhs;
    }

public:
    int x_      = -1;
    int y_      = -1;
    int width_  = -1;
    int height_ = -1;
};
} // namespace Metavision

#endif // METAVISION_HAL_DEVICE_ROI_H
