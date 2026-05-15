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

#include <algorithm> // std::min

#include "metavision/psee_hw_layer/facilities/psee_roi.h"
#include "metavision/hal/facilities/i_geometry.h"

namespace Metavision {

namespace {

std::vector<uint32_t> get_roi_bitword(const std::vector<bool> &is_pixel_in_roi, int word_size) {
    std::vector<uint32_t> roi;
    const int nelem = is_pixel_in_roi.size();
    unsigned int p1 = 0;
    auto pixel_ind  = 0;
    for (; pixel_ind < nelem; ++pixel_ind) {
        if (is_pixel_in_roi[pixel_ind]) {
            p1 |= (1 << (pixel_ind % word_size));
        }

        if (((pixel_ind + 1) % word_size) == 0) {
            roi.push_back(p1);
            p1 = 0;
        }
    }
    if (pixel_ind % word_size != 0) {
        roi.push_back(p1);
    }
    return roi;
}

} // anonymous namespace

PseeROI::PseeROI(int width, int height) : device_height_(height), device_width_(width) {}

int PseeROI::device_width() const {
    return device_width_;
}

int PseeROI::device_height() const {
    return device_height_;
}

bool PseeROI::set_mode(const Mode &mode) {
    return mode == Mode::ROI;
}

I_ROI::Mode PseeROI::get_mode() const {
    return Mode::ROI;
}

size_t PseeROI::get_max_supported_windows_count() const {
    return 1;
}

bool PseeROI::write_ROI_windows(const std::vector<Window> &windows) {
    write_ROI(create_ROIs(windows));
    return true;
}

bool PseeROI::set_windows_impl(const std::vector<Window> &windows) {
    if (write_ROI_windows(windows)) {
        is_lines = false;
        active_windows_ = windows;
        return true;
    }
    return false;
}

std::vector<I_ROI::Window> PseeROI::get_windows() const {
    if (is_lines) {
        return std::vector<I_ROI::Window>();
    }
    return active_windows_;
}

bool PseeROI::set_lines(const std::vector<bool> &cols, const std::vector<bool> &rows) {
    if ((cols.size() != static_cast<size_t>(device_width_)) || (rows.size() != static_cast<size_t>(device_height_))) {
        return false;
    }
    is_lines = true;
    auto windows = lines_to_windows(cols, rows);
    active_windows_ = windows;
    write_ROI(create_ROIs(windows));
    return true;
}

bool PseeROI::get_lines(std::vector<bool> &cols, std::vector<bool> &rows) const {
    if (!is_lines) {
        return false;
    }

    lines_from_windows(active_windows_, cols, rows);
    return true;
}

void PseeROI::lines_from_windows(const std::vector<Window> &windows, std::vector<bool> &cols, std::vector<bool> &rows) const {
    if (cols.size() != static_cast<size_t>(device_width_)) {
        cols = std::vector<bool>(device_width_);
    }
    std::fill(cols.begin(), cols.end(), false);

    if (rows.size() != static_cast<size_t>(device_height_)) {
        rows = std::vector<bool>(device_height_);
    }
    std::fill(rows.begin(), rows.end(), false);

    for (auto &window : windows) {
        for (int i = window.x; i < window.x + window.width; ++i) {
            cols[i] = true;
        }
        for (int i = window.y; i < window.y + window.height; ++i) {
            rows[i] = true;
        }
    }
}

std::vector<uint32_t> PseeROI::create_ROIs(const std::vector<Window> &windows, int device_width, int device_height,
                                           bool x_flipped, int word_size, int x_offset, int y_offset) {
    const auto nelem     = device_width + device_height;
    auto is_pixel_in_roi = std::vector<bool>(nelem, false);

    for (auto const &roi : windows) {
        // clamp into [0, width - 1] and [0, height - 1]
        const auto max_x = std::min(roi.x + x_offset + roi.width, device_width - x_offset);
        const auto max_y = std::min(roi.y + y_offset + roi.height, device_height - y_offset);
        const auto min_x = std::max(roi.x + x_offset, 0);
        const auto min_y = std::max(roi.y + y_offset, 0);

        for (auto col_ind = min_x; col_ind < max_x; ++col_ind) {
            const auto ind       = x_flipped ? device_width - 1 - col_ind : col_ind;
            is_pixel_in_roi[ind] = true;
        }

        for (auto row_ind = (device_width + min_y); row_ind < (device_width + max_y); ++row_ind) {
            is_pixel_in_roi[row_ind] = true;
        }
    }

    return get_roi_bitword(is_pixel_in_roi, word_size);
}

std::vector<I_ROI::Window> PseeROI::lines_to_windows(const std::vector<bool> &cols_to_enable, const std::vector<bool> &rows_to_enable) {
    auto windows = std::vector<Window>();
    auto vxroi   = std::vector<std::pair<int, int>>();
    auto vyroi   = std::vector<std::pair<int, int>>();

    /*
     * Here, we convert the boolean vector into a vector of DeviceROI which are converted into a bitword vector
     * with the function create_ROIs(std::vector<DeviceRoi>)
     */

    size_t col_index    = 0;
    bool prev_to_enable = false;
    std::pair<int, int> cur_roi;
    for (auto to_enable : cols_to_enable) {
        if (to_enable) {
            if (!prev_to_enable) {
                prev_to_enable = true;
                cur_roi.first  = col_index;
            }
        } else {
            if (prev_to_enable) {
                cur_roi.second = col_index - cur_roi.first;
                prev_to_enable = false;
                vxroi.push_back(cur_roi);
            }
        }
        ++col_index;
    }

    if (prev_to_enable) {
        cur_roi.second = col_index - cur_roi.first;
        prev_to_enable = false;
        vxroi.push_back(cur_roi);
    }

    size_t row_index = 0;
    prev_to_enable   = false;
    for (auto to_enable : rows_to_enable) {
        if (to_enable) {
            if (!prev_to_enable) {
                prev_to_enable = true;
                cur_roi.first  = row_index;
            }
        } else {
            if (prev_to_enable) {
                cur_roi.second = row_index - cur_roi.first;
                prev_to_enable = false;
                vyroi.push_back(cur_roi);
            }
        }
        ++row_index;
    }

    if (prev_to_enable) {
        cur_roi.second = row_index - cur_roi.first;
        prev_to_enable = false;
        vyroi.push_back(cur_roi);
    }

    for (auto xroi : vxroi) {
        for (auto yroi : vyroi) {
            windows.push_back(Window(xroi.first, yroi.first, xroi.second, yroi.second));
        }
    }

    return windows;
}

std::vector<uint32_t> PseeROI::create_ROIs(const std::vector<bool> &cols_to_enable,
                                           const std::vector<bool> &rows_to_enable, int x_offset, int y_offset) {
    auto windows = lines_to_windows(cols_to_enable, rows_to_enable);
    return create_ROIs(windows, device_width_, device_height_, roi_x_flipped(), get_word_size(), x_offset, y_offset);
}

std::vector<uint32_t> PseeROI::create_ROIs(const std::vector<Window> &windows) {
    return create_ROIs(windows, device_width_, device_height_, roi_x_flipped(), get_word_size());
}

bool PseeROI::roi_x_flipped() const {
    return false;
}

int PseeROI::get_word_size() const {
    return 32;
}

} // namespace Metavision
