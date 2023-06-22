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

#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath> // floor

#include "metavision/psee_hw_layer/devices/genx320/genx320_ll_roi.h"
#include "metavision/psee_hw_layer/utils/register_map.h"
#include "metavision/hal/utils/device_config.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/resources_folder.h"
#include "metavision/sdk/base/utils/generic_header.h"

namespace Metavision {

GenX320LowLevelRoi::Grid::Grid(int columns, int rows) : columns_(columns), rows_(rows) {
    grid_.resize(rows_ * columns_, 0xFFFFFFFF);
}

std::string GenX320LowLevelRoi::Grid::to_string() const {
    std::string out = "\n";

    for (unsigned int y = 0; y < rows_; y++) {
        std::ostringstream row_str;
        row_str << "|| " << std::dec << std::setw(3) << y << " || ";
        for (unsigned int x = 0; x < columns_; x++) {
            row_str << std::hex << std::setw(8) << std::setfill('0') << grid_[y * columns_ + x];
            if (x == columns_ - 1) {
                row_str << " ||\n";
            } else {
                row_str << " | ";
            }
        }
        out += row_str.str();
    }
    return out;
}

std::tuple<unsigned int, unsigned int> GenX320LowLevelRoi::Grid::get_size() const {
    return std::make_tuple(columns_, rows_);
}

void GenX320LowLevelRoi::Grid::set_pixel(const unsigned int &column, const unsigned int &row, const bool &enable) {
    std::stringstream ss;

    if (column >= (columns_ * 32)) {
        ss << "Column index " << column << " out of range for sensor width (" << columns_ * 32 << ")";
        MV_HAL_LOG_ERROR() << ss.str();
        throw HalException(HalErrorCode::InvalidArgument, ss.str());
    } else if (row >= rows_) {
        ss << "Row index " << row << " out of range for sensor height (" << rows_ << ")";
        MV_HAL_LOG_ERROR() << ss.str();
        throw HalException(HalErrorCode::InvalidArgument, ss.str());
    } else {
        unsigned int vector_idx = floor(column / 32);
        unsigned int bit_idx    = column % 32;
        unsigned int reg_val    = grid_[row * columns_ + vector_idx];
        unsigned int mask       = (1 << bit_idx);

        uint32_t saved_fields = reg_val & (~mask);
        uint32_t write_field  = (enable << bit_idx);
        uint32_t new_reg_val  = saved_fields | write_field;

        ss << "Pixel selected   : " << std::dec << column << " x " << row << "\n";
        ss << "Vector ID        : " << vector_idx << "\n";
        ss << "Vector value     : 0x" << std::hex << std::setw(8) << std::setfill('0') << reg_val << "\n";
        ss << "Vector bit index : " << std::dec << bit_idx << "\n";
        ss << "Saved fields     : 0x" << std::hex << std::setw(8) << std::setfill('0') << saved_fields << "\n";
        ss << "Write fields     : 0x" << std::hex << std::setw(8) << std::setfill('0') << write_field << "\n";
        ss << "Write reg        : 0x" << std::hex << std::setw(8) << std::setfill('0') << new_reg_val;

        MV_HAL_LOG_DEBUG() << ss.str();

        grid_[row * columns_ + vector_idx] = new_reg_val;
    }
}

unsigned int &GenX320LowLevelRoi::Grid::get_vector(const unsigned int &vector_id, const unsigned int &row) {
    std::stringstream ss;

    if (row >= rows_) {
        ss << "Row index " << row << " out of range for LL ROI grid (" << columns_ << "x" << rows_ << ")";
        MV_HAL_LOG_ERROR() << ss.str();
        throw HalException(HalErrorCode::InvalidArgument, ss.str());
    } else if (vector_id >= columns_) {
        ss << "Vector index " << vector_id << " out of range for LL ROI grid (" << columns_ << "x" << rows_ << ")";
        MV_HAL_LOG_ERROR() << ss.str();
        throw HalException(HalErrorCode::InvalidArgument, ss.str());
    } else {
        return grid_[row * columns_ + vector_id];
    }
}

void GenX320LowLevelRoi::Grid::set_vector(const unsigned int &vector_id, const unsigned int &row,
                                          const unsigned int &val) {
    std::stringstream ss;

    if (row >= rows_) {
        ss << "Row index " << row << " out of range for LL ROI grid (" << columns_ << "x" << rows_ << ")";
        MV_HAL_LOG_ERROR() << ss.str();
        throw HalException(HalErrorCode::InvalidArgument, ss.str());
    } else if (vector_id >= columns_) {
        ss << "Vector index " << vector_id << " out of range for LL ROI grid (" << columns_ << "x" << rows_ << ")";
        MV_HAL_LOG_ERROR() << ss.str();
        throw HalException(HalErrorCode::InvalidArgument, ss.str());
    } else {
        grid_[row * columns_ + vector_id] = val;
    }
}

std::filesystem::path GenX320LowLevelRoi::default_calibration_path() {
    static auto calib_path = std::filesystem::path(ResourcesFolder::get_user_path()) / "active_pixel_calib.txt";
    return calib_path;
}

GenX320LowLevelRoi::GenX320LowLevelRoi(const DeviceConfig &config, const std::shared_ptr<RegisterMap> &regmap,
                                       const std::string &sensor_prefix) :
    register_map_(regmap), sensor_prefix_(sensor_prefix) {
    // Reset ROI to full resolution
    reset();

    // Disable ROI master and ROI driver
    (*register_map_)["roi_master_ctrl"]["roi_master_en"].write_value(0);
    (*register_map_)["roi_driver_ctrl"]["roi_driver_en"].write_value(0);

    // Disable roi y (D-Latch enable pin) auto clear (shadow trigger needs to be called for each roi y reg access)
    (*register_map_)["roi_ctrl"]["px_roi_halt_programming"].write_value(0);

    if (!config.get<bool>("ignore_active_pixel_calibration_data", false)) {
        auto calib_path = default_calibration_path();
        if (std::filesystem::exists(calib_path)) {
            MV_HAL_LOG_TRACE() << "Found calibration data at" << calib_path;
            MV_HAL_LOG_TRACE() << "Loading the calibration data";
            load_calibration_file(calib_path);
        }
    }
}

void GenX320LowLevelRoi::reset() {
    (*register_map_)["roi_ctrl"]["px_roi_halt_programming"].write_value(1);
    (*register_map_)["roi_ctrl"]["px_sw_rstn"].write_value(0);
    (*register_map_)["roi_ctrl"]["px_sw_rstn"].write_value(1);
    (*register_map_)["roi_ctrl"]["px_roi_halt_programming"].write_value(0);
}

bool GenX320LowLevelRoi::apply(GenX320LowLevelRoi::Grid &user_grid) {
    // ROI X Active column. Pins 'data' of the integrated pixel's Latch. 0: Enable, 1:Disable (default: Enable)
    // ROI Y Active row. Pins 'ENABLE' of the integrated pixel's Latch. 0: Disable, 1:Enable (default: Enable)

    auto grid_size = user_grid.get_size();
    auto x_max     = std::get<0>(grid_size);
    auto y_max     = std::get<1>(grid_size);
    std::string reg_y_name;
    std::string reg_x_name;
    unsigned int reg_val = 0;

    if (x_max != 10 || y_max != 320) {
        std::stringstream ss;
        ss << "Grid size " << x_max << "x" << y_max << " invalid for GenX320. (Expected size : " << 10 << "x" << 320
           << ")";
        MV_HAL_LOG_ERROR() << ss.str();
        throw HalException(HalErrorCode::InvalidArgument, ss.str());
    }

    MV_HAL_LOG_TRACE() << "Applying ROI" << x_max << "x" << y_max;

    (*register_map_)["roi_ctrl"].write_value({{"roi_td_en", 1}, {"px_iphoto_en", 0}});

    // Iterate over each rows
    for (unsigned int y = 0; y < y_max; y++) {
        // Compute roi_y register name and select only one row at a time from row index
        std::ostringstream reg_id;
        reg_id << std::setw(2) << std::setfill('0') << floor(y / 32);
        reg_y_name = "roi/td_roi_y" + reg_id.str();
        reg_val    = (0x00000001 << (y % 32));
        MV_HAL_LOG_DEBUG() << "ROI Y" << y << std::setw(3) << reg_y_name << ":" << std::hex << std::setw(8)
                           << std::setfill('0') << reg_val;

        // Send roi y register access, enabling all D-Latches a current row index
        (*register_map_)[reg_y_name].write_value(reg_val);

        // Iterate over each columns composed of 32 bits vectors
        for (unsigned int x = 0; x < x_max; x++) {
            // Compute roi_x register name from column index
            std::ostringstream reg_id;
            reg_id << std::setw(2) << std::setfill('0') << x;
            reg_x_name = "roi/td_roi_x" + reg_id.str();
            MV_HAL_LOG_DEBUG() << "ROI" << y << "x" << x << ":" << std::hex << std::setw(8) << std::setfill('0')
                               << ~user_grid.get_vector(x, y);

            // Send roi x register access, setting all D-Latches data lanes from given user config on current row.
            (*register_map_)[reg_x_name].write_value(~user_grid.get_vector(x, y));
        }

        // Apply configuration to the hardware by triggering D-Latches toggling of their respective output
        (*register_map_)["roi_ctrl"]["roi_td_shadow_trigger"].write_value(1);

        // Clear roi y register. Deselect current row
        (*register_map_)[reg_y_name].write_value(0);
        (*register_map_)["roi_ctrl"]["roi_td_shadow_trigger"].write_value(1);
    }

    // Disable ROI
    (*register_map_)["roi_ctrl"]["roi_td_en"].write_value(0);

    return true;
}

bool GenX320LowLevelRoi::load_calibration_file(const std::filesystem::path &path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        return false;
    }

    Grid g(10, 320);
    GenericHeader header(ifs);
    std::string line;
    while (std::getline(ifs, line)) {
        int x, y;
        if (std::istringstream(line) >> x >> y) {
            g.set_pixel(x, y, false);
        }
    }
    apply(g);

    return true;
}

} // namespace Metavision
