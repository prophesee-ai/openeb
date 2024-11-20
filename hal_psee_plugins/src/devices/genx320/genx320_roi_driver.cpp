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

#include <iostream>
#include <cmath> // floor
#include <fstream>
#include <iomanip>

#include "metavision/psee_hw_layer/devices/genx320/genx320_roi_driver.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/resources_folder.h"
#include "metavision/sdk/base/utils/generic_header.h"

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {

GenX320RoiDriver::Grid::Grid(int columns, int rows) : columns_(columns), rows_(rows) {
    grid_.resize(rows_ * columns_, 0xFFFFFFFF);
}

std::string GenX320RoiDriver::Grid::to_string() const {
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

std::tuple<unsigned int, unsigned int> GenX320RoiDriver::Grid::get_size() const {
    return std::make_tuple(columns_, rows_);
}

void GenX320RoiDriver::Grid::clear() {
    for (unsigned int y = 0; y < rows_; y++) {
        for (unsigned int x = 0; x < columns_; x++) {
            grid_[y * columns_ + x] = 0xFFFFFFFF;
        }
    }
}

void GenX320RoiDriver::Grid::set_pixel(const unsigned int &column, const unsigned int &row, const bool &enable) {
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

unsigned int &GenX320RoiDriver::Grid::get_vector(const unsigned int &vector_id, const unsigned int &row) {
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

unsigned int GenX320RoiDriver::Grid::get_vector(const unsigned int &vector_id, const unsigned int &row) const {
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

void GenX320RoiDriver::Grid::set_vector(const unsigned int &vector_id, const unsigned int &row,
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

std::filesystem::path GenX320RoiDriver::default_calibration_path() {
#ifndef __ANDROID__
    static auto calib_path = std::filesystem::path(ResourcesFolder::get_user_path()) / "active_pixel_calib.txt";
    return calib_path;
#else
    throw HalException(HalErrorCode::OperationNotImplemented);
#endif
}

GenX320RoiDriver::GenX320RoiDriver(int width, int height, const std::shared_ptr<RegisterMap> &regmap,
                                   const std::string &sensor_prefix, const DeviceConfig &config) :
    register_map_(regmap),
    sensor_prefix_(sensor_prefix),
    mode_(I_ROI::Mode::ROI),
    roi_window_cnt_(0),
    grid_(10, 320),
    device_width_(width),
    device_height_(height) {
    reset_to_full_roi();
    set_driver_mode(GenX320RoiDriver::DriverMode::IO);

    if (!config.get<bool>("ignore_active_pixel_calibration_data", false)) {
        // /!\ Calibration configuration will disable the default pixel reset IO mode
        auto calib_path = default_calibration_path();
        if (std::filesystem::exists(calib_path)) {
            MV_HAL_LOG_TRACE() << "Found calibration data at" << calib_path;
            MV_HAL_LOG_TRACE() << "Loading the calibration data";
            set_driver_mode(GenX320RoiDriver::DriverMode::LATCH);
            load_calibration_file(calib_path);
        }
    }
}

bool GenX320RoiDriver::set_pixel(const unsigned int &column, const unsigned int &row, const bool &enable) {
    grid_.set_pixel(column, row, enable);
    return true;
}

bool GenX320RoiDriver::set_grid(GenX320RoiDriver::Grid &user_grid) {
    auto grid_size = user_grid.get_size();
    auto x_max     = std::get<0>(grid_size);
    auto y_max     = std::get<1>(grid_size);

    if (x_max != 10 || y_max != 320) {
        std::stringstream ss;
        ss << "Grid size " << x_max << "x" << y_max << " invalid for GenX320. (Expected size : " << 10 << "x" << 320
           << ")";
        MV_HAL_LOG_ERROR() << ss.str();
        return false;
    } else {
        grid_ = user_grid;
        return true;
    }
}

void GenX320RoiDriver::apply_grid() {
    // ROI X Active column. Pins 'data' of the integrated pixel's Latch. 0: Enable, 1:Disable (default: Enable)
    // ROI Y Active row. Pins 'ENABLE' of the integrated pixel's Latch. 0: Disable, 1:Enable (default: Enable)

    std::string reg_y_name;
    std::string reg_x_name;
    unsigned int reg_val = 0;

    // Iterate over each rows
    for (unsigned int y = 0; y < 320; y++) {
        // Compute roi_y register name and select only one row at a time from row index
        std::ostringstream reg_id;
        reg_id << std::setw(2) << std::setfill('0') << floor(y / 32);
        reg_y_name = "roi/td_roi_y" + reg_id.str();
        reg_val    = (0x00000001 << (y % 32));
        MV_HAL_LOG_DEBUG() << "ROI Y" << y << std::setw(3) << reg_y_name << ":" << std::hex << std::setw(8)
                           << std::setfill('0') << reg_val;

        // Send roi y register access, enabling all D-Latches a current row index
        (*register_map_)[sensor_prefix_ + reg_y_name].write_value(reg_val);

        // Iterate over each columns composed of 32 bits vectors
        for (unsigned int x = 0; x < 10; x++) {
            // Compute roi_x register name from column index
            std::ostringstream reg_id;
            reg_id << std::setw(2) << std::setfill('0') << x;
            reg_x_name = "roi/td_roi_x" + reg_id.str();
            MV_HAL_LOG_DEBUG() << "ROI" << y << "x" << x << ":" << std::hex << std::setw(8) << std::setfill('0')
                               << ~grid_.get_vector(x, y);

            // Send roi x register access, setting all D-Latches data lanes from given user config on current row.
            (*register_map_)[sensor_prefix_ + reg_x_name].write_value(~grid_.get_vector(x, y));
        }

        // Apply configuration to the hardware by triggering D-Latches toggling of their respective output
        (*register_map_)[sensor_prefix_ + "roi_ctrl"]["roi_td_shadow_trigger"].write_value(0);
        (*register_map_)[sensor_prefix_ + "roi_ctrl"]["roi_td_shadow_trigger"].write_value(1);

        // Clear roi y register. Deselect current row
        (*register_map_)[sensor_prefix_ + reg_y_name].write_value(0);
        (*register_map_)[sensor_prefix_ + "roi_ctrl"]["roi_td_shadow_trigger"].write_value(0);
        (*register_map_)[sensor_prefix_ + "roi_ctrl"]["roi_td_shadow_trigger"].write_value(1);
    }

    (*register_map_)[sensor_prefix_ + "roi_ctrl"]["roi_td_shadow_trigger"].write_value(0);
}

bool GenX320RoiDriver::load_calibration_file(const std::filesystem::path &path) {
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
    set_grid(g);
    apply_grid();

    return true;
}

int GenX320RoiDriver::device_width() const {
    return device_width_;
}

int GenX320RoiDriver::device_height() const {
    return device_height_;
}

unsigned int GenX320RoiDriver::get_max_windows_count() const {
    return 18;
}

bool GenX320RoiDriver::set_windows(const std::vector<I_ROI::Window> &windows) {
    roi_window_cnt_ = windows.size();
    main_window_    = windows[0];

    for (unsigned int i = 0; i < roi_window_cnt_; i++) {
        int x_min = windows[i].x;
        int y_min = windows[i].y;
        int x_max = x_min + windows[i].width;
        int y_max = y_min + windows[i].height;

        std::string win_x_coord = "roi_win_x" + std::to_string(i);
        std::string win_y_coord = "roi_win_y" + std::to_string(i);

        (*register_map_)[sensor_prefix_ + win_x_coord].write_value(
            vfield{{"roi_win_start_x", x_min}, {"roi_win_end_p1_x", x_max}});
        (*register_map_)[sensor_prefix_ + win_y_coord].write_value(
            vfield{{"roi_win_start_y", y_min}, {"roi_win_end_p1_y", y_max}});
    }

    is_lines_ = false;
    return true;
}

bool GenX320RoiDriver::set_lines(const std::vector<bool> &cols, const std::vector<bool> &rows) {
    std::vector<uint32_t> columns_sel(10, 0);
    uint32_t val     = 0;
    uint32_t vect_id = 0;

    grid_.clear();

    for (unsigned int i = 0; i <= cols.size(); i++) {
        if ((i % 32) == 0) {
            if (i != 0) {
                columns_sel[vect_id] = val;
                vect_id++;
                val = 0;
            } else if (i == cols.size()) {
                columns_sel[vect_id] = val;
                break;
            }
        }
        val += (cols[i] << (i % 32));
    }

    for (unsigned int y = 0; y < 320; y++) {
        if (rows[y] == true) {
            for (unsigned int x = 0; x < 10; x++) {
                grid_.set_vector(x, y, columns_sel[x]);
            }
        } else {
            for (unsigned int x = 0; x < 10; x++) {
                grid_.set_vector(x, y, 0);
            }
        }
    }

    is_lines_ = true;
    return true;
}

bool GenX320RoiDriver::get_lines(std::vector<bool> &cols, std::vector<bool> &rows) const {
    if (!is_lines_) {
        return false;
    }

    if (cols.size() != static_cast<size_t>(device_width_)) {
        cols = std::vector<bool>(device_width_);
    }
    std::fill(cols.begin(), cols.end(), false);

    if (rows.size() != static_cast<size_t>(device_height_)) {
        rows = std::vector<bool>(device_height_);
    }
    std::fill(rows.begin(), rows.end(), false);

    for (unsigned int y = 0; y < 320; ++y) {
        for (unsigned int vect_id = 0; vect_id < 10; ++vect_id) {
            unsigned int vect_val = grid_.get_vector(vect_id, y);
            for (unsigned int i = 0; i < 32; ++i) {
                if (vect_val & (1U << i)) {
                    cols[32 * vect_id + i] = true;
                    rows[y]                = true;
                }
            }
        }
    }

    return true;
}

void GenX320RoiDriver::apply_windows(unsigned int window_count) {
    (*register_map_)[sensor_prefix_ + "roi_ctrl"]["px_roi_halt_programming"].write_value(0);

    (*register_map_)[sensor_prefix_ + "roi_master_ctrl"].write_value(
        vfield{{"roi_master_en", 1},
               {"roi_master_run", 1},
               {"roi_master_mode", static_cast<std::underlying_type<I_ROI::Mode>::type>(mode_)},
               {"roi_win_nb", window_count}});
}

void GenX320RoiDriver::reset_to_full_roi() {
    auto default_window = I_ROI::Window(0, 0, 320, 320);

    (*register_map_)[sensor_prefix_ + "roi_win_x0"].write_value(
        vfield{{"roi_win_start_x", default_window.x}, {"roi_win_end_p1_x", default_window.width}});
    (*register_map_)[sensor_prefix_ + "roi_win_y0"].write_value(
        vfield{{"roi_win_start_y", default_window.y}, {"roi_win_end_p1_y", default_window.height}});

    set_driver_mode(GenX320RoiDriver::DriverMode::MASTER);

    I_ROI::Mode save_mode = mode_;
    mode_                 = I_ROI::Mode::ROI;
    apply_windows(1);
    mode_ = save_mode;
}

bool GenX320RoiDriver::set_roi_mode(const I_ROI::Mode &mode) {
    mode_ = mode;
    return true;
}

I_ROI::Mode GenX320RoiDriver::get_roi_mode() const {
    return mode_;
}

GenX320RoiDriver::DriverMode GenX320RoiDriver::get_driver_mode() const {
    return driver_mode_;
}

bool GenX320RoiDriver::set_driver_mode(const GenX320RoiDriver::DriverMode &driver_mode) {
    driver_mode_ = driver_mode;

    (*register_map_)[sensor_prefix_ + "roi_ctrl"]["roi_td_en"].write_value(1);
    (*register_map_)[sensor_prefix_ + "roi_ctrl"]["px_iphoto_en"].write_value(0);
    (*register_map_)[sensor_prefix_ + "roi_ctrl"]["px_sw_rstn"].write_value(1);
    (*register_map_)[sensor_prefix_ + "roi_ctrl"]["roi_td_shadow_trigger"].write_value(0);

    if (driver_mode_ == GenX320RoiDriver::DriverMode::MASTER) {
        (*register_map_)[sensor_prefix_ + "roi_master_chicken_bit"]["roi_driver_register_if_en"].write_value(0);
        (*register_map_)[sensor_prefix_ + "roi_ctrl"]["px_roi_halt_programming"].write_value(0);
        (*register_map_)[sensor_prefix_ + "roi_master_ctrl"].write_value(
            vfield{{"roi_master_en", 1}, {"roi_master_run", 0}});
    } else if (driver_mode_ == GenX320RoiDriver::DriverMode::IO) {
        (*register_map_)[sensor_prefix_ + "roi_master_ctrl"]["roi_master_en"].write_value(0);
        (*register_map_)[sensor_prefix_ + "roi_ctrl"]["roi_td_en"].write_value(1);
        (*register_map_)[sensor_prefix_ + "roi_ctrl"]["px_roi_halt_programming"].write_value(1);
        open_all_latches();
    }

    return true;
}

bool GenX320RoiDriver::enable(bool state) {
    if (!state) {
        reset_to_full_roi();
    } else {
        if (driver_mode_ == GenX320RoiDriver::DriverMode::MASTER) {
            // Re-apply first roi
            (*register_map_)[sensor_prefix_ + "roi_win_x0"].write_value(
                vfield{{"roi_win_start_x", main_window_.x}, {"roi_win_end_p1_x", main_window_.x + main_window_.width}});
            (*register_map_)[sensor_prefix_ + "roi_win_y0"].write_value(vfield{
                {"roi_win_start_y", main_window_.y}, {"roi_win_end_p1_y", main_window_.y + main_window_.height}});

            apply_windows(roi_window_cnt_);
        } else if (driver_mode_ == GenX320RoiDriver::DriverMode::LATCH) {
            apply_grid();
        }
    }

    return true;
}

std::vector<I_ROI::Window> GenX320RoiDriver::get_windows() const {
    std::vector<I_ROI::Window> roi_win_list;

    if (is_lines_) {
        return std::vector<I_ROI::Window>();
    }

    for (unsigned int i = 0; i < roi_window_cnt_; i++) {
        std::string win_x_coord = "roi_win_x" + std::to_string(i);
        std::string win_y_coord = "roi_win_y" + std::to_string(i);

        auto x_min = (*register_map_)[sensor_prefix_ + win_x_coord]["roi_win_start_x"].read_value();
        auto x_max = (*register_map_)[sensor_prefix_ + win_x_coord]["roi_win_end_p1_x"].read_value();

        auto y_min = (*register_map_)[sensor_prefix_ + win_y_coord]["roi_win_start_y"].read_value();
        auto y_max = (*register_map_)[sensor_prefix_ + win_y_coord]["roi_win_end_p1_y"].read_value();

        I_ROI::Window my_win(x_min, y_min, x_max - x_min, y_max - y_min);
        roi_win_list.push_back(my_win);
    }
    return roi_win_list;
}

void GenX320RoiDriver::print_windows_config() {
    std::cout << "Windows cnt = " << roi_window_cnt_ << std::endl;

    auto win_list = get_windows();

    for (unsigned int i = 0; i < roi_window_cnt_; i++) {
        auto window = win_list[i];

        std::cout << "Window " << i << " = " << window.x << ", " << window.y << " (" << window.width << ", "
                  << window.height << ")" << std::endl;
    }
}

GenX320RoiDriver::Grid GenX320RoiDriver::get_grid() const {
    return grid_;
}

void GenX320RoiDriver::print_grid_config() {
    std::cout << grid_.to_string() << std::endl;
}

void GenX320RoiDriver::pixel_reset(const bool &enable) {
    (*register_map_)[sensor_prefix_ + "roi_ctrl"]["px_sw_rstn"].write_value(!enable);
}

void GenX320RoiDriver::open_all_latches() {
    std::string reg_y_name;
    std::string reg_x_name;
    unsigned int reg_val = 0;

    // Iterate over each columns composed of 32 bits vectors
    for (unsigned int x = 0; x < 10; x++) {
        // Compute roi_x register name from column index
        std::ostringstream reg_id;
        reg_id << std::setw(2) << std::setfill('0') << x;
        reg_x_name = "roi/td_roi_x" + reg_id.str();

        // Send roi x register access, enabling all D-Latches data lanes
        (*register_map_)[sensor_prefix_ + reg_x_name].write_value(0);
    }

    // Iterate over each rows
    for (unsigned int y = 0; y < 10; y++) {
        // Compute roi_y register name from row index
        std::ostringstream reg_id;
        reg_id << std::setw(2) << std::setfill('0') << y;
        reg_y_name = "roi/td_roi_y" + reg_id.str();

        // Send roi y register access, enabling all D-Latches
        (*register_map_)[sensor_prefix_ + reg_y_name].write_value(0xFFFFFFFF);
    }

    // Apply configuration to the hardware by triggering D-Latches toggling of their respective output
    (*register_map_)[sensor_prefix_ + "roi_ctrl"]["roi_td_shadow_trigger"].write_value(0);
    (*register_map_)[sensor_prefix_ + "roi_ctrl"]["roi_td_shadow_trigger"].write_value(1);

    // Clear trigger
    (*register_map_)[sensor_prefix_ + "roi_ctrl"]["roi_td_shadow_trigger"].write_value(0);
}

} // namespace Metavision
