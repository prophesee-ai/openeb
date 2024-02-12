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

#include "metavision/psee_hw_layer/devices/gen41/gen41_roi_command.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {

Gen41ROICommand::Gen41ROICommand(int width, int height, const std::shared_ptr<RegisterMap> &regmap,
                                 const std::string &sensor_prefix) :
    PseeROI(width, height), register_map_(regmap), sensor_prefix_(sensor_prefix), mode_(I_ROI::Mode::ROI) {
    roi_save_ = create_ROIs(std::vector<I_ROI::Window>({I_ROI::Window(0, 0, width, height)}));
    reset_to_full_roi();
}

void Gen41ROICommand::reset_to_full_roi() {
    constexpr int roi_step = 0x004;
    uint32_t col_td_ind, row_td_ind;
    uint32_t td_roi_x00_addr, td_roi_x39_addr;
    uint32_t td_roi_y00_addr, td_roi_y22_addr;

    // setting x registers
    td_roi_x00_addr = (*register_map_)[sensor_prefix_ + "roi/td_roi_x00"].get_address();
    td_roi_x39_addr = (*register_map_)[sensor_prefix_ + "roi/td_roi_x39"].get_address();
    for (col_td_ind = td_roi_x00_addr; col_td_ind <= td_roi_x39_addr; col_td_ind += roi_step) {
        (*register_map_)[col_td_ind]["effective"] = "enable";
    }

    // setting y registers
    td_roi_y00_addr = (*register_map_)[sensor_prefix_ + "roi/td_roi_y00"].get_address();
    td_roi_y22_addr = (*register_map_)[sensor_prefix_ + "roi/td_roi_y22"].get_address();
    for (row_td_ind = td_roi_y00_addr; row_td_ind <= td_roi_y22_addr; row_td_ind += roi_step) {
        (*register_map_)[row_td_ind]["effective"] = "enable";
    }
}

void Gen41ROICommand::write_ROI(const std::vector<unsigned int> &vroiparams) {
    constexpr int roi_step = 0x004;
    uint32_t param_ind     = 0;
    uint32_t col_td_ind, row_td_ind;
    roi_save_ = vroiparams;
    uint32_t td_roi_x00_addr, td_roi_x39_addr;
    uint32_t td_roi_y00_addr, td_roi_y22_addr;

    td_roi_x00_addr = (*register_map_)[sensor_prefix_ + "roi/td_roi_x00"].get_address();
    td_roi_x39_addr = (*register_map_)[sensor_prefix_ + "roi/td_roi_x39"].get_address();

    td_roi_y00_addr = (*register_map_)[sensor_prefix_ + "roi/td_roi_y00"].get_address();
    td_roi_y22_addr = (*register_map_)[sensor_prefix_ + "roi/td_roi_y22"].get_address();
    uint32_t xsize  = ((td_roi_x39_addr - td_roi_x00_addr) / roi_step) + 1;
    uint32_t ysize  = ((td_roi_y22_addr - td_roi_y00_addr) / roi_step) + 1;
    if (vroiparams.size() != (xsize + ysize)) {
        MV_HAL_LOG_WARNING() << "Error setting ROI.";
    }

    // setting x registers
    for (col_td_ind = td_roi_x00_addr; col_td_ind <= td_roi_x39_addr; col_td_ind += roi_step, ++param_ind) {
        (*register_map_)[col_td_ind] = ~vroiparams[param_ind];
    }

    // setting y registers
    for (row_td_ind = td_roi_y00_addr; row_td_ind <= td_roi_y22_addr; row_td_ind += roi_step, ++param_ind) {
        uint32_t v = ~vroiparams[param_ind];
        if (row_td_ind == td_roi_y22_addr) {
            v |= 0xffff0000;
        }
        (*register_map_)[row_td_ind] = v;
    }
}

bool Gen41ROICommand::set_mode(const I_ROI::Mode &mode) {
    mode_ = mode;
    return true;
}

I_ROI::Mode Gen41ROICommand::get_mode() const {
    return mode_;
}

bool Gen41ROICommand::enable(bool state) {
    enabled_ = state;

    (*register_map_)[sensor_prefix_ + "roi_ctrl"].write_value(vfield{{"roi_td_en", state},
                                                                     {"td_roi_roni_n_en", (mode_ == I_ROI::Mode::ROI)},
                                                                     {"px_td_rstn", 1},
                                                                     {"roi_td_shadow_trigger", 1}});
    (*register_map_)[sensor_prefix_ + "roi_win_ctrl"]["roi_master_en"].write_value(0);
    (*register_map_)[sensor_prefix_ + "roi_win_ctrl"]["roi_win_done"].write_value(0);

    return true;
}

bool Gen41ROICommand::write_ROI_windows(const std::vector<Window> &windows) {
    if (windows.empty()) {
        return true;
    }

    // Only one ROI supported in window mode
    auto &window = windows[0];

    if (mode_ == Mode::ROI) {
        (*register_map_)[sensor_prefix_ + "roi_win_start_addr"]["roi_win_start_x"].write_value(window.x);
        (*register_map_)[sensor_prefix_ + "roi_win_start_addr"]["roi_win_start_y"].write_value(window.y);
        (*register_map_)[sensor_prefix_ + "roi_win_end_addr"]["roi_win_end_x"].write_value(window.x + window.width);
        (*register_map_)[sensor_prefix_ + "roi_win_end_addr"]["roi_win_end_y"].write_value(window.y + window.height);

        (*register_map_)[sensor_prefix_ + "roi_win_ctrl"]["roi_master_en"].write_value(1);
        while (!(*register_map_)[sensor_prefix_ + "roi_win_ctrl"]["roi_win_done"].read_value()) {}
    } else {
        // RONI window mode doesn't behave as expected, so use lines to setup a proper RONI window
        std::vector<bool> cols(device_width_, true);
        std::vector<bool> rows(device_height_, true);

        for (int i = window.x; i < window.x + window.width; ++i) {
            cols[i] = false;
        }
        for (int i = window.y; i < window.y + window.height; ++i) {
            rows[i] = false;
        }

        auto windows = lines_to_windows(cols, rows);
        write_ROI(create_ROIs(windows));
    }

    return true;
}

bool Gen41ROICommand::is_enabled() const {
    return enabled_;
}

} // namespace Metavision
