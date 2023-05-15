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
    uint32_t td_roi_x00_addr, td_roi_x39_addr, td_roi_x40_addr;
    uint32_t td_roi_y00_addr, td_roi_y22_addr;

    td_roi_x00_addr = (*register_map_)[sensor_prefix_ + "roi/td_roi_x00"].get_address();
    td_roi_x39_addr = (*register_map_)[sensor_prefix_ + "roi/td_roi_x39"].get_address();

    td_roi_y00_addr = (*register_map_)[sensor_prefix_ + "roi/td_roi_y00"].get_address();
    td_roi_y22_addr = (*register_map_)[sensor_prefix_ + "roi/td_roi_y22"].get_address();
    uint32_t xsize  = ((td_roi_x39_addr - td_roi_x00_addr) / roi_step) + 1;
    uint32_t ysize  = ((td_roi_y22_addr - td_roi_y00_addr) / roi_step) + 1;
    if (vroiparams.size() != (xsize + ysize)) {
        MV_HAL_LOG_WARNING() << "Error setting roi for Gen 41 sensor.";
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

bool Gen41ROICommand::enable(bool state) {
    write_ROI(roi_save_);
    if (!state) {
        reset_to_full_roi();

    } else {
        write_ROI(roi_save_);
    }

    (*register_map_)[sensor_prefix_ + "roi_ctrl"].write_value(vfield{{"roi_td_en", 1},
                                                                     {"td_roi_roni_n_en", (mode_ == I_ROI::Mode::ROI)},
                                                                     {"px_td_rstn", 1},
                                                                     {"roi_td_shadow_trigger", 1}});

    return true;
}

} // namespace Metavision
