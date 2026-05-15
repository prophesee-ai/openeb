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

#include <iomanip>
#include <iostream>

#include "metavision/hal/utils/hal_log.h"
#include "metavision/psee_hw_layer/boards/fx3/fx3_libusb_board_command.h"

#include "metavision/psee_hw_layer/devices/gen31/gen31_roi_command.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {

Gen31ROICommand::Gen31ROICommand(int width, int height, const std::shared_ptr<RegisterMap> &regmap,
                                 const std::string &prefix) :
    PseeROI(width, height), register_map_(regmap), prefix_(prefix) {
    /// this step is necessary because of a flaw in sensor conception that enables half ROI by default.
    /// If the ROI is not set to the full pixel matrix, then a ring of width 1 appears
    enable(false);
}

void Gen31ROICommand::reset_to_full_roi() {
    constexpr int roi_step = 0x004;
    uint32_t col_td_ind, row_td_ind;
    uint32_t td_start, td_end;

    // setting x registers
    td_start = (*register_map_)[prefix_ + "td_roi_x00"].get_address();
    td_end   = (*register_map_)[prefix_ + "td_roi_x20"].get_address();
    for (col_td_ind = td_start; col_td_ind < td_end; col_td_ind += roi_step) {
        (*register_map_)[col_td_ind] = 0xFFFFFFFF;
    }

    // setting y registers
    td_start = (*register_map_)[prefix_ + "td_roi_y00"].get_address();
    td_end   = (*register_map_)[prefix_ + "td_roi_y15"].get_address();
    for (row_td_ind = td_start; row_td_ind < td_end; row_td_ind += roi_step) {
        (*register_map_)[row_td_ind] = 0xFFFFFFFF;
    }
}

void Gen31ROICommand::write_ROI(const std::vector<unsigned int> &vroiparams) {
    constexpr int roi_step = 0x004;
    uint32_t param_ind     = 0;
    uint32_t col_td_ind, row_td_ind;
    uint32_t td_start, td_end;
    roi_save_ = vroiparams;

    if (vroiparams.size() != 35) {
        MV_HAL_LOG_WARNING() << "Data provided to write ROI is not of the good size for Gen31 sensor";
        return;
    }

    // setting x registers
    td_start = (*register_map_)[prefix_ + "td_roi_x00"].get_address();
    td_end   = (*register_map_)[prefix_ + "td_roi_x20"].get_address();
    for (col_td_ind = td_start; col_td_ind < td_end; col_td_ind += roi_step, ++param_ind) {
        (*register_map_)[col_td_ind] = vroiparams[param_ind];
    }

    // setting y registers
    td_start = (*register_map_)[prefix_ + "td_roi_y00"].get_address();
    td_end   = (*register_map_)[prefix_ + "td_roi_y15"].get_address();
    for (row_td_ind = td_start; row_td_ind < td_end; row_td_ind += roi_step, ++param_ind) {
        (*register_map_)[row_td_ind] = vroiparams[param_ind];
    }
}

bool Gen31ROICommand::enable(bool state) {
    if (!state) {
        reset_to_full_roi();
    } else {
        write_ROI(roi_save_);
    }

    enabled_ = state;

    (*register_map_)[prefix_ + "roi_ctrl"]["roi_td_en"]             = true;
    (*register_map_)[prefix_ + "roi_ctrl"]["roi_td_shadow_trigger"] = true;
    (*register_map_)[prefix_ + "roi_ctrl"]["roi_td_shadow_trigger"] = false;

    return true;
}

bool Gen31ROICommand::is_enabled() const {
    return enabled_;
}

} // namespace Metavision
