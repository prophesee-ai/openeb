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
#include "boards/utils/psee_libusb_board_command.h"
#include "devices/gen3/gen3_roi_command.h"
#include "devices/gen3/legacy_regmap_headers/legacy/stereo_pc_mapping.h"

namespace Metavision {

Gen3ROICommand::Gen3ROICommand(int width, int height, const std::shared_ptr<PseeLibUSBBoardCommand> &board_cmd) :
    PseeROI(width, height), icmd_(board_cmd), base_sensor_address_(CCAM3_SENSOR_IF_BASE_ADDR) {
    /// this step is necessary because of a flaw in sensor conception that enables half ROI by default.
    /// If the ROI is not set to the full pixel matrix, then a ring of width 1 appears.
    enable(false);
}

void Gen3ROICommand::reset_to_full_roi() {
    constexpr int roi_step = 0x004;
    uint32_t col_td_ind, row_td_ind;

    // setting x registers
    for (col_td_ind = CCAM3_SISLEY_ROI_TD_X_START_ADDR; col_td_ind < CCAM3_SISLEY_ROI_TD_X_LAST_ADDR;
         col_td_ind += roi_step) {
        MV_HAL_LOG_DEBUG() << Metavision::Log::no_space << std::hex << std::showbase << std::internal
                           << std::setfill('0') << col_td_ind << "\t|\t" << 0XFFFFFFFF << std::dec;
        icmd_->write_register(col_td_ind, 0xFFFFFFFF);
    }

    // setting y registers
    for (row_td_ind = CCAM3_SISLEY_ROI_TD_Y_START_ADDR; row_td_ind < CCAM3_SISLEY_ROI_TD_Y_LAST_ADDR;
         row_td_ind += roi_step) {
        MV_HAL_LOG_DEBUG() << Metavision::Log::no_space << std::hex << std::showbase << std::internal
                           << std::setfill('0') << row_td_ind << "\t|\t" << 0XFFFFFFFF << std::dec;
        icmd_->write_register(row_td_ind, 0XFFFFFFFF);
    }
}

void Gen3ROICommand::write_ROI(const std::vector<unsigned int> &vroiparams) {
    constexpr int roi_step = 0x004;
    uint32_t param_ind     = 0;
    uint32_t col_td_ind, row_td_ind;
    roi_save_ = vroiparams;

    if (vroiparams.size() != 35) {
        MV_HAL_LOG_WARNING() << "Data provided to write ROI is not of the good size for Gen3 sensor";
        return;
    }

    // setting x registers
    for (col_td_ind = CCAM3_SISLEY_ROI_TD_X_START_ADDR; col_td_ind < CCAM3_SISLEY_ROI_TD_X_LAST_ADDR;
         col_td_ind += roi_step, ++param_ind) {
        MV_HAL_LOG_DEBUG() << Metavision::Log::no_space << std::hex << std::showbase << std::internal
                           << std::setfill('0') << col_td_ind << "\t|\t" << vroiparams[param_ind] << std::dec;
        icmd_->write_register(col_td_ind, vroiparams[param_ind]);
    }

    // setting y registers
    for (row_td_ind = CCAM3_SISLEY_ROI_TD_Y_START_ADDR; row_td_ind < CCAM3_SISLEY_ROI_TD_Y_LAST_ADDR;
         row_td_ind += roi_step, ++param_ind) {
        MV_HAL_LOG_DEBUG() << Metavision::Log::no_space << std::hex << std::showbase << std::internal
                           << std::setfill('0') << row_td_ind << "\t|\t" << vroiparams[param_ind] << std::dec
                           << std::endl;
        icmd_->write_register(row_td_ind, vroiparams[param_ind]);
    }
}

void Gen3ROICommand::enable(bool state) {
    if (!state) {
        reset_to_full_roi();
    } else {
        write_ROI(roi_save_);
    }

    icmd_->write_register(base_sensor_address_ + SISLEY_SENSOR_ROI_CTRL_ADDR, SISLEY_SENSOR_ROI_CTRL);
    icmd_->send_register_bit(base_sensor_address_ + SISLEY_SENSOR_ROI_CTRL_ADDR,
                             SISLEY_SENSOR_ROI_CTRL_ROI_TD_EN_BIT_IDX, 1);
    icmd_->send_register_bit(base_sensor_address_ + SISLEY_SENSOR_ROI_CTRL_ADDR,
                             SISLEY_SENSOR_ROI_CTRL_ROI_TD_SHADOW_TRIG_BIT_IDX, 1);
}

} // namespace Metavision
