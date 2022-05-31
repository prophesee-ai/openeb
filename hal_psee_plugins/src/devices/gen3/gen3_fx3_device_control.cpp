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

#include "metavision/hal/utils/hal_log.h"
#include "boards/utils/psee_libusb_board_command.h"
#include "devices/gen3/legacy_regmap_headers/tep_register_control_register_map.h"
#include "devices/gen3/legacy_regmap_headers/legacy/stereo_pc_mapping.h"
#include "devices/gen3/gen3_fx3_device_control.h"

namespace Metavision {

Gen3Fx3DeviceControl::Gen3Fx3DeviceControl(const std::shared_ptr<PseeLibUSBBoardCommand> &board_cmd) :
    Gen3DeviceControl(board_cmd) {}

void Gen3Fx3DeviceControl::enable_interface(bool state) {
    MV_HAL_LOG_DEBUG() << "-------------- Disable FX3 interface";
    MV_HAL_LOG_DEBUG() << Metavision::Log::no_space << std::hex << std::showbase << std::internal << std::setfill('0')
                       << get_base_address() + CCAM2_CONTROL_ADDR << std::dec << "\t|\t"
                       << TEP_CCAM2_CONTROL_HOST_IF_ENABLE_BIT_IDX << std::dec << " " << (state ? 1 : 0);
    icmd_->send_register_bit(get_base_address() + CCAM2_CONTROL_ADDR, TEP_CCAM2_CONTROL_HOST_IF_ENABLE_BIT_IDX,
                             state); // Enable FX3 interface
}

void Gen3Fx3DeviceControl::start_impl() {
    bool gen3EM = is_gen3EM();
    start_camera_common_0(gen3EM);
    enable_interface(true);

    start_camera_common_1(gen3EM);
#ifdef __ANDROID__
    icmd_->write_register(get_base_address() + CCAM3_FX3_HOST_IF_PKT_END_ENABLE_ADDR, 0x0);
#endif
}

void Gen3Fx3DeviceControl::stop_impl() {
    enable_interface(false);
    stop_camera_common();
}

void Gen3Fx3DeviceControl::initialize() {
    Gen3DeviceControl::initialize();
    initialize_common_0();
    stop_impl();
    initialize_common_1();
}

void Gen3Fx3DeviceControl::destroy() {
    destroy_camera();
    Gen3DeviceControl::destroy();
}

bool Gen3Fx3DeviceControl::set_evt_format_impl(EvtFormat fmt) {
    return false;
}

void Gen3Fx3DeviceControl::reset_ts_internal() {
    // Performs a hard reset (via FPGA pin) if FX3 FW version is at or above 1.3.0
    // Otherwise performs a soft reset by setting the TEP_TRIGGER_SOFT_RESET_BIT
    if (icmd_->get_board_release_version() >= 0x00010300) {
        // -------------- HARD RESET
        MV_HAL_LOG_DEBUG() << "-------------- HARD RESET";
        icmd_->reset_fpga();
    }
    // -------------- SOFT RESET
    Gen3DeviceControl::reset_ts_internal();
}

} // namespace Metavision
