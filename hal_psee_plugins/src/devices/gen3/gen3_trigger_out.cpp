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

#ifdef _MSC_VER
#define NOMINMAX // libusb.h includes windows.h which defines min max macros that we don't want
#endif

#include <algorithm>
#include <cstdint>

#include "facilities/psee_device_control.h"
#include "boards/utils/psee_libusb_board_command.h"
#include "devices/gen3/gen3_trigger_out.h"
#include "devices/gen3/legacy_regmap_headers/ccam3_single_gen3.h"

static constexpr uint32_t MAX_SIGNAL_PERIOD_US_FPGA_OLDER_THAN_3_0 = (1 << 8) - 1;

namespace Metavision {

Gen3TriggerOut::Gen3TriggerOut(const std::shared_ptr<PseeLibUSBBoardCommand> &board_cmd,
                               const std::shared_ptr<PseeDeviceControl> &device_control) :
    PseeTriggerOut(device_control), board_command_(board_cmd) {
    disable();
}

bool Gen3TriggerOut::enable() {
    if (get_device_control()->get_mode() == I_DeviceControl::SyncMode::MASTER) {
        return false;
    }
    board_command_->send_register_bit(CCAM3_EXT_SYNC_OUT_MODE_ADDR, CCAM3_EXT_SYNC_OUT_MODE_VALUE_BIT_IDX, 1);

    board_command_->send_register_bit(CCAM3_SYSTEM_MONITOR_EXT_TRIGGERS_OUT_ENABLE_ADDR,
                                      CCAM3_SYSTEM_MONITOR_EXT_TRIGGERS_OUT_ENABLE_VALUE_BIT_IDX, 1);
    return true;
}

void Gen3TriggerOut::disable() {
    board_command_->send_register_bit(CCAM3_EXT_SYNC_OUT_MODE_ADDR, CCAM3_EXT_SYNC_OUT_MODE_VALUE_BIT_IDX, 0);

    board_command_->send_register_bit(CCAM3_SYSTEM_MONITOR_EXT_TRIGGERS_OUT_ENABLE_ADDR,
                                      CCAM3_SYSTEM_MONITOR_EXT_TRIGGERS_OUT_ENABLE_VALUE_BIT_IDX, 0);
}

void Gen3TriggerOut::set_duty_cycle(double period_ratio) {
    /*
     * Convert the ratio (between 0 and 1) to a signal duration
     * -> if ratio is 0.5 i.e. 50%, then the pulse will last 50% of the pulse period
     * -> example:
     *  pulse period is 1000us
     *  duty cycle is 0.5
     *  Then the signal will last 500us every 1000 us.
     */
    period_ratio_ = std::min(1., std::max(0., period_ratio));
    const auto signal_period_us =
        board_command_->read_register(CCAM3_SYSTEM_MONITOR_EXT_TRIGGERS_OUT_PULSE_PERIOD_ADDR);

    board_command_->write_register(CCAM3_SYSTEM_MONITOR_EXT_TRIGGERS_OUT_PULSE_WIDTH_ADDR,
                                   signal_period_us * period_ratio_);
}

void Gen3TriggerOut::set_period(uint32_t signal_period_us) {
    /*
     * Triggers out on 32 bits are supported only from fpga version 3.0.
     * Before trigger out were supported only on Pronto and with 8 bits resolutions for the duty cycle
     * With newer system version, the resolution is 32 bits
     */
    if (board_command_->get_system_version() < 0x3000) {
        signal_period_us = std::max(2u, std::min(signal_period_us, MAX_SIGNAL_PERIOD_US_FPGA_OLDER_THAN_3_0));
    }

    board_command_->write_register(CCAM3_SYSTEM_MONITOR_EXT_TRIGGERS_OUT_PULSE_PERIOD_ADDR, signal_period_us);
    /* reapply duty cycle in order to update pulse width */
    set_duty_cycle(period_ratio_);
}

bool Gen3TriggerOut::is_enabled() {
    bool sync_out_en =
        board_command_->read_register_bit(CCAM3_EXT_SYNC_OUT_MODE_ADDR, CCAM3_EXT_SYNC_OUT_MODE_VALUE_BIT_IDX);

    bool trig_out_en = board_command_->read_register_bit(CCAM3_SYSTEM_MONITOR_EXT_TRIGGERS_OUT_ENABLE_ADDR,
                                                         CCAM3_SYSTEM_MONITOR_EXT_TRIGGERS_OUT_ENABLE_VALUE_BIT_IDX);
    return sync_out_en && trig_out_en;
}

} // namespace Metavision
