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

#include "devices/common/ccam_trigger_out.h"
#include "facilities/psee_device_control.h"
#include "utils/register_map.h"

namespace Metavision {

CCamTriggerOut::CCamTriggerOut(const std::shared_ptr<RegisterMap> &regmap,
                               const std::shared_ptr<PseeDeviceControl> &device_control, const std::string &prefix) :
    PseeTriggerOut(device_control), prefix_(prefix), register_map_(regmap) {
    disable();
}

bool CCamTriggerOut::enable() {
    if (get_device_control()->get_mode() == I_DeviceControl::SyncMode::MASTER)
        return false;

    if (prefix_.empty()) {
        // Gen31 evk1
        (*register_map_)["SYSTEM_CONTROL/EXT_SYNC_OUT_MODE"].write_value(1);
    } else {
        // Gen4 evk1
        (*register_map_)[prefix_ + "SYSTEM_CONTROL/TIME_BASE_CONTROL"]["EXT_SYNC_OUT_TRIGGER_MODE"].write_value(1);
    }
    (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/OUT_ENABLE"].write_value(1);
    return true;
}

void CCamTriggerOut::disable() {
    if (prefix_.empty()) {
        (*register_map_)["SYSTEM_CONTROL/EXT_SYNC_OUT_MODE"].write_value(0);
    } else {
        (*register_map_)[prefix_ + "SYSTEM_CONTROL/TIME_BASE_CONTROL"]["EXT_SYNC_OUT_TRIGGER_MODE"].write_value(0);
    }
    (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/OUT_ENABLE"].write_value(0);
}

void CCamTriggerOut::set_duty_cycle(double period_ratio) {
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
        (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/OUT_PULSE_PERIOD"].read_value();

    (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/OUT_PULSE_WIDTH"].write_value(signal_period_us *
                                                                                          period_ratio_);
}

void CCamTriggerOut::set_period(uint32_t signal_period_us) {
    (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/OUT_PULSE_PERIOD"].write_value(signal_period_us);
    /* reapply duty cycle in order to update pulse width */
    set_duty_cycle(period_ratio_);
}

bool CCamTriggerOut::is_enabled() {
    bool sync_out =
        prefix_.empty() ?
            (*register_map_)["SYSTEM_CONTROL/EXT_SYNC_OUT_MODE"].read_value() :
            (*register_map_)[prefix_ + "SYSTEM_CONTROL/TIME_BASE_CONTROL"]["EXT_SYNC_OUT_TRIGGER_MODE"].read_value();
    bool out_en = (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/OUT_ENABLE"].read_value();
    return out_en && sync_out;
}

} // namespace Metavision
