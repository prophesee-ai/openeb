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

#include <algorithm>
#include <cstdint>

#include "devices/gen31/gen31_ccam5_trigger_out.h"
#include "devices/gen31/gen31_ccam5_tz_device.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

static constexpr uint32_t MAX_SIGNAL_PERIOD_US_FPGA_OLDER_THAN_3_0 = (1 << 8) - 1;

namespace Metavision {

Gen31Ccam5TriggerOut::Gen31Ccam5TriggerOut(const std::shared_ptr<RegisterMap> &regmap,
                                           const std::shared_ptr<TzCcam5Gen31> &dev) :
    tz_dev_(dev), register_map_(regmap) {
    disable();
}

bool Gen31Ccam5TriggerOut::enable() {
    if (tz_dev_->get_mode() == I_CameraSynchronization::SyncMode::MASTER)
        return false;
    (*register_map_)["SYSTEM_MONITOR/EXT_TRIGGERS/OUT_ENABLE"] = 1;
    return true;
}

bool Gen31Ccam5TriggerOut::disable() {
    if (tz_dev_->get_mode() == I_CameraSynchronization::SyncMode::MASTER)
        return false;
    (*register_map_)["SYSTEM_MONITOR/EXT_TRIGGERS/OUT_ENABLE"] = 0;
    return true;
}

bool Gen31Ccam5TriggerOut::set_duty_cycle(double period_ratio) {
    /*
     * Convert the ratio (between 0 and 1) to a signal duration
     * -> if ratio is 0.5 i.e. 50%, then the pulse will last 50% of the pulse period
     * -> example:
     *  pulse period is 1000us
     *  duty cycle is 0.5
     *  Then the signal will last 500us every 1000 us.
     */
    period_ratio_               = std::min(1., std::max(0., period_ratio));
    const auto signal_period_us = (*register_map_)["SYSTEM_MONITOR/EXT_TRIGGERS/OUT_PULSE_PERIOD"].read_value();

    (*register_map_)["SYSTEM_MONITOR/EXT_TRIGGERS/OUT_PULSE_WIDTH"] =
        static_cast<uint32_t>(signal_period_us * period_ratio_);

    return true;
}

double Gen31Ccam5TriggerOut::get_duty_cycle() const {
    return period_ratio_;
}

bool Gen31Ccam5TriggerOut::set_period(uint32_t signal_period_us) {
    /*
     * Triggers out on 32 bits are supported only from fpga version 3.0.
     * Before trigger out were supported only on Pronto and with 8 bits resolutions for the duty cycle
     * With newer system version, the resolution is 32 bits
     */
    if ((*register_map_)["SYSTEM_CONFIG/VERSION"].read_value() < 0x3000) {
        signal_period_us = std::max(2u, std::min(signal_period_us, MAX_SIGNAL_PERIOD_US_FPGA_OLDER_THAN_3_0));
    }

    (*register_map_)["SYSTEM_MONITOR/EXT_TRIGGERS/OUT_PULSE_PERIOD"] = signal_period_us;
    /* reapply duty cycle in order to update pulse width */
    set_duty_cycle(period_ratio_);

    return true;
}

uint32_t Gen31Ccam5TriggerOut::get_period() const {
    return (*register_map_)["SYSTEM_MONITOR/EXT_TRIGGERS/OUT_PULSE_PERIOD"].read_value();
}

bool Gen31Ccam5TriggerOut::is_enabled() const {
    return (*register_map_)["SYSTEM_MONITOR/EXT_TRIGGERS/OUT_ENABLE"].read_value();
}

} // namespace Metavision
