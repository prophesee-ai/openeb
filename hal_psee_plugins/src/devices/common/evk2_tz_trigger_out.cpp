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

#include <cstdint>

#include "metavision/hal/facilities/i_trigger_out.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/psee_hw_layer/devices/common/evk2_tz_trigger_out.h"
#include "metavision/psee_hw_layer/devices/psee-video/tz_psee_video.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {

Evk2TzTriggerOut::Evk2TzTriggerOut(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix,
                                   const std::shared_ptr<TzPseeVideo> tzDev) :
    register_map_(regmap), prefix_(prefix), tz_dev_(tzDev) {
    disable();
}

Evk2TzTriggerOut::~Evk2TzTriggerOut() {
    try {
        disable();
    } catch (...) {}
}

bool Evk2TzTriggerOut::enable() {
    if (tz_dev_->get_mode() == I_CameraSynchronization::SyncMode::MASTER) {
        MV_HAL_LOG_WARNING() << "Master sync mode is enabled. Cannot enable trigger out.";
        return false;
    }

    (*register_map_)[prefix_ + "SYSTEM_CONTROL/IO_CONTROL"]["SYNC_OUT_MODE"].write_value(1);
    (*register_map_)[prefix_ + "SYSTEM_CONTROL/IO_CONTROL"]["SYNC_OUT_EN_HSIDE"].write_value(1);
    (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/OUT_ENABLE"]["VALUE"].write_value(1);
    return true;
}

bool Evk2TzTriggerOut::disable() {
    (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/OUT_ENABLE"]["VALUE"].write_value(0);

    if (tz_dev_->get_mode() != I_CameraSynchronization::SyncMode::MASTER) {
        (*register_map_)[prefix_ + "SYSTEM_CONTROL/IO_CONTROL"]["SYNC_OUT_MODE"].write_value(0);
        (*register_map_)[prefix_ + "SYSTEM_CONTROL/IO_CONTROL"]["SYNC_OUT_EN_HSIDE"].write_value(0);
    } else {
        MV_HAL_LOG_DEBUG() << "Master sync mode is enabled. SYNC_OUT_MODE/EN_HSIDE config will not be changed.";
        return false;
    }
    return true;
}

bool Evk2TzTriggerOut::set_duty_cycle(double period_ratio) {
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

    return true;
}

double Evk2TzTriggerOut::get_duty_cycle() const {
    return period_ratio_;
}

bool Evk2TzTriggerOut::set_period(uint32_t signal_period_us) {
    (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/OUT_PULSE_PERIOD"].write_value(signal_period_us);
    /* reapply duty cycle in order to update pulse width */
    set_duty_cycle(period_ratio_);

    return true;
}

uint32_t Evk2TzTriggerOut::get_period() const {
    return (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/OUT_PULSE_PERIOD"].read_value();
}

bool Evk2TzTriggerOut::is_enabled() const {
    bool sync_out = (*register_map_)[prefix_ + "SYSTEM_CONTROL/IO_CONTROL"]["SYNC_OUT_MODE"].read_value();
    bool hside_en = (*register_map_)[prefix_ + "SYSTEM_CONTROL/IO_CONTROL"]["SYNC_OUT_EN_HSIDE"].read_value();
    bool out_en   = (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/OUT_ENABLE"].read_value();
    return out_en && sync_out && hside_en;
}

} // namespace Metavision
