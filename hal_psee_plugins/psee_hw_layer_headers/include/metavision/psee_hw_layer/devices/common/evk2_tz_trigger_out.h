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

#ifndef METAVISION_HAL_EVK2_TZ_TRIGGER_OUT_H
#define METAVISION_HAL_EVK2_TZ_TRIGGER_OUT_H

#include <string>
#include "metavision/hal/facilities/i_trigger_out.h"

namespace Metavision {

class TzPseeVideo;
class RegisterMap;

class Evk2TzTriggerOut : public I_TriggerOut {
public:
    Evk2TzTriggerOut(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix,
                     const std::shared_ptr<TzPseeVideo> tzDev);

    ~Evk2TzTriggerOut();

    bool enable() override final;
    bool disable() override final;
    bool set_period(uint32_t period_us) override final;
    uint32_t get_period() const override final;
    bool set_duty_cycle(double period_ratio) override final;
    double get_duty_cycle() const override final;
    bool is_enabled() override final;

private:
    double period_ratio_ = 0.5;
    std::shared_ptr<RegisterMap> register_map_;
    std::string prefix_;
    std::shared_ptr<TzPseeVideo> tz_dev_;
};

} // namespace Metavision

#endif // METAVISION_HAL_EVK2_TZ_TRIGGER_OUT_H
