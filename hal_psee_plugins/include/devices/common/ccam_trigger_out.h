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

#ifndef METAVISION_HAL_CCAM_TRIGGER_OUT_H
#define METAVISION_HAL_CCAM_TRIGGER_OUT_H

#include <string>
#include "facilities/psee_trigger_out.h"

namespace Metavision {

class PseeDeviceControl;
class RegisterMap;

class CCamTriggerOut : public PseeTriggerOut {
public:
    CCamTriggerOut(const std::shared_ptr<RegisterMap> &regmap, const std::shared_ptr<PseeDeviceControl> &device_control,
                   const std::string &prefix);

    bool enable() override final;
    void disable() override final;
    void set_period(uint32_t period_us) override final;
    void set_duty_cycle(double period_ratio) override final;
    bool is_enabled() override final;

private:
    double period_ratio_ = 0.5;
    std::shared_ptr<RegisterMap> register_map_;
    std::string prefix_;
};

} // namespace Metavision

#endif // METAVISION_HAL_CCAM_TRIGGER_OUT_H
