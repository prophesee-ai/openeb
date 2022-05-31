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

#ifndef METAVISION_HAL_GEN41_ANTIFLICKER_MODULE_H
#define METAVISION_HAL_GEN41_ANTIFLICKER_MODULE_H

#include <string>
#include <map>

#include "metavision/hal/facilities/i_antiflicker_module.h"

namespace Metavision {

class RegisterMap;

class Gen41AntiFlickerModule : public I_AntiFlickerModule {
public:
    Gen41AntiFlickerModule(const std::shared_ptr<RegisterMap> &regmap, const std::string &sensor_prefix);

    virtual void enable() override;
    virtual void disable() override;

    virtual void set_frequency(uint32_t frequency_center, uint32_t bandwidth, bool stop = true) override;
    virtual void set_frequency_band(uint32_t min_freq, uint32_t max_freq, bool stop = true) override;

private:
    std::shared_ptr<RegisterMap> register_map_;
    std::string sensor_prefix_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN41_ANTIFLICKER_MODULE_H
