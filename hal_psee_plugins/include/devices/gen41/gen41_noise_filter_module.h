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

#ifndef METAVISION_HAL_GEN41_NOISE_FILTER_MODULE_H
#define METAVISION_HAL_GEN41_NOISE_FILTER_MODULE_H

#include <string>
#include <map>

#include "metavision/hal/facilities/i_noise_filter_module.h"

namespace Metavision {

class RegisterMap;

class Gen41NoiseFilterModule : public I_NoiseFilterModule {
public:
    Gen41NoiseFilterModule(const std::shared_ptr<RegisterMap> &regmap, const std::string &sensor_prefix);

    virtual void enable(I_NoiseFilterModule::Type type, uint32_t threshold) override;
    virtual void disable() override;

private:
    std::shared_ptr<RegisterMap> register_map_;
    std::string sensor_prefix_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN41_NOISE_FILTER_MODULE_H
