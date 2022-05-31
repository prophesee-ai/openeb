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

#ifndef METAVISION_HAL_GEN31_SENSOR_H
#define METAVISION_HAL_GEN31_SENSOR_H

#include <string>
#include <memory>

#include "devices/gen31/gen31_analog.h"
#include "devices/gen31/gen31_digital.h"

namespace Metavision {

class RegisterMap;

class Gen31Sensor {
public:
    Gen31Sensor(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix, bool is_em);

    long long get_chip_id();
    void init();
    void start();
    void stop();
    void destroy();
    void bgen_init();

private:
    std::shared_ptr<RegisterMap> register_map_;
    Gen31Analog analog_;
    Gen31Digital digital_;
    const std::string prefix_;
    bool is_em_ = false;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN31_SENSOR_H
