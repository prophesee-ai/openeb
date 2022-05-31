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

#include "facilities/tz_monitoring.h"

namespace Metavision {

TzMonitoring::TzMonitoring(const std::shared_ptr<TemperatureProvider> &temp,
                           const std::shared_ptr<IlluminationProvider> &illu) :
    temp_(temp), illu_(illu) {}

int TzMonitoring::get_temperature() {
    if (temp_)
        return temp_->get_temperature();
    return -274;
}

int TzMonitoring::get_illumination() {
    if (illu_)
        return illu_->get_illumination();
    return 0;
}

} // namespace Metavision
