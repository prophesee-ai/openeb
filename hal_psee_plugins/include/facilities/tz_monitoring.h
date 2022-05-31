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

#ifndef METAVISION_HAL_TZ_MONITORING_H
#define METAVISION_HAL_TZ_MONITORING_H

#include <memory>

#include "metavision/hal/facilities/i_monitoring.h"

namespace Metavision {

class TemperatureProvider {
public:
    /// @brief Gets temperature
    /// @return Sensor's temperature (in C)
    virtual int get_temperature() = 0;
};

class IlluminationProvider {
public:
    /// @brief Gets illumination
    /// @return Sensor's illumination (in lux)
    virtual int get_illumination() = 0;
};

class TzMonitoring : public I_Monitoring {
public:
    TzMonitoring(const std::shared_ptr<TemperatureProvider> &temp, const std::shared_ptr<IlluminationProvider> &illu);

    /// @brief Gets temperature
    /// @return Sensor's temperature (in C)
    virtual int get_temperature();

    /// @brief Gets illumination
    /// @return Sensor's illumination (in lux)
    virtual int get_illumination();

private:
    std::shared_ptr<TemperatureProvider> temp_;
    std::shared_ptr<IlluminationProvider> illu_;
};

} // namespace Metavision

#endif // METAVISION_HAL_TZ_MONITORING_H
