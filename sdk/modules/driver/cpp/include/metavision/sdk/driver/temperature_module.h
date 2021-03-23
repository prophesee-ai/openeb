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

#ifndef METAVISION_SDK_DRIVER_TEMPERATURE_MODULE_H
#define METAVISION_SDK_DRIVER_TEMPERATURE_MODULE_H

namespace Metavision {

/// @note This class is deprecated since version 2.1.0 and will be removed in next releases
/// @brief Facility class to handle the Temperature module
/// Allows enabling the temperature sensor in order to receive @ref EventTemperature events
class TemperatureModule {
public:
    /// @brief Destructor
    ~TemperatureModule();

    /// @brief Enables temperature monitoring
    void enable();

    /// @brief Disables temperature monitoring
    void disable();

    /// @brief Returns the temperature
    float get_temperature(int source);
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_TEMPERATURE_MODULE_H
