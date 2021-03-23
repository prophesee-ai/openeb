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

#ifndef METAVISION_HAL_DEVICE_BUILDER_H
#define METAVISION_HAL_DEVICE_BUILDER_H

#include <memory>
#include <vector>

#include "metavision/hal/facilities/detail/facility_wrapper.h"

namespace Metavision {

/// @brief Base class used to pass parameters to builder functions
class DeviceBuilderParameters {};

class Device;
class I_HALSoftwareInfo;
class I_PluginSoftwareInfo;

/// @brief Builder class to handle Device creation
class DeviceBuilder {
public:
    /// @brief Constructor
    /// @param i_hal_sw_info Information on the HAL software version
    /// @param i_plugin_sw_info Information on the plugin software version
    DeviceBuilder(std::unique_ptr<I_HALSoftwareInfo> i_hal_sw_info,
                  std::unique_ptr<I_PluginSoftwareInfo> i_plugin_sw_info);

    /// @brief Move constructor
    DeviceBuilder(DeviceBuilder &&);

    /// @brief Move operator
    DeviceBuilder &operator=(DeviceBuilder &&);

    /// @brief Gets the information on the HAL software version
    /// @return Information on the HAL software version
    const std::shared_ptr<I_HALSoftwareInfo> &get_hal_software_info() const;

    /// @brief Gets the information on the plugin software version
    /// @return Information on the plugin software version
    const std::shared_ptr<I_PluginSoftwareInfo> &get_plugin_software_info() const;

    /// @brief Convenience function to add a facility that will be registered on the created device
    /// @tparam FacilityType Type of facility
    /// @param facility Facility to be registered to the device
    /// @return A shared pointer to the facility
    template<typename FacilityType>
    std::shared_ptr<FacilityType> add_facility(std::unique_ptr<FacilityType> facility) {
        auto ptr = std::shared_ptr<FacilityType>(std::move(facility));
        facilities_.push_back(std::make_unique<FacilityWrapper>(ptr));
        return ptr;
    }

    /// @brief Builds a device
    /// @return The created device
    std::unique_ptr<Device> operator()();

private:
    std::shared_ptr<I_HALSoftwareInfo> i_hal_sw_info_;
    std::shared_ptr<I_PluginSoftwareInfo> i_plugin_sw_info_;
    std::vector<std::unique_ptr<FacilityWrapper>> facilities_;
};

} // namespace Metavision

#endif // METAVISION_HAL_DEVICE_BUILDER_H
