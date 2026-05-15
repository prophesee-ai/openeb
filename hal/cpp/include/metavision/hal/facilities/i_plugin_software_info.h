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

#ifndef METAVISION_HAL_I_PLUGIN_SOFTWARE_INFO_H
#define METAVISION_HAL_I_PLUGIN_SOFTWARE_INFO_H

#include <string>

#include "metavision/sdk/base/utils/software_info.h"
#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Provides information about the Plugin software
class I_PluginSoftwareInfo : public I_RegistrableFacility<I_PluginSoftwareInfo> {
public:
    /// @brief Constructor
    /// @param plugin_integrator_name Name of the plugin integrator
    /// @param plugin_name Name of the plugin
    /// @param software_info Information about the HAL software version
    I_PluginSoftwareInfo(const std::string &plugin_integrator_name, const std::string &plugin_name,
                         const Metavision::SoftwareInfo &software_info);

    /// @brief Gets plugin integrator name
    /// @return The name of the plugin integrator
    const std::string &get_plugin_integrator_name() const;

    /// @brief Gets plugin name
    /// @return The plugin name
    const std::string &get_plugin_name() const;

    /// @brief Gets HAL's software information
    /// @return HAL's software information
    const Metavision::SoftwareInfo &get_software_info() const;

private:
    std::string plugin_name_;
    std::string plugin_integrator_name_;
    Metavision::SoftwareInfo pimpl_;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_PLUGIN_SOFTWARE_INFO_H
