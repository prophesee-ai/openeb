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

#include "metavision/hal/facilities/i_plugin_software_info.h"

namespace Metavision {

I_PluginSoftwareInfo::I_PluginSoftwareInfo(const std::string &plugin_integrator_name, const std::string &plugin_name,
                                           const Metavision::SoftwareInfo &software_info) :
    plugin_integrator_name_(plugin_integrator_name), plugin_name_(plugin_name), pimpl_(software_info) {}

const std::string &I_PluginSoftwareInfo::get_plugin_integrator_name() const {
    return plugin_integrator_name_;
}

const std::string &I_PluginSoftwareInfo::get_plugin_name() const {
    return plugin_name_;
}

const Metavision::SoftwareInfo &I_PluginSoftwareInfo::get_software_info() const {
    return pimpl_;
}

} // namespace Metavision
