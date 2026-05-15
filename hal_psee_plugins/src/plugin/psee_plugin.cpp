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

#include "metavision/hal/plugin/plugin.h"
#include "metavision/hal/utils/hal_software_info.h"
#include "metavision/sdk/base/utils/software_info.h"
#include "plugin/psee_plugin.h"

namespace Metavision {

const std::string &get_psee_plugin_integrator_name() {
    static const std::string integrator("Prophesee");
    return integrator;
}

void initialize_psee_plugin(Plugin &plugin, std::string integrator_name) {
    plugin.set_integrator_name(integrator_name);
    plugin.set_plugin_info(get_build_software_info());
    plugin.set_hal_info(get_hal_software_info());
}

void initialize_psee_plugin(Plugin &plugin) {
    plugin.set_integrator_name(get_psee_plugin_integrator_name());
    plugin.set_plugin_info(get_build_software_info());
    plugin.set_hal_info(get_hal_software_info());
}

} // namespace Metavision