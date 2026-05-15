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

#include "hal_python_binder.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "pb_doc_hal.h"

namespace Metavision {

static DeviceFacilityGetter<I_PluginSoftwareInfo> getter("get_i_plugin_software_info");

static HALFacilityPythonBinder<I_PluginSoftwareInfo> bind(
    [](auto &module, auto &class_binding) {
        class_binding
            .def("get_plugin_integrator_name", &I_PluginSoftwareInfo::get_plugin_integrator_name,
                 pybind_doc_hal["Metavision::I_PluginSoftwareInfo::get_plugin_integrator_name"])
            .def("get_plugin_name", &I_PluginSoftwareInfo::get_plugin_name,
                 pybind_doc_hal["Metavision::I_PluginSoftwareInfo::get_plugin_name"])
            .def("get_software_info", &I_PluginSoftwareInfo::get_software_info,
                 pybind_doc_hal["Metavision::I_PluginSoftwareInfo::get_software_info"]);
    },
    "I_PluginSoftwareInfo", pybind_doc_hal["Metavision::I_PluginSoftwareInfo"]);

} // namespace Metavision
