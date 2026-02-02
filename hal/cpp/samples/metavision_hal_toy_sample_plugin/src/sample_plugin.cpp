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

#include <string>
#include <memory>
#include <metavision/hal/utils/hal_software_info.h>
#include <metavision/hal/plugin/plugin_entrypoint.h>

#include "sample_camera_discovery.h"
#include "sample_file_discovery.h"
#include "sample_hw_identification.h"

int SAMPLE_PLUGIN_VERSION_MAJOR           = 0;
int SAMPLE_PLUGIN_VERSION_MINOR           = 1;
int SAMPLE_PLUGIN_VERSION_PATCH           = 0;
std::string SAMPLE_PLUGIN_VERSION_SUFFIX  = "dev";
std::string SAMPLE_PLUGIN_VCS_BRANCH      = "hal-sample-plugin-vcs-branch";
std::string SAMPLE_PLUGIN_VCS_COMMIT      = "hal-sample-plugin-vcs-commit";
std::string SAMPLE_PLUGIN_VCS_COMMIT_DATE = "hal-sample-plugin-vcs-commit-date";

namespace {
Metavision::SoftwareInfo get_sample_plugin_software_info() {
    return Metavision::SoftwareInfo(SAMPLE_PLUGIN_VERSION_MAJOR, SAMPLE_PLUGIN_VERSION_MINOR,
                                    SAMPLE_PLUGIN_VERSION_PATCH, SAMPLE_PLUGIN_VERSION_SUFFIX, SAMPLE_PLUGIN_VCS_BRANCH,
                                    SAMPLE_PLUGIN_VCS_COMMIT, SAMPLE_PLUGIN_VCS_COMMIT_DATE);
}
} // namespace

// The plugin name is the name of the library loaded without lib and the extension
void initialize_plugin(void *plugin_ptr) {
    Metavision::Plugin &plugin = Metavision::plugin_cast(plugin_ptr);

    plugin.set_integrator_name(SampleHWIdentification::SAMPLE_INTEGRATOR);
    plugin.set_plugin_info(get_sample_plugin_software_info());
    plugin.set_hal_info(Metavision::get_hal_software_info());

    plugin.add_camera_discovery(std::make_unique<SampleCameraDiscovery>());
    plugin.add_file_discovery(std::make_unique<SampleFileDiscovery>());
}
