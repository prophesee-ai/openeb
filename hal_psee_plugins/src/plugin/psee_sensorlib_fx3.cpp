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

#if !defined(__ANDROID__) || defined(ANDROID_USES_LIBUSB)
#include "boards/fx3/fx3_camera_discovery.h"
#endif
#include "devices/sensorlib/sensorlib_fx3_facilities_builder.h"
#include "boards/rawfile/psee_file_discovery.h"
#include "devices/utils/device_system_id.h"
#include "metavision/hal/plugin/plugin.h"
#include "metavision/hal/plugin/plugin_entrypoint.h"
#include "metavision/hal/utils/hal_software_info.h"
#include "plugin/psee_plugin.h"

void initialize_plugin(void *plugin_ptr) {
    using namespace Metavision;

    Plugin &plugin = plugin_cast(plugin_ptr);
    initialize_psee_plugin(plugin);

#if !defined(__ANDROID__) || defined(ANDROID_USES_LIBUSB)
    // Register live camera discoveries
    auto &fx3_disc = plugin.add_camera_discovery(std::make_unique<Fx3CameraDiscovery>());
    fx3_disc.device_builder_factory().insert(SYSTEM_FX3_UNKNOWN, build_sensorlib_fx3_device);
#endif

    auto &file_disc = plugin.add_file_discovery(std::make_unique<PseeFileDiscovery>());
}
