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
#include "devices/golden_fallbacks/golden_fallback_treuzell_facilities_builder.h"
#include "devices/golden_fallbacks/golden_fallback_fx3_facilities_builder.h"
#include "boards/treuzell/tz_camera_discovery.h"
#include "boards/fx3/fx3_camera_discovery.h"
#endif
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
    // Fx3 golden fallback
    auto &fx3_disc = plugin.add_camera_discovery(std::make_unique<Fx3CameraDiscovery>());
    fx3_disc.register_device_builder(SYSTEM_CCAM3_GOLDEN_FALLBACK, build_golden_fallback_fx3_device);

    // Treuzell golden fallback (a.k.a evk2 & evk3)
    auto &evk2_disc = plugin.add_camera_discovery(std::make_unique<TzCameraDiscovery>());
    evk2_disc.register_device_builder(SYSTEM_CCAM5_GOLDEN_FALLBACK, build_golden_fallback_treuzell_device);
#endif
}
