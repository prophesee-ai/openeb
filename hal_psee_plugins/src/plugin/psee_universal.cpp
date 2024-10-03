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
#include <functional>

#include "metavision/hal/plugin/plugin.h"
#include "metavision/hal/plugin/plugin_entrypoint.h"
#include "plugin/psee_plugin.h"

using namespace Metavision;

struct PluginDiscovery {
    static std::vector<std::function<void(Plugin &)>> plugin_discovery;
    PluginDiscovery(std::function<void(Plugin &)> f) {
        plugin_discovery.push_back(f);
    }

    static void discover(Plugin &plugin) {
        for (const auto &f : plugin_discovery) {
            f(plugin);
        }
    }
};
std::vector<std::function<void(Plugin &)>> PluginDiscovery::plugin_discovery;

#if !defined(__ANDROID__) || defined(ANDROID_USES_LIBUSB)
#include "boards/fx3/fx3_camera_discovery.h"
#include "boards/treuzell/tz_camera_discovery.h"
PluginDiscovery register_treuzell([](Plugin &plugin) {
    auto tz_cam_discovery = std::make_unique<TzCameraDiscovery>();
    // Register the known USB vendor ID, with the subclass used for Treuzell
    tz_cam_discovery->add_usb_id(0x03fd, 0x5832, 0x19);
    tz_cam_discovery->add_usb_id(0x03fd, 0x5832, 0x0);
    tz_cam_discovery->add_usb_id(0x04b4, 0x00f4, 0x19);
    tz_cam_discovery->add_usb_id(0x04b4, 0x00f5, 0x19);
    tz_cam_discovery->add_usb_id(0x1FC9, 0x5838, 0x19);

    // Register live camera discoveries
    auto &fx3_disc = plugin.add_camera_discovery(std::make_unique<Fx3CameraDiscovery>());
    auto &tz_disc  = plugin.add_camera_discovery(std::move(tz_cam_discovery));
});
#endif // !defined(__ANDROID__) || defined(ANDROID_USES_LIBUSB)

#if !defined(__ANDROID__) && defined(HAS_V4L2)
#include "boards/v4l2/v4l2_camera_discovery.h"
PluginDiscovery register_v4l2([](Plugin &plugin) {
    auto &v4l2_disc = plugin.add_camera_discovery(std::make_unique<V4l2CameraDiscovery>());
});
#endif // HAS_V4L2

#include "boards/rawfile/psee_file_discovery.h"
PluginDiscovery register_psee_file([](Plugin &plugin) {
    auto &file_disc = plugin.add_file_discovery(std::make_unique<PseeFileDiscovery>());
});

void initialize_plugin(void *plugin_ptr) {
    Plugin &plugin = plugin_cast(plugin_ptr);
    initialize_psee_plugin(plugin);
    PluginDiscovery::discover(plugin);
}
