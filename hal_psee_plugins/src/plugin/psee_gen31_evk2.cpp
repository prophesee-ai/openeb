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
#include "boards/treuzell/tz_camera_discovery.h"
#include "boards/treuzell/tz_libusb_board_command.h"
#include "devices/treuzell/tz_streamer.h"
#include "devices/gen31/gen31_evk2_tz_device.h"
#endif
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
    // Register the known USB vendor ID, with the subclass used for Treuzell
    TzLibUSBBoardCommand::add_usb_id(0x03fd, 0x5832, 0x19);
    TzLibUSBBoardCommand::add_usb_id(0x03fd, 0x5832, 0x0);
    // Register live camera discoveries
    auto &evk2_disc = plugin.add_camera_discovery(std::make_unique<TzCameraDiscovery>());
    evk2_disc.factory().insert("treuzell,streamer", TzStreamer::build);
    evk2_disc.factory().insert("psee,video", TzEvk2Gen31::build, TzEvk2Gen31::can_build);
#endif

    auto &file_disc = plugin.add_file_discovery(std::make_unique<PseeFileDiscovery>());
}
