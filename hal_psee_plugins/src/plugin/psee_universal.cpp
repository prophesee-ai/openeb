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
#include "devices/gen3/gen3_fx3_facilities_builder.h"
#include "devices/gen31/gen31_fx3_facilities_builder.h"
#ifdef HAL_GEN4_SUPPORT
#include "devices/gen4/gen4_evk1_facilities_builder.h"
#endif
#include "devices/golden_fallbacks/golden_fallback_treuzell_facilities_builder.h"
#include "devices/golden_fallbacks/golden_fallback_fx3_facilities_builder.h"
#include "devices/treuzell/tz_streamer.h"
#include "devices/gen31/gen31_ccam5_tz_device.h"
#include "devices/gen41/gen41_tz_device.h"
#include "devices/imx636/imx636_tz_device.h"
#include "devices/treuzell/tz_psee_video.h"
#include "devices/treuzell/ti_tmp103.h"
#include "boards/fx3/fx3_camera_discovery.h"
#include "boards/treuzell/tz_camera_discovery.h"
#include "boards/treuzell/tz_libusb_board_command.h"
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
    TzLibUSBBoardCommand::add_usb_id(0x04b4, 0x00f4, 0x19);
    TzLibUSBBoardCommand::add_usb_id(0x04b4, 0x00f5, 0x19);
    // Register live camera discoveries
    auto &fx3_disc = plugin.add_camera_discovery(std::make_unique<Fx3CameraDiscovery>());
    fx3_disc.register_device_builder(SYSTEM_CCAM3_GEN3, build_gen3_fx3_device);
    fx3_disc.register_device_builder(SYSTEM_CCAM3_GEN31, build_gen31_fx3_device);
#ifdef HAL_GEN4_SUPPORT
    fx3_disc.register_device_builder(SYSTEM_CCAM3_GEN4, build_gen4_evk1_device);
#endif
    fx3_disc.register_device_builder(SYSTEM_CCAM3_GOLDEN_FALLBACK, build_golden_fallback_fx3_device);
    auto &tz_disc = plugin.add_camera_discovery(std::make_unique<TzCameraDiscovery>());
    tz_disc.factory().insert("treuzell,streamer", TzStreamer::build);
    tz_disc.factory().insert("psee,video", TzPseeVideo::build);
    tz_disc.factory().insert("psee,ccam5_fpga", TzCcam5Gen31::build);
    tz_disc.factory().insert("psee,ccam5_gen41", TzGen41::build);
    tz_disc.factory().insert("psee,ccam5_gen42", TzImx636::build);
    tz_disc.factory().insert("ti,tmp103", TiTmp103::build);
#endif

    auto &file_disc = plugin.add_file_discovery(std::make_unique<PseeFileDiscovery>());
}
