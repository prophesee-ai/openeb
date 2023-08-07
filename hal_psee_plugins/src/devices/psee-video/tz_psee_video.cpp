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

#include "metavision/psee_hw_layer/devices/psee-video/tz_psee_video.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_control_frame.h"
#include "boards/treuzell/treuzell_command_definition.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

TzPseeVideo::TzPseeVideo(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id, std::shared_ptr<TzDevice> parent) :
    TzDevice(cmd, dev_id, parent), TzPseeFpgaDevice() {
    try {
        destroy();
    } catch (const std::system_error &e) {}
    try {
        initialize();
    } catch (const std::system_error &e) { MV_HAL_LOG_TRACE() << name << "did not enable:" << e.what(); }
}

TzPseeVideo::~TzPseeVideo() {
    try {
        destroy();
    } catch (const std::system_error &e) {}
}

void TzPseeVideo::spawn_facilities(DeviceBuilder &device_builder, const DeviceConfig &device_config) {}

std::list<StreamFormat> TzPseeVideo::get_supported_formats() const {
    std::list<StreamFormat> formats;
    // Same as tzDevice, but expect Metis 1.7.0 if not implemented
    TzDeviceStringsCtrlFrame format(TZ_PROP_DEVICE_OUTPUT_FORMAT, tzID);
    try {
        cmd->transfer_tz_frame(format);
        formats.push_back(StreamFormat(format.get_strings()[0]));
    } catch (const std::system_error &e) { formats.push_back(StreamFormat("EVT3;height=720;width=1280")); }

    return formats;
}

StreamFormat TzPseeVideo::get_output_format() const {
    // Same as tzDevice, but expect Metis 1.7.0 if not implemented
    TzDeviceStringsCtrlFrame format(TZ_PROP_DEVICE_OUTPUT_FORMAT, tzID);
    try {
        cmd->transfer_tz_frame(format);
        return StreamFormat(format.get_strings()[0]);
    } catch (const std::system_error &e) {
        MV_HAL_LOG_TRACE() << name << "defaulting to EVT3:" << e.what();
        StreamFormat fmt("EVT3");
        fmt["width"]  = "1280";
        fmt["height"] = "720";
        return fmt;
    }
}

long TzPseeVideo::get_system_id() const {
    return TzPseeFpgaDevice::get_system_id();
}

bool TzPseeVideo::set_mode_standalone() {
    return true;
}

bool TzPseeVideo::set_mode_master() {
    return false;
}

bool TzPseeVideo::set_mode_slave() {
    return false;
}

I_CameraSynchronization::SyncMode TzPseeVideo::get_mode() {
    return I_CameraSynchronization::SyncMode::STANDALONE;
}

} // namespace Metavision
