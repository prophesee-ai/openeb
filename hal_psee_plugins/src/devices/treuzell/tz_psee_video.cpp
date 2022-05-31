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

#ifdef _MSC_VER
#define NOMINMAX
#endif

#include "devices/treuzell/tz_psee_video.h"
#include "boards/treuzell/tz_libusb_board_command.h"
#include "boards/treuzell/tz_control_frame.h"

namespace Metavision {

TzPseeVideo::TzPseeVideo(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id, std::shared_ptr<TzDevice> parent) :
    TzDevice(cmd, dev_id, parent), TzPseeFpgaDevice() {}

void TzPseeVideo::spawn_facilities(DeviceBuilder &device_builder) {}

TzDevice::StreamFormat TzPseeVideo::get_output_format() {
    return {std::string("NONE"), nullptr};
}

void TzPseeVideo::start() {
    throw std::system_error(TZ_NOT_IMPLEMENTED, TzError());
}

void TzPseeVideo::stop() {
    throw std::system_error(TZ_NOT_IMPLEMENTED, TzError());
}

long TzPseeVideo::get_system_id() const {
    return TzPseeFpgaDevice::get_system_id();
}

long TzPseeVideo::get_system_version() const {
    return TzPseeFpgaDevice::get_system_version();
}

bool TzPseeVideo::set_mode_standalone() {
    return false;
}

bool TzPseeVideo::set_mode_master() {
    return false;
}

bool TzPseeVideo::set_mode_slave() {
    return false;
}

I_DeviceControl::SyncMode TzPseeVideo::get_mode() {
    return I_DeviceControl::SyncMode::STANDALONE;
}

} // namespace Metavision
