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

#include "devices/treuzell/tz_streamer.h"
#include "boards/treuzell/tz_libusb_board_command.h"
#include "boards/treuzell/tz_control_frame.h"
#include "boards/treuzell/treuzell_command_definition.h"

namespace Metavision {

TzStreamer::TzStreamer(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id, std::shared_ptr<TzDevice> parent) :
    TzDevice(cmd, dev_id, parent) {
    // Try to stop any previous activity. Ignore failure, as it is the expected behavior from a stopped device
    try {
        stop();
    } catch (std::system_error &e) {}
}

std::shared_ptr<TzDevice> TzStreamer::build(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id,
                                            std::shared_ptr<TzDevice> parent) {
    return std::make_shared<TzStreamer>(cmd, dev_id, parent);
}

void TzStreamer::spawn_facilities(DeviceBuilder &device_builder) {}

TzStreamer::~TzStreamer() {}

void TzStreamer::start() {
    TzGenericCtrlFrame streamon(TZ_PROP_DEVICE_STREAM | TZ_WRITE_FLAG);
    streamon.push_back32(tzID);
    streamon.push_back32(1);
    cmd->transfer_tz_frame(streamon);
}

void TzStreamer::stop() {
    TzGenericCtrlFrame streamoff(TZ_PROP_DEVICE_STREAM | TZ_WRITE_FLAG);
    streamoff.push_back32(tzID);
    streamoff.push_back32(0);
    cmd->transfer_tz_frame(streamoff);
}

TzDevice::StreamFormat TzStreamer::get_output_format() {
    auto input = child.lock();
    if (input)
        return input->get_output_format();
    else
        return {std::string("NONE"), nullptr};
}

} // namespace Metavision
