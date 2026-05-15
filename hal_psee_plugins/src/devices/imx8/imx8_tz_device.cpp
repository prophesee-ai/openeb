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

#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include "devices/imx8/imx8_tz_device.h"
#include "devices/treuzell/tz_device_builder.h"

using namespace Metavision;

static TzRegisterBuildMethod method("fsl,imx8mq-csi", TzImx8Device::build);

TzImx8Device::TzImx8Device(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id,
                           std::shared_ptr<TzDevice> parent) :
    TzUnknownDevice(cmd, dev_id, parent) {}

// TzMainDevice
bool TzImx8Device::set_mode_standalone() {
    return true;
}

bool TzImx8Device::set_mode_master() {
    return false;
}

bool TzImx8Device::set_mode_slave() {
    return false;
}

I_CameraSynchronization::SyncMode TzImx8Device::get_mode() const {
    return I_CameraSynchronization::SyncMode::STANDALONE;
}

I_HW_Identification::SensorInfo TzImx8Device::get_sensor_info() {
    return I_HW_Identification::SensorInfo{};
}

std::shared_ptr<TzImx8Device> TzImx8Device::build(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t id,
                                                  std::shared_ptr<TzDevice> parent) {
    return std::make_shared<TzImx8Device>(cmd, id, parent);
}
