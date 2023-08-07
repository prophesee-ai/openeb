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
#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"
#include "devices/treuzell/tz_device_builder.h"

namespace Metavision {

static std::pair<TzDeviceBuilder::Build_Fun, TzDeviceBuilder::Check_Fun> get_method(uint32_t chip_id) {
    /* Some of the following IDs are prototypes that will never get to mass production. IDs are left under-documented on
     * purpose */
    switch (chip_id) {
    case 0xA0301002:
    case 0xA0301003:
    case 0xA0301004:
    case 0xA0301005:
        return TzRegisterBuildMethod::recall("psee,gen41");
    case 0xA0401806:
        return TzRegisterBuildMethod::recall("psee,gen42");
    case 0x30501C01:
        return TzRegisterBuildMethod::recall("psee,saphir");
    default:
        break;
    }
    return {nullptr, nullptr};
}

static std::shared_ptr<TzDevice> build(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id,
                                       std::shared_ptr<TzDevice> parent) {
    /* If the board generically reports an I2C/SPI sensor, we may try to identify the sensor
     * through direct register access and build it if we know how */
    auto method = get_method(cmd->read_device_register(dev_id, 0x14)[0]);
    if (!method.first) {
        return nullptr;
    }
    if (method.second && !method.second(cmd, dev_id)) {
        return nullptr;
    }
    return method.first(cmd, dev_id, parent);
}

/* We are overly optimistic, and include no check method, we may fail at build time, but those keys are usually used on
 * systems where the previous device needs to be initialized to allow register access on this one; this is not a
 * suitable use case for the check method */
static TzRegisterBuildMethod method("psee,i2c-sensor", build);
static TzRegisterBuildMethod method1("psee,spi-sensor", build);

} // namespace Metavision
