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

#include "devices/treuzell/tz_issd_device.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include "devices/common/issd.h"
#include "metavision/psee_hw_layer/utils/register_map.h"
#include "metavision/hal/utils/hal_log.h"
#include <thread>

namespace Metavision {

TzIssdDevice::TzIssdDevice(const Issd &issd) : issd(issd) {
    stop();
    destroy();
    initialize();
}

TzIssdDevice::~TzIssdDevice() {
    try {
        destroy();
    } catch (...) { MV_HAL_LOG_WARNING() << "Treuzell ISSD Device destruction failed!"; }
}

void TzIssdDevice::initialize() {
    ApplyRegisterOperationSequence(issd.init);
}

void TzIssdDevice::destroy() {
    ApplyRegisterOperationSequence(issd.destroy);
}

void TzIssdDevice::start() {
    ApplyRegisterOperationSequence(issd.start);
}

void TzIssdDevice::stop() {
    ApplyRegisterOperationSequence(issd.stop);
}

void TzIssdDevice::ApplyRegisterOperation(const RegisterOperation operation) {
    switch (operation.action) {
    case RegisterAction::READ: {
        uint32_t value = (*register_map).read(operation.address);
    } break;
    case RegisterAction::WRITE: {
        (*register_map).write(operation.address, operation.data);
    } break;
    case RegisterAction::WRITE_FIELD: {
        uint32_t saved_fields = (*register_map).read(operation.address) & (~operation.mask);
        uint32_t write_field  = operation.data & operation.mask;
        uint32_t write_reg    = saved_fields | write_field;
        (*register_map).write(operation.address, write_reg);
    } break;
    case RegisterAction::DELAY: {
        std::this_thread::sleep_for(std::chrono::milliseconds(operation.usec / 1000));
    } break;
    default:
        // FIXME raise error.
        break;
    };
}

void TzIssdDevice::ApplyRegisterOperationSequence(const std::vector<RegisterOperation> sequence) {
    for (auto operation : sequence) {
        ApplyRegisterOperation(operation);
    }
}

} // namespace Metavision
