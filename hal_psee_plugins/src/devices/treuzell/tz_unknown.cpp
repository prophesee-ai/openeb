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

#include "devices/treuzell/tz_unknown.h"
#include "devices/treuzell/tz_device_builder.h"
#include "metavision/psee_hw_layer/boards/treuzell/board_command.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

TzUnknownDevice::TzUnknownDevice(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id,
                                 std::shared_ptr<TzDevice> parent) :
    TzDevice(cmd, dev_id, parent) {
    try {
        initialize();
    } catch (const std::system_error &e) { MV_HAL_LOG_TRACE() << name << "did not enable:" << e.what(); }
}

TzUnknownDevice::~TzUnknownDevice() {
    try {
        destroy();
    } catch (const std::system_error &e) { MV_HAL_LOG_TRACE() << name << "did not disable:" << e.what(); }
}

std::shared_ptr<TzDevice> TzUnknownDevice::build(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id,
                                                 std::shared_ptr<TzDevice> parent) {
    return std::make_shared<TzUnknownDevice>(cmd, dev_id, parent);
}
static TzRegisterBuildMethod method("", TzUnknownDevice::build);

void TzUnknownDevice::spawn_facilities(DeviceBuilder &device_builder, const DeviceConfig &device_config) {}

} // namespace Metavision
