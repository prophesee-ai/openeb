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

#include "devices/treuzell/ti_tmp103.h"
#include "devices/treuzell/tz_device_builder.h"
#include "metavision/psee_hw_layer/boards/treuzell/board_command.h"

namespace Metavision {

TiTmp103::TiTmp103(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id, std::shared_ptr<TzDevice> parent) :
    TzDevice(cmd, dev_id, parent) {}

std::shared_ptr<TzDevice> TiTmp103::build(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id,
                                          std::shared_ptr<TzDevice> parent) {
    return std::make_shared<TiTmp103>(cmd, dev_id, parent);
}
static TzRegisterBuildMethod method("ti,tmp103", TiTmp103::build);

void TiTmp103::spawn_facilities(DeviceBuilder &device_builder, const DeviceConfig &device_config) {}

TiTmp103::~TiTmp103() {}

void TiTmp103::start() {}

void TiTmp103::stop() {}

int TiTmp103::get_temperature() {
    return cmd->read_device_register(tzID, 0)[0];
}

} // namespace Metavision
