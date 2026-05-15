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

#include "metavision/psee_hw_layer/facilities/psee_trigger_out.h"
#include "metavision/psee_hw_layer/facilities/psee_device_control.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"

namespace Metavision {

PseeTriggerOut::PseeTriggerOut(const std::shared_ptr<PseeDeviceControl> &device_control) :
    device_control_(device_control) {
    if (!device_control_) {
        throw(HalException(PseeHalPluginErrorCode::DeviceControlNotFound, "Device control facility is null."));
    }
}

void PseeTriggerOut::setup() {
    get_device_control()->set_trigger_out(std::static_pointer_cast<PseeTriggerOut>(shared_from_this()));
}

const std::shared_ptr<PseeDeviceControl> &PseeTriggerOut::get_device_control() const {
    return device_control_;
}

void PseeTriggerOut::teardown() {
    disable();
}

} // namespace Metavision
