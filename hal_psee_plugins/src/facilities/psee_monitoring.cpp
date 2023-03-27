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

#include "metavision/psee_hw_layer/facilities/psee_monitoring.h"
#include "metavision/hal/facilities/i_hw_register.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"

namespace Metavision {

PseeMonitoring::PseeMonitoring(const std::shared_ptr<I_HW_Register> &hw_register) : i_hw_register_(hw_register) {
    if (!hw_register) {
        throw(HalException(PseeHalPluginErrorCode::HWRegisterNotFound, "HW Register facility not set."));
    }
}

const std::shared_ptr<I_HW_Register> &PseeMonitoring::get_hw_register() const {
    return i_hw_register_;
}

} // namespace Metavision
