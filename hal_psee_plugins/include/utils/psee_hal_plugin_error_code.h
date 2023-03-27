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

#ifndef METAVISION_HAL_PSEE_HAL_PLUGIN_ERROR_CODE_H
#define METAVISION_HAL_PSEE_HAL_PLUGIN_ERROR_CODE_H

#include "metavision/hal/utils/hal_error_code.h"

namespace Metavision {

/// Failed Initialization Error specific code
namespace PseeHalPluginErrorCode {
enum : HalErrorCodeType {
    // Failed initialization errors
    BoardCommandNotFound  = HalErrorCode::InternalInitializationError | 0xA,
    UnknownSystemId       = HalErrorCode::InternalInitializationError | 0xB,
    InvalidFPGAState      = HalErrorCode::InternalInitializationError | 0xE,
    DeviceControlNotFound = HalErrorCode::InternalInitializationError | 0xF,
    TriggerInNotFound     = HalErrorCode::InternalInitializationError | 0x10,
    TriggerOutNotFound    = HalErrorCode::InternalInitializationError | 0x11,
    ErcNotFound           = HalErrorCode::InternalInitializationError | 0x12,
    HWRegisterNotFound    = HalErrorCode::InternalInitializationError | 0x13,
    STCModuleNotFound     = HalErrorCode::InternalInitializationError | 0x14,
    AFKModuleNotFound     = HalErrorCode::InternalInitializationError | 0x15,
    ERCModuleNotFound     = HalErrorCode::InternalInitializationError | 0x16,
    EventsStreamNotFound  = HalErrorCode::InternalInitializationError | 0x20,
    CCam5NotBooted        = HalErrorCode::InternalInitializationError | 0x21,
    SensorInitError       = HalErrorCode::InternalInitializationError | 0x22,
    UnknownFormat         = HalErrorCode::InternalInitializationError | 0x23,
    UnsupportedFirmware   = HalErrorCode::InternalInitializationError | 0x24,

    // Invalid argument errors
    InvalidAFKValue = HalErrorCode::InvalidArgument | 0xC0,
};
}

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_HAL_PLUGIN_ERROR_CODE_H
