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

#include "metavision/hal/facilities/i_decoder.h"
#include "metavision/hal/utils/hal_exception.h"

namespace Metavision {

size_t I_Decoder::add_protocol_violation_callback(const ProtocolViolationCallback_t &cb) {
    throw HalException(HalErrorCode::OperationNotImplemented, "Decoder protocol violation detection not implemented");
}

bool I_Decoder::remove_protocol_violation_callback(size_t callback_id) {
    return false;
}

} // namespace Metavision
