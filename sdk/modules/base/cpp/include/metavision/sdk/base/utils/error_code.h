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

#ifndef METAVISION_SDK_BASE_ERROR_CODE_H
#define METAVISION_SDK_BASE_ERROR_CODE_H

namespace Metavision {

/// Internal errors masks
/// An error code is an hexadecimal 32 bits mask composed as follow:
/// -    12 first bits for the APIs ID (0xFFF00000)
/// -    8 following bits for the error type (0x000FF000)
/// -    4 following bits for internal error type (0x00000F00)
/// -    8 following bits for the error number (0x000000FF)

/// Part of the mask reserved for the API id
static constexpr int API_ID_CODE_MASK = 0xFFF00000;
/// Part of the mask reserved to specify the error type
static constexpr int ERROR_TYPE_CODE_MASK = 0xFF000;
/// Part of the mask reserved to specify an internal error type
static constexpr int INTERNAL_ERROR_TYPE_CODE_MASK = 0xF00;
/// Part of the mask reserved for the error number
static constexpr int ERROR_NUMBER_MASK = 0xFF;

/// @brief Returns the public part of a Metavision SDK error code
int get_public_error_code(int error_code);

} // namespace Metavision

#include "detail/error_code_impl.h"

#endif // METAVISION_SDK_BASE_ERROR_CODE_H
