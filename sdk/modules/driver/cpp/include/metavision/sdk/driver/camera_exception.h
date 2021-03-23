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

#ifndef METAVISION_SDK_DRIVER_CAMERA_EXCEPTION_H
#define METAVISION_SDK_DRIVER_CAMERA_EXCEPTION_H

#include <string>
#include <system_error>

#include "metavision/sdk/driver/camera_error_code.h"

namespace Metavision {

/// @brief Class for all exceptions thrown by Metavision SDK Driver
///
/// @sa http://www.cplusplus.com/reference/system_error/system_error/
/// @sa http://en.cppreference.com/w/cpp/error/error_code
class CameraException : public std::system_error {
public:
    /// @brief Creates an exception of type e with default error message
    /// @param e Camera error code
    /// @sa @ref CameraErrorCode
    CameraException(CameraErrorCodeType e);

    /// @brief Creates an exception of type e with an error description in additional_info
    /// @param e Camera error code
    /// @param additional_info error description
    /// @sa @ref CameraErrorCode
    CameraException(CameraErrorCodeType e, std::string additional_info);
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_CAMERA_EXCEPTION_H
