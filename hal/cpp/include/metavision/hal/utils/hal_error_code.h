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

#ifndef METAVISION_HAL_HAL_ERROR_CODE_H
#define METAVISION_HAL_HAL_ERROR_CODE_H

namespace Metavision {

/// @brief Alias type used for @ref HalErrorCode enums
using HalErrorCodeType = int;

/// @brief Enum that holds error codes for HAL exceptions
namespace HalErrorCode {
enum Enum : HalErrorCodeType {
    /// Base Hal camera error
    CameraError = 0x100000,

    /// Camera failed initialization
    FailedInitialization = CameraError | 0x01000,
    /// Failed initialization due to a camera not found
    CameraNotFound = FailedInitialization | 0x1,
    /// Golden fallback
    GoldenFallbackBooted = FailedInitialization | 0x2,
    /// Metavision Internal Initialization problem
    InternalInitializationError = FailedInitialization | 0x100,

    /// Errors related to invalid arguments
    InvalidArgument = CameraError | 0x02000,
    /// Value given out of supported range
    ValueOutOfRange = InvalidArgument | 0x1,
    /// Requested value does not exist
    NonExistingValue = InvalidArgument | 0x2,
    /// Requested operation cannot be performed on given argument
    OperationNotPermitted = InvalidArgument | 0x3,
    /// Value for requested setting is not supported
    UnsupportedValue = InvalidArgument | 0x4,

    /// Errors related to calling deprecated function that have no equivalent in current API
    DeprecatedFunctionCalled = CameraError | 0x03000,

    /// Operation is not implemented
    OperationNotImplemented = CameraError | 0x04000,

    /// Operation reached maximum retries limit
    MaximumRetriesExeeded = CameraError | 0x05000,
};
}

} // namespace Metavision

#endif // METAVISION_HAL_HAL_ERROR_CODE_H
