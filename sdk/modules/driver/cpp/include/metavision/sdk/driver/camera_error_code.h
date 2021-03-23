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

#ifndef METAVISION_SDK_DRIVER_CAMERA_ERROR_CODE_H
#define METAVISION_SDK_DRIVER_CAMERA_ERROR_CODE_H

namespace Metavision {

/// @brief Alias type used for CameraErrorCode enums
/// @sa @ref CameraErrorCode
using CameraErrorCodeType = int;

/// @brief Enum that holds camera error codes for Camera Exception
/// @sa @ref CameraException
namespace CameraErrorCode {
enum Enum : CameraErrorCodeType {
    /// Base Metavision SDK Driver camera error
    CameraError = 0x100000,

    /// Camera failed initialization
    FailedInitialization = CameraError | 0x01000,
    /// Failed initialization due to a camera not found
    CameraNotFound = FailedInitialization | 0x1,
    /// Metavision SDK Driver Internal Initialization problem
    InternalInitializationError = FailedInitialization | 0x100,

    /// Camera runtime errors
    RuntimeError = CameraError | 0x02000,
    /// Fails to set biases.
    BiasesError = RuntimeError | 0x200,
    /// Error due to a non initialized camera
    CameraNotInitialized = RuntimeError | 0x2,
    /// Invalid RAW file
    InvalidRawfile = RuntimeError | 0x3,
    /// Error while retrieving data from source
    DataTransferFailed = RuntimeError | 0x4,
    /// Fails to set ROI.
    RoiError = RuntimeError | 0x6,
    /// Error while trying to use a feature unsupported by the camera
    UnsupportedFeature = RuntimeError | 0x100,
    /// Device firmware is not up to date
    FirmwareIsNotUpToDate = RuntimeError | 0x07,
    /// Error while trying to use a deprecated feature
    DeprecatedFeature = RuntimeError | 0x08,

    /// Errors related to invalid arguments
    InvalidArgument = CameraError | 0x03000,
    /// File does not exist
    FileDoesNotExist = InvalidArgument | 0x1,
    /// File is not a regular file
    NotARegularFile = InvalidArgument | 0x2,
    /// File extension is not the one expected
    WrongExtension = InvalidArgument | 0x3,
    /// Could not open file
    CouldNotOpenFile = InvalidArgument | 0x4,
};
}

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_CAMERA_ERROR_CODE_H
