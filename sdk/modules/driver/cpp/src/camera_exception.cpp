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

#include <string>
#include <system_error>

#include "metavision/sdk/driver/camera_error_code.h"
#include "metavision/sdk/driver/internal/camera_error_code_internal.h"
#include "metavision/sdk/driver/camera_exception.h"
#include "metavision/sdk/base/utils/error_category.h"
#include "metavision/sdk/base/utils/error_code.h"

namespace Metavision {

namespace { // anonymous namespace

std::string get_error_message(CameraErrorCodeType error_code, const std::string &additional_info = "") {
    std::string msg_to_ret;

    switch (error_code) {
    // ---------------------
    // public errors
    case CameraErrorCode::FailedInitialization:
        msg_to_ret = "Error while initializing the camera.";
        break;

    case CameraErrorCode::CameraNotFound:
        msg_to_ret = "Camera not found. Check that a camera is plugged on your system and retry.";
        break;

    case CameraErrorCode::RuntimeError:
        msg_to_ret = "Camera runtime error.";
        break;

    case CameraErrorCode::BiasesError:
        msg_to_ret = "Could not set given biases.";
        break;

    case BiasesErrors::UnsupportedBias:
        msg_to_ret = "Could not set bias.";
        break;

    case BiasesErrors::UnsupportedBiasFile:
        msg_to_ret = "Could not set biases from file.";
        break;

    case CameraErrorCode::RoiError:
        msg_to_ret = "Could not set given ROI on the sensor.";
        break;

    case CameraErrorCode::FirmwareIsNotUpToDate:
        msg_to_ret = "Device's firmware do not support the requested feature. Update it to solve this issue.";
        break;

    case CameraErrorCode::CameraNotInitialized:
        msg_to_ret = "Camera is not yet initialized.";
        break;

    case CameraErrorCode::InvalidRawfile:
        msg_to_ret = "RAW file can not be used as input source.";
        break;

    case CameraErrorCode::DataTransferFailed:
        msg_to_ret = "An error occurred while retrieving data from input source.";
        break;

    case CameraErrorCode::InvalidArgument:
        msg_to_ret = "Invalid camera argument provided.";
        break;

    case CameraErrorCode::FileDoesNotExist:
        msg_to_ret = "No such file or directory.";
        break;

    case CameraErrorCode::NotARegularFile:
        msg_to_ret = "File is not a regular file.";
        break;

    case CameraErrorCode::WrongExtension:
        msg_to_ret =
            "File extension doesn't match the one expected. Verify that you are using the correct file for the "
            "correct use.";
        break;

    case CameraErrorCode::CouldNotOpenFile:
        msg_to_ret = "Could not open file.";
        break;

        // ---------------------
        // internal errors

    case CameraErrorCode::InternalInitializationError:
    case InternalInitializationErrors::ILLBiasesNotFound:
    case InternalInitializationErrors::IEventsStreamNotFound:
    case InternalInitializationErrors::IDeviceControlNotFound:
    case InternalInitializationErrors::IDecoderNotFound:
    case InternalInitializationErrors::ICDDecoderNotFound:
    case InternalInitializationErrors::IGeometryNotFound:
    case InternalInitializationErrors::IBoardIdentificationNotFound:
    case InternalInitializationErrors::IRoiNotFound:
    case InternalInitializationErrors::UnknownSystemId:
    case InternalInitializationErrors::InvalidFPGAState:
        msg_to_ret = "Error while initializing the camera.";
        break;

    case CameraErrorCode::UnsupportedFeature:
    case UnsupportedFeatureErrors::RoiUnavailable:
    case UnsupportedFeatureErrors::BiasesUnavailable:
    case UnsupportedFeatureErrors::TriggerOutUnavailable:
    case UnsupportedFeatureErrors::ExtTriggerUnavailable:
    case UnsupportedFeatureErrors::AntiFlickerModuleUnavailable:
    case UnsupportedFeatureErrors::ErcModuleUnavailable:
    case UnsupportedFeatureErrors::EventTrailFilterModuleUnavailable:
        msg_to_ret = "Unsupported feature of the camera.";
        break;

    case CameraErrorCode::DeprecatedFeature:
        msg_to_ret = "Feature is deprecated. It will be removed in next releases.";
        break;

    default:
        msg_to_ret = "(unrecognized error)";
        break;
    }

    msg_to_ret = (additional_info.empty() ? "" : (additional_info + "\n")) + msg_to_ret;
    return msg_to_ret;
}
} // anonymous namespace

std::error_code make_error_code(int e) {
    return {e, ErrorCategory(e)};
}

CameraException::CameraException(CameraErrorCodeType e, std::string additional_info) :
    std::system_error(get_public_error_code(e),
                      ErrorCategory(e, "Metavision SDK Driver exception", get_error_message(e, additional_info))) {}

CameraException::CameraException(CameraErrorCodeType e) : CameraException(e, "") {}
} // namespace Metavision
