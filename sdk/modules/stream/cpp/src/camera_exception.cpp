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

#include "metavision/sdk/base/utils/error_code.h"
#include "metavision/sdk/stream/camera_error_code.h"
#include "metavision/sdk/stream/internal/camera_error_code_internal.h"
#include "metavision/sdk/stream/camera_exception.h"

namespace Metavision {

namespace { // anonymous namespace

std::string get_error_message(CameraErrorCodeType error_code) {
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
    case InternalInitializationErrors::IEventsStreamNotFound:
    case InternalInitializationErrors::IDeviceControlNotFound:
    case InternalInitializationErrors::IDecoderNotFound:
    case InternalInitializationErrors::ICDDecoderNotFound:
    case InternalInitializationErrors::IGeometryNotFound:
    case InternalInitializationErrors::IBoardIdentificationNotFound:
    case InternalInitializationErrors::UnknownSystemId:
    case InternalInitializationErrors::InvalidFPGAState:
        msg_to_ret = "Error while initializing the camera.";
        break;

    case CameraErrorCode::UnsupportedFeature:
    case UnsupportedFeatureErrors::ExtTriggerUnavailable:
        msg_to_ret = "Unsupported feature of the camera.";
        break;

    case CameraErrorCode::DeprecatedFeature:
        msg_to_ret = "Feature is deprecated. It will be removed in next releases.";
        break;

    default:
        msg_to_ret = "(unrecognized error)";
        break;
    }

    return msg_to_ret;
}
} // anonymous namespace

class CameraErrorCategory : public std::error_category {
public:
    CameraErrorCategory() {}
    CameraErrorCategory(const CameraErrorCategory &) = delete;

    virtual const char *name() const noexcept override {
        return "Metavision SDK Stream exception";
    }

    virtual std::string message(int err) const override {
        return get_error_message(err);
    }
};

const std::error_category &camera_error_category() {
    // The category singleton
    static CameraErrorCategory instance;
    return instance;
}

CameraException::CameraException(CameraErrorCodeType e, std::string additional_info) :
    BaseException(e, get_public_error_code(e), camera_error_category(), additional_info) {}

CameraException::CameraException(CameraErrorCodeType e) : CameraException(e, "") {}
} // namespace Metavision
