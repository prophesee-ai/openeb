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

#ifndef METAVISION_SDK_DRIVER_CAMERA_ERROR_CODE_INTERNAL_H
#define METAVISION_SDK_DRIVER_CAMERA_ERROR_CODE_INTERNAL_H

#include "metavision/sdk/driver/camera_error_code.h"

namespace Metavision {

/// Failed Initialization Error specific code
namespace InternalInitializationErrors {
enum : CameraErrorCodeType {
    // Failed initialization errors
    IBoardIdentificationNotFound = CameraErrorCode::InternalInitializationError | 0x1,
    ILLBiasesNotFound            = CameraErrorCode::InternalInitializationError | 0x2,
    IEventsStreamNotFound        = CameraErrorCode::InternalInitializationError | 0x3,
    IDeviceControlNotFound       = CameraErrorCode::InternalInitializationError | 0x4,
    IDecoderNotFound             = CameraErrorCode::InternalInitializationError | 0x5,
    ICDDecoderNotFound           = CameraErrorCode::InternalInitializationError | 0x6,
    IGeometryNotFound            = CameraErrorCode::InternalInitializationError | 0x9,
    IRoiNotFound                 = CameraErrorCode::InternalInitializationError | 0xA,
    UnknownSystemId              = CameraErrorCode::InternalInitializationError | 0xB,
    InvalidFPGAState             = CameraErrorCode::InternalInitializationError | 0xE,
};
}

/// Runtime Error specific code
namespace UnsupportedFeatureErrors {
enum : CameraErrorCodeType {
    RoiUnavailable                     = CameraErrorCode::UnsupportedFeature | 0x2,
    BiasesUnavailable                  = CameraErrorCode::UnsupportedFeature | 0x3,
    TriggerOutUnavailable              = CameraErrorCode::UnsupportedFeature | 0x4,
    ExtTriggerUnavailable              = CameraErrorCode::UnsupportedFeature | 0x5,
    AntiFlickerModuleUnavailable       = CameraErrorCode::UnsupportedFeature | 0xD,
    EventTrailFilterModuleUnavailable  = CameraErrorCode::UnsupportedFeature | 0xF,
    ErcModuleUnavailable               = CameraErrorCode::UnsupportedFeature | 0x10,
    OfflineStreamingControlUnavailable = CameraErrorCode::UnsupportedFeature | 0x11,
    RawRecordingUnavailable            = CameraErrorCode::UnsupportedFeature | 0x12,
    DeviceUnavailable                  = CameraErrorCode::UnsupportedFeature | 0x13,
    RawDataUnavailable                 = CameraErrorCode::UnsupportedFeature | 0x14,
    CDUnavailable                      = CameraErrorCode::UnsupportedFeature | 0x15,
    ERCCounterUnavailable              = CameraErrorCode::UnsupportedFeature | 0x16,
    FrameHistoUnavailable              = CameraErrorCode::UnsupportedFeature | 0x17,
    FrameDiffUnavailable               = CameraErrorCode::UnsupportedFeature | 0x18,
};
}

namespace BiasesErrors {
enum : CameraErrorCodeType {
    UnsupportedBiasFile = CameraErrorCode::BiasesError | 0x1,
    UnsupportedBias     = CameraErrorCode::BiasesError | 0x2,
};
}

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_CAMERA_ERROR_CODE_INTERNAL_H
