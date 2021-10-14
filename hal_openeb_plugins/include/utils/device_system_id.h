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
#ifndef DEVICE_SYSTEM_ID_H
#define DEVICE_SYSTEM_ID_H

namespace Metavision {

#define SYSTEM_FLAG_EVK_PROXY 0x80

enum SystemId : long {
    SYSTEM_CCAM2_STEREO                = 0x08,
    SYSTEM_CCAM2_STEREO_MAPPING        = 0x09,
    SYSTEM_STEREO_DEMO                 = 0x0A,
    SYSTEM_CCAM3_STEREO_LEFT_GTP       = 0x0B,
    SYSTEM_CCAM3_STEREO_LEFT           = 0x0D,
    SYSTEM_CCAM2_STEREO_MERGE_IMU      = 0x0F,
    SYSTEM_CCAM3_GOLDEN_FALLBACK       = 0x11,
    SYSTEM_CCAM5_GOLDEN_FALLBACK       = 0x12,
    SYSTEM_CCAM3_GEN2                  = 0x14,
    SYSTEM_CCAM3_GEN3                  = 0x15,
    SYSTEM_CCAM3_GEN31                 = 0x1C,
    SYSTEM_CCAM3_GEN4                  = 0x1A,                                              // Regular Gen 4 evk
    SYSTEM_CCAM4_GEN3                  = 0x17,                                              // Onboard rev A
    SYSTEM_CCAM4_GEN3_EVK              = (SYSTEM_FLAG_EVK_PROXY | SYSTEM_CCAM4_GEN3),       // Onboard rev A
    SYSTEM_CCAM4_GEN3_REV_B            = 0x19,                                              // Onboard rev B
    SYSTEM_CCAM4_GEN3_REV_B_EVK        = (SYSTEM_FLAG_EVK_PROXY | SYSTEM_CCAM4_GEN3_REV_B), // Onboard rev B
    SYSTEM_CCAM4_GEN4_EVK              = (SYSTEM_FLAG_EVK_PROXY | SYSTEM_CCAM3_GEN4),       // Onboard
    SYSTEM_CCAM4_GEN3_REV_B_EVK_BRIDGE = 0x1D,
    SYSTEM_CCAM5_GEN31                 = 0x23,
    SYSTEM_CCAM5_GEN4                  = 0x1E,
    SYSTEM_CCAM5_GEN4_FIXED_FRAME      = 0x2E,
    SYSTEM_CCAM5_GEN4_EVK_BRIDGE       = 0x20,
    SYSTEM_VISIONCAM_GEN3              = 0x21,
    SYSTEM_VISIONCAM_GEN3_EVK          = (SYSTEM_FLAG_EVK_PROXY | SYSTEM_VISIONCAM_GEN3),
    SYSTEM_VISIONCAM_GEN31             = 0x22,
    SYSTEM_VISIONCAM_GEN31_EVK         = (SYSTEM_FLAG_EVK_PROXY | SYSTEM_VISIONCAM_GEN31),
    SYSTEM_CX3_CCAM5_GEN4              = SYSTEM_CCAM5_GEN4,
    SYSTEM_EVK2_GEN31                  = 0x29,
    SYSTEM_EVK2_GEN4                   = 0x1F,
    SYSTEM_EVK2_GEN41                  = 0x27,
    SYSTEM_EVK3_GEN31_EVT2             = SYSTEM_CCAM5_GEN31,
    SYSTEM_EVK3_GEN31_EVT3             = 0x28,
    SYSTEM_EVK3_GEN41                  = 0x30,
    SYSTEM_INVALID_NO_FPGA             = static_cast<long>(0xFFFFFFFF)
};

} // namespace Metavision

#endif // DEVICE_SYSTEM_ID_H
