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

#ifndef METAVISION_HAL_CCAM5_SINGLE_GEN31_SYSTEM_CONTROL_REGISTERMAP_H
#define METAVISION_HAL_CCAM5_SINGLE_GEN31_SYSTEM_CONTROL_REGISTERMAP_H

static RegmapElement ccam5_single_gen31_SystemControlRegisterMap[] = {
    // clang-format off
    {R, {{"ATIS_CONTROL", 0x0}}},
    {F, {{"EN_VDDA", 0, 1, 0x0}}},
    {F, {{"EN_VDDC", 1, 1, 0x0}}},
    {F, {{"EN_VDDD", 2, 1, 0x0}}},
    {F, {{"SENSOR_SOFT_RESET", 3, 1, 0x1}}},
    {F, {{"IN_EVT_NO_BLOCKING_MODE", 4, 1, 0x1}}},
    {F, {{"SISLEY_HVGA_REMAP_BYPASS", 8, 1, 0x1}}},
    {F, {{"MASTER_MODE", 12, 1, 0x0}}},
    {F, {{"USE_EXT_START", 14, 1, 0x0}}},
    {F, {{"SENSOR_TB_IOBUF_EN_N", 15, 1, 0x1}}},
    {F, {{"SENSOR_TB_PE_RST_N", 16, 1, 0x1}}},
    {F, {{"TD_RSTN", 18, 1, 0x1}}},
    {F, {{"EM_RSTN", 19, 1, 0x1}}},
    {F, {{"EN_EXT_CTRL_RSTB", 20, 1, 0x0}}},
    {F, {{"FLIP_X_EN", 21, 1, 0x0}}},
    {F, {{"FLIP_Y_EN", 22, 1, 0x0}}},
    {R, {{"BOARD_CONTROL_STATUS", 0x04}}},
    {F, {{"VERSION", 0, 2, 0x1}}},
    {R, {{"CCAM2_CONTROL", 0x08}}},
    {F, {{"HOST_IF_EN", 8, 1, 0x0}}},
    {F, {{"STEREO_MERGE_ENABLE", 9, 1, 0x0}}},
    {F, {{"ENABLE_OUT_OF_FOV", 11, 1, 0x0}}},
    {F, {{"TH_RECOVERY_BYPASS", 12, 1, 0x0}}},
    {F, {{"CCAM_ID", 13, 1, 0x0}}},
    {R, {{"CCAM2_TRIGGER", 0x0C}}},
    {F, {{"SOFT_RESET", 0, 1, 0x0}}},
    {R, {{"OUT_OF_FOV_FILTER_SIZE", 0x10}}},
    {F, {{"WIDTH", 0, 11, 640}}},
    {F, {{"VALUE", 16, 11, 480}}},
    {R, {{"OUT_OF_FOV_FILTER_ORIGIN", 0x14}}},
    {F, {{"X", 0, 11, 640}}},
    {F, {{"Y", 16, 11, 480}}},
    {R, {{"EVT_RATE_CONTROL", 0x18}}},
    {F, {{"ENABLE", 0, 1, 0}}},
    {F, {{"T_DROP_FACTOR", 16, 16, 0}}},
    // clang-format on
};
static uint32_t ccam5_single_gen31_SystemControlRegisterMapSize =
    sizeof(ccam5_single_gen31_SystemControlRegisterMap) / sizeof(ccam5_single_gen31_SystemControlRegisterMap[0]);
#endif // METAVISION_HAL_CCAM5_SINGLE_GEN31_SYSTEM_CONTROL_REGISTERMAP_H
