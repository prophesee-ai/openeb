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

#ifndef SRC_INC_MIPITX_REGISTERMAP_H_
#define SRC_INC_MIPITX_REGISTERMAP_H_

static RegmapElement MIPITXRegisterMap[] = {
    // clang-format off
    {R, {{"CONTROL", 0x0}}},           {F, {{"ENABLE", 0, 1, 0x0}}},      {F, {{"ENABLE_PACKET_TIMEOUT", 1, 1, 0x0}}},
    {R, {{"DATA_IDENTIFIER", 0x4}}},   {F, {{"DATA_TYPE", 0, 6, 0x30}}},  {F, {{"VIRTUAL_CHANNEL", 6, 2, 0x0}}},
    {R, {{"FRAME_PERIOD", 0x8}}},      {F, {{"VALUE_US", 0, 16, 0x3f0}}}, {R, {{"PACKET_TIMEOUT", 0xc}}},
    {F, {{"VALUE_US", 0, 16, 0x1f8}}}, {R, {{"PACKET_SIZE", 0x10}}},      {F, {{"VALUE", 0, 14, 0x2000}}},
    {R, {{"START_TIME", 0x14}}},       {F, {{"VALUE", 0, 16, 0x50}}},     {R, {{"START_FRAME_TIME", 0x18}}},
    {F, {{"VALUE", 0, 16, 0x50}}},     {R, {{"END_FRAME_TIME", 0x1c}}},   {F, {{"VALUE", 0, 16, 0x50}}},
    {R, {{"INTER_FRAME_TIME", 0x20}}}, {F, {{"VALUE", 0, 16, 0x50}}},     {R, {{"INTER_PACKET_TIME", 0x24}}},
    {F, {{"VALUE", 0, 16, 0x50}}}
    // clang-format on
};
static uint32_t MIPITXRegisterMapSize = sizeof(MIPITXRegisterMap) / sizeof(MIPITXRegisterMap[0]);
#endif
