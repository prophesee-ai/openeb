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

#ifndef SRC_INC_SYSTEM_CONFIG_REGISTERMAP_H_
#define SRC_INC_SYSTEM_CONFIG_REGISTERMAP_H_

static RegmapElement SystemConfigRegisterMap[] = {
    // clang-format off
    {R, {{"ID", 0x0}}},          {F, {{"VALUE", 0, 8, 0x1c}}}, {R, {{"VERSION", 0x4}}},
    {F, {{"MICRO", 0, 8, 0x0}}}, {F, {{"MINOR", 8, 8, 0x0}}},  {F, {{"MAJOR", 16, 8, 0x0}}},
    {R, {{"BUILD_DATE", 0x8}}},  {F, {{"VALUE", 0, 32, 0x0}}}, {R, {{"VERSION_CONTROL_ID", 0xc}}},
    {F, {{"VALUE", 0, 32, 0x0}}}
    // clang-format on
};
static uint32_t SystemConfigRegisterMapSize = sizeof(SystemConfigRegisterMap) / sizeof(SystemConfigRegisterMap[0]);
#endif
