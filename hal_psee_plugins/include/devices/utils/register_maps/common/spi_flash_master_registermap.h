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

#ifndef SRC_INC_SPIFLASH_MASTER_REGISTERMAP_H_
#define SRC_INC_SPIFLASH_MASTER_REGISTERMAP_H_

static RegmapElement SPIFlashMasterRegisterMap[] = {
    // clang-format off
    {R, {{"ADDRESS_INIT", 0x0}}},
    {F, {{"", 0, 24, 0x0}}},
    {R, {{"FLASH_ACCESS", 0x4}}},
    {F, {{"", 0, 32, 0x0}}},
    {R, {{"WRITE_ENABLE", 0x8}}},
    {F, {{"", 0, 1, 0x0}}},
    {R, {{"SECTOR_ERASE", 0xc}}},
    {F, {{"", 0, 24, 0x0}}},
    {R, {{"BULK_ERASE", 0x10}}},
    {F, {{"", 0, 1, 0x0}}},
    {R, {{"READ_STATUS", 0x14}}},
    {F, {{"WRITE_IN_PROGRESS", 24, 1, 0x0}}},
    {F, {{"WRITE_ENABLE_LATCH", 25, 1, 0x0}}},
    {F, {{"BP_LSB", 26, 3, 0x0}}},
    {F, {{"TOP_BOTTOM", 29, 1, 0x0}}},
    {F, {{"BP_MSB", 30, 1, 0x0}}},
    {F, {{"STATUS_WRITE_ENABLE", 31, 1, 0x0}}},
    {R, {{"READ_FLAG", 0x18}}},
    {F, {{"PROTECTION", 25, 1, 0x0}}},
    {F, {{"PROGRAM_SUSPEND", 26, 1, 0x0}}},
    {F, {{"PROGRAM", 28, 1, 0x0}}},
    {F, {{"ERASE", 29, 1, 0x0}}},
    {F, {{"ERASE_SUSPEND", 30, 1, 0x0}}},
    {F, {{"PROGRAM_ERASE", 31, 1, 0x0}}}
    // clang-format on
};
static uint32_t SPIFlashMasterRegisterMapSize =
    sizeof(SPIFlashMasterRegisterMap) / sizeof(SPIFlashMasterRegisterMap[0]);
#endif
