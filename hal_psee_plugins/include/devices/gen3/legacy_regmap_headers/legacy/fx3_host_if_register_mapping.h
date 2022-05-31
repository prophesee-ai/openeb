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

#ifndef METAVISION_HAL_FX3_HOST_IF_REGISTER_MAPPING_H
#define METAVISION_HAL_FX3_HOST_IF_REGISTER_MAPPING_H

//-----------------------------------------------------------------------------
// Register Bank name is    FX3_HOST_IF
//-----------------------------------------------------------------------------

#define FX3_HOST_IF_REGISTER_MAPPING_BASE_ADDR 0x00000000 // regbank_address
#define FX3_HOST_IF_PKT_END_ENABLE_ADDR 0x00000000        // regbank_address
#define FX3_HOST_IF_PKT_END_INTERVAL_US_ADDR 0x00000004   // regbank_address
#define FX3_HOST_IF_PKT_END_DATA_COUNT_ADDR 0x00000008    // regbank_address
#define REG00_ADDR_BIT_ADDR 0x00000000                    // regbank_address
#define REG01_ADDR_BIT_ADDR 0x00000004                    // regbank_address
#define REG02_ADDR_BIT_ADDR 0x00000008                    // regbank_address
#define REG03_ADDR_BIT_ADDR 0x0000000C                    // regbank_address
#define REG04_ADDR_BIT_ADDR 0x00000010                    // regbank_address
#define REG05_ADDR_BIT_ADDR 0x00000014                    // regbank_address
#define REG06_ADDR_BIT_ADDR 0x00000018                    // regbank_address
#define REG07_ADDR_BIT_ADDR 0x0000001C                    // regbank_address
#define REG08_ADDR_BIT_ADDR 0x00000020                    // regbank_address
#define REG09_ADDR_BIT_ADDR 0x00000024                    // regbank_address
#define REG10_ADDR_BIT_ADDR 0x00000028                    // regbank_address
#define REG11_ADDR_BIT_ADDR 0x0000002C                    // regbank_address
#define REG12_ADDR_BIT_ADDR 0x00000030                    // regbank_address
#define REG13_ADDR_BIT_ADDR 0x00000034                    // regbank_address
#define REG14_ADDR_BIT_ADDR 0x00000038                    // regbank_address
#define REG15_ADDR_BIT_ADDR 0x0000003C                    // regbank_address
#define FX3_HOST_IF_LAST_ADDR 0x0000003C                  // regbank_address
//  last register bank address defined at 0x0000003C

#define SHORT_PACKET_ENABLE_C 0x00000001            // FX3_HOST_IF_LAST_ADDR
#define SHORT_PACKET_INTERVAL_US_WIDTH_C 0x00000400 // FX3_HOST_IF_LAST_ADDR
#define SHORT_PACKET_SKIP_DATA_COUNT_C 0x00000400   // FX3_HOST_IF_LAST_ADDR
#define REG00_SHORT_PACKET_ENABLE_IDX_C 0           // FX3_HOST_IF_LAST_ADDR
#define REG00_SHORT_PACKET_ENABLE_SKIP_IDX_C 1      // FX3_HOST_IF_LAST_ADDR

#endif // METAVISION_HAL_FX3_HOST_IF_REGISTER_MAPPING_H
