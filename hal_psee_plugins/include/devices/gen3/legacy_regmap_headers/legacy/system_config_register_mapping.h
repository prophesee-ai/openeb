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

#ifndef METAVISION_HAL_SYSTEM_CONFIG_REGISTER_MAPPING_H
#define METAVISION_HAL_SYSTEM_CONFIG_REGISTER_MAPPING_H

//-----------------------------------------------------------------------------
// Register Bank name is    SYSTEM_CONFIG
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//  Register Bank Base Address
//-----------------------------------------------------------------------------
#define SYSTEM_CONFIG_BASE_ADDRESS 0x00000000 // regbank_address

//-----------------------------------------------------------------------------
//  Register Address Map
//-----------------------------------------------------------------------------
#define SYSTEM_CONFIG_BASE_ADDR 0x00000000               // regbank_address
#define SYSTEM_CONFIG_ADDR 0x00000000                    // regbank_address
#define SYSTEM_CONFIG_ID_ADDR 0x00000000                 // regbank_address
#define SYSTEM_CONFIG_VERSION_ADDR 0x00000004            // regbank_address
#define SYSTEM_CONFIG_BUILD_DATE_ADDR 0x00000008         // regbank_address
#define SYSTEM_CONFIG_VERSION_CONTROL_ID_ADDR 0x0000000C // regbank_address
#define SYSTEM_CONFIG_LAST_ADDR 0x0000000C               // regbank_address
//  last register bank address defined at 0x0000000C

#define STEREO_SYS_DATA0_ADDR 0x00000000 // regbank_address
#define STEREO_SYS_DATA1_ADDR 0x00000004 // regbank_address
#define STEREO_SYS_DATA2_ADDR 0x00000008 // regbank_address
#define STEREO_SYS_DATA4_ADDR 0x0000000C // regbank_address

#endif // METAVISION_HAL_SYSTEM_CONFIG_REGISTER_MAPPING_H
