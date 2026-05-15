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

#ifndef METAVISION_HAL_CONFIG_REGISTERS_MAP_H
#define METAVISION_HAL_CONFIG_REGISTERS_MAP_H

// -----------------------------------
// REGBANK ADDR
// -----------------------------------

//-----------------------------------------------------------------------------
//  Register Bank Number
//-----------------------------------------------------------------------------
#define CCAM2_SYSTEM_CONTROL_REG_NB 25 // natural

//-----------------------------------------------------------------------------
//  Register Bank Base Address
//-----------------------------------------------------------------------------
#define CCAM2_SYSTEM_CONTROL_BASE_ADDRESS 0x00000000 // regbank_address

#define CCAM2IF_LEFT_BASE_ADDRESS 0x00000000
#define CCAM2IF_RIGHT_BASE_ADDRESS 0x00000400

//-----------------------------------------------------------------------------
//  Register Address Map
//-----------------------------------------------------------------------------
#define CCAM2_SYSTEM_CONTROL_BASE_ADDR 0x00000000      // regbank_address
#define TEP_ATIS_CONTROL_ADDR 0x00000000               // regbank_address
#define TEP_ATIS_BIASROI_UPDATE_VALUE0_ADDR 0x00000002 // regbank_address
#define TEP_ATIS_BIASROI_UPDATE_VALUE1_ADDR 0x00000004 // regbank_address
#define TEP_ATIS_BIAS_UPDATE_VALUE2_ADDR 0x00000006    // regbank_address
#define TEP_CCAM2_CONTROL_ADDR 0x00000008              // regbank_address
#define TEP_TRIGGERS_ADDR 0x0000000A                   // regbank_address
#define TEP_SYSTEM_STATUS_ADDR 0x0000000C              // regbank_address
#define TEP_FOUT_LSB_STATUS_ADDR 0x0000000E            // regbank_address
#define TEP_FOUT_MSB_STATUS_ADDR 0x00000010            // regbank_address
#define TEP_FIFO_WRCOUNT_STATUS_ADDR 0x00000012        // regbank_address
#define TEP_FIFO_CHECKPIX_STATUS_ADDR 0x00000014       // regbank_address
#define TEP_TLAST_REARMUS_ADDR 0x00000016              // regbank_address
#define TEP_OVERFLOW_HITCOUNT_ADDR 0x00000018          // regbank_address
#define TEP_CCAM2_MODE_ADDR 0x0000001A                 // regbank_address
#define TEP_SERIAL_LSB_ADDR 0x0000001C                 // regbank_address
#define TEP_SERIAL_MSB_ADDR 0x00000020                 // regbank_address
#define TEP_NOTIFY_PACKETCOUNT_ADDR 0x00000022         // regbank_address
#define TEP_SNFETCH_FADDR_LSB_ADDR 0x00000024          // regbank_address
#define TEP_SNFETCH_FADDR_MSB_ADDR 0x00000026          // regbank_address
#define TEP_SNFETCH_RDATA_LSB_ADDR 0x00000028          // regbank_address
#define TEP_SNFETCH_RDATA_MSB_ADDR 0x0000002A          // regbank_address
#define TEP_SNFETCH_READ_ITER_ADDR 0x0000002C          // regbank_address
#define TEP_SNFETCH_TIME_COUNT_ADDR 0x0000002E         // regbank_address
#define TEP_BIAS_LOAD_ITERATION_COUNT_ADDR 0x00000030  // regbank_address
#define TEP_FLASH_PROGRAM_SEL_SLAVE_ADDR 0x00000032    // regbank_address
#define SYSTEM_CONTROL_LAST_ADDR 0x00000032            // regbank_address

//-----------------------------------------------------------------------------
//  Register Bank Number
//-----------------------------------------------------------------------------
//#define   CCAM2_SYSTEM_CONTROL_REG_NB 25 // natural

//-----------------------------------------------------------------------------
//  Register Bank Base Address
//-----------------------------------------------------------------------------
//#define   CCAM2_SYSTEM_CONTROL_BASE_ADDRESS                           0x00000000 // regbank_address

//-----------------------------------------------------------------------------
//  Register Address Map
//-----------------------------------------------------------------------------
//#define   SYSTEM_CONTROL_ADDR                                         0x00000000 // regbank_address
//#define   ATIS_CONTROL_ADDR                                           0x00000000 // regbank_address
//#define   ATIS_BIASROI_UPDATE_VALUE0_ADDR                             0x00000002 // regbank_address
//#define   ATIS_BIASROI_UPDATE_VALUE1_ADDR                             0x00000004 // regbank_address
//#define   ATIS_BIAS_UPDATE_VALUE2_ADDR                                0x00000006 // regbank_address
//#define   CCAM2_CONTROL_ADDR                                          0x00000008 // regbank_address
//#define   TRIGGERS_ADDR                                               0x0000000A // regbank_address
//#define   SYSTEM_STATUS_ADDR                                          0x0000000C // regbank_address
//#define   FOUT_LSB_STATUS_ADDR                                        0x0000000E // regbank_address
//#define   FOUT_MSB_STATUS_ADDR                                        0x00000010 // regbank_address
//#define   FIFO_WRCOUNT_STATUS_ADDR                                    0x00000012 // regbank_address
//#define   FIFO_CHECKPIX_STATUS_ADDR                                   0x00000014 // regbank_address
//#define   TLAST_REARMUS_ADDR                                          0x00000016 // regbank_address
//#define   OVERFLOW_HITCOUNT_ADDR                                      0x00000018 // regbank_address
//#define   CCAM2_MODE_ADDR                                             0x0000001A // regbank_address
//#define   SERIAL_LSB_ADDR                                             0x0000001C // regbank_address
//#define   SERIAL_MSB_ADDR                                             0x00000020 // regbank_address
//#define   NOTIFY_PACKETCOUNT_ADDR                                     0x00000022 // regbank_address
//#define   SNFETCH_FADDR_LSB_ADDR                                      0x00000024 // regbank_address
//#define   SNFETCH_FADDR_MSB_ADDR                                      0x00000026 // regbank_address
//#define   SNFETCH_RDATA_LSB_ADDR                                      0x00000028 // regbank_address
//#define   SNFETCH_RDATA_MSB_ADDR                                      0x0000002A // regbank_address
//#define   SNFETCH_READ_ITER_ADDR                                      0x0000002C // regbank_address
//#define   SNFETCH_TIME_COUNT_ADDR                                     0x0000002E // regbank_address
//#define   BIAS_LOAD_ITERATION_COUNT_ADDR                              0x00000030 // regbank_address
//#define   FLASH_PROGRAM_SEL_SLAVE_ADDR                                0x00000032 // regbank_address
//#define   SYSTEM_CONTROL_LAST_ADDR                                    0x00000032 // regbank_address

// -----------------------------------
// GEN1 - GEN2 CONFIG REGISTERS
// -----------------------------------

// Mapping Bank SYSTEM_CONFIG from base address : 0x00000800 - Prefix : STEREO_
#define STEREO_SYSTEM_CONFIG_BASE_ADDR 0x00000800
#define STEREO_SYSTEM_CONFIG_ADDR 0x00000800
#define STEREO_SYSTEM_CONFIG_ID_ADDR 0x00000800
#define STEREO_SYSTEM_CONFIG_VERSION_ADDR 0x00000804
#define STEREO_SYSTEM_CONFIG_BUILD_DATE_ADDR 0x00000808
#define STEREO_SYSTEM_CONFIG_VERSION_CONTROL_ID_ADDR 0x0000080c
#define STEREO_SYSTEM_CONFIG_LAST_ADDR 0x0000080c
#define STEREO_STEREO_SYS_DATA0_ADDR 0x00000800
#define STEREO_STEREO_SYS_DATA1_ADDR 0x00000804
#define STEREO_STEREO_SYS_DATA2_ADDR 0x00000808
#define STEREO_STEREO_SYS_DATA4_ADDR 0x0000080c

#define STEREO_SYSTEM_CONTROL_BASE_ADDR 0x00000810
// Mapping Bank STEREO_SYSTEM_CONTROL from base address : 0x00000810 - Prefix : STEREO_
#define STEREO_SYSTEM_CONTROL_BASE_ADDRESS 0x00000810
#define STEREO_DEVICE_SYSTEM_CONTROL_ADDR 0x00000810

#endif // METAVISION_HAL_CONFIG_REGISTERS_MAP_H
