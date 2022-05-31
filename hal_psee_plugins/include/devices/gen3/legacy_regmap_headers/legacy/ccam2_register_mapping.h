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

#ifndef METAVISION_HAL_CCAM2_REGISTER_MAPPING_H
#define METAVISION_HAL_CCAM2_REGISTER_MAPPING_H

//-----------------------------------------------------------------------------
// Register Bank name is    CCAM2_SYSTEM_CONTROL
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//  Register Bank Number
//-----------------------------------------------------------------------------
#define CCAM2_SYSTEM_CONTROL_REG_NB 25 // natural

//-----------------------------------------------------------------------------
//  Register Bank Base Address
//-----------------------------------------------------------------------------
#define CCAM2_SYSTEM_CONTROL_BASE_ADDRESS 0x00000000 // regbank_address

//-----------------------------------------------------------------------------
//  Register Address Map
//-----------------------------------------------------------------------------
#define SYSTEM_CONTROL_ADDR 0x00000000                         // regbank_address
#define ATIS_CONTROL_ADDR 0x00000000                           // regbank_address
#define ATIS_BIASROI_UPDATE_VALUE0_ADDR 0x00000002             // regbank_address
#define ATIS_BIASROI_UPDATE_VALUE1_ADDR 0x00000004             // regbank_address
#define ATIS_BIAS_UPDATE_VALUE2_ADDR 0x00000006                // regbank_address
#define CCAM2_CONTROL_ADDR 0x00000008                          // regbank_address
#define TRIGGERS_ADDR 0x0000000A                               // regbank_address
#define SYSTEM_STATUS_ADDR 0x0000000C                          // regbank_address
#define FOUT_LSB_STATUS_ADDR 0x0000000E                        // regbank_address
#define FOUT_MSB_STATUS_ADDR 0x00000010                        // regbank_address
#define FIFO_WRCOUNT_STATUS_ADDR 0x00000012                    // regbank_address
#define FIFO_CHECKPIX_STATUS_ADDR 0x00000014                   // regbank_address
#define TLAST_REARMUS_ADDR 0x00000016                          // regbank_address
#define OVERFLOW_HITCOUNT_ADDR 0x00000018                      // regbank_address
#define CCAM2_MODE_ADDR 0x0000001A                             // regbank_address
#define SERIAL_LSB_ADDR 0x0000001C                             // regbank_address
#define SERIAL_MSB_ADDR 0x00000020                             // regbank_address
#define NOTIFY_PACKETCOUNT_ADDR 0x00000022                     // regbank_address
#define SNFETCH_FADDR_LSB_ADDR 0x00000024                      // regbank_address
#define SNFETCH_FADDR_MSB_ADDR 0x00000026                      // regbank_address
#define SNFETCH_RDATA_LSB_ADDR 0x00000028                      // regbank_address
#define SNFETCH_RDATA_MSB_ADDR 0x0000002A                      // regbank_address
#define SNFETCH_READ_ITER_ADDR 0x0000002C                      // regbank_address
#define SNFETCH_TIME_COUNT_ADDR 0x0000002E                     // regbank_address
#define BIAS_LOAD_ITERATION_COUNT_ADDR 0x00000030              // regbank_address
#define FLASH_PROGRAM_SEL_SLAVE_ADDR 0x00000032                // regbank_address
#define SYSTEM_CONTROL_LAST_ADDR 0x00000032                    // regbank_address
#define TEP_ATIS_CONTROL_EN_VDDA_BIT_IDX 0                     // SYSTEM_CONTROL_LAST_ADDR
#define TEP_ATIS_CONTROL_EN_VDDC_BIT_IDX 1                     // SYSTEM_CONTROL_LAST_ADDR
#define TEP_ATIS_CONTROL_EN_VDDD_BIT_IDX 2                     // SYSTEM_CONTROL_LAST_ADDR
#define TEP_ATIS_CONTROL_SENSOR_SOFT_RESET_BIT_IDX 3           // SYSTEM_CONTROL_LAST_ADDR
#define TEP_ATIS_CONTROL_IN_EVT_NO_BLOCKING_MODE_BIT_IDX 4     // SYSTEM_CONTROL_LAST_ADDR
#define TEP_ATIS_CONTROL_SISLEY_HVGA_REMAP_BYPASS_BIT_IDX 8    // SYSTEM_CONTROL_LAST_ADDR
#define TEP_ATIS_CONTROL_ROI_TD_RSTN_BIT_IDX 18                // SYSTEM_CONTROL_LAST_ADDR
#define TEP_ATIS_CONTROL_EN_EXT_CTRL_RSTB_BIT_IDX 20           // SYSTEM_CONTROL_LAST_ADDR
#define TEP_ATIS_CONTROL_EN_VDDA_DEFAULT 0                     // SYSTEM_CONTROL_LAST_ADDR
#define TEP_ATIS_CONTROL_EN_VDDC_DEFAULT 0                     // SYSTEM_CONTROL_LAST_ADDR
#define TEP_ATIS_CONTROL_EN_VDDD_DEFAULT 0                     // SYSTEM_CONTROL_LAST_ADDR
#define TEP_ATIS_CONTROL_SENSOR_SOFT_RESET_DEFAULT 1           // SYSTEM_CONTROL_LAST_ADDR
#define TEP_ATIS_CONTROL_IN_EVT_NO_BLOCKING_MODE_DEFAULT 1     // SYSTEM_CONTROL_LAST_ADDR
#define TEP_ATIS_CONTROL_SISLEY_HVGA_REMAP_BYPASS_DEFAULT 1    // SYSTEM_CONTROL_LAST_ADDR
#define TEP_ATIS_CONTROL_ROI_TD_RSTN_DEFAULT 0                 // SYSTEM_CONTROL_LAST_ADDR
#define TEP_ATIS_CONTROL_EN_EXT_CTRL_RSTB_DEFAULT 0            // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_CONTROL_ENABLE_64BITS_EVENT_BIT_IDX 3        // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_CONTROL_BYPASS_MAPPING_BIT_IDX 5             // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_CONTROL_HOST_IF_ENABLE_BIT_IDX 8             // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_CONTROL_STEREO_MERGE_ENABLE_BIT_IDX 9        // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_CONTROL_ENABLE_IMU_BIT_IDX 10                // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_CONTROL_ENABLE_OUT_OF_FOV_BIT_IDX 11         // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_CONTROL_ENABLE_64BITS_EVENT_DEFAULT 1        // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_CONTROL_BYPASS_MAPPING_DEFAULT 1             // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_CONTROL_MODE_DEFAULT 0                       // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_CONTROL_MODE_INIT 0                          // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_CONTROL_MODE_MASTER 1                        // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_CONTROL_MODE_SLAVE 2                         // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_CONTROL_ENABLE_IMU_DEFAULT 0                 // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_CONTROL_ENABLE_OUT_OF_FOV_DEFAULT 0          // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_MODE_MODE_DEFAULT 0                          // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_MODE_MODE_INIT 0                             // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_MODE_MODE_MASTER 1                           // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_MDOE_MODE_SLAVE 2                            // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_MODE_MODE_BIT_IDX 0                          // SYSTEM_CONTROL_LAST_ADDR
#define TEP_CCAM2_MODE_ATIS_DEBUG_SEL_BIT_IDX 4                // SYSTEM_CONTROL_LAST_ADDR
#define TEP_TRIGGER_SOFT_RESET_BIT_IDX 0                       // SYSTEM_CONTROL_LAST_ADDR
#define TEP_TRIGGER_BIAS_DIN_VALID_BIT_IDX 1                   // SYSTEM_CONTROL_LAST_ADDR
#define TEP_TRIGGER_TS_RESET_BIT_IDX 2                         // SYSTEM_CONTROL_LAST_ADDR
#define TEP_TRIGGER_ROI_DIN_BIT_IDX 3                          // SYSTEM_CONTROL_LAST_ADDR
#define TEP_TRIGGER_FIFO_RESET_BIT_IDX 7                       // SYSTEM_CONTROL_LAST_ADDR
#define TEP_TRIGGER_FLASH_ACCESS_BIT_IDX 8                     // SYSTEM_CONTROL_LAST_ADDR
#define TEP_TRIGGER_MAPPING_FETCH_BIT_IDX 9                    // SYSTEM_CONTROL_LAST_ADDR
#define TEP_TRIGGER_PROG_DAC_BIT_IDX 10                        // SYSTEM_CONTROL_LAST_ADDR
#define TEP_TRIGGER_DAC_DIN_VALID_BIT_IDX 11                   // SYSTEM_CONTROL_LAST_ADDR
#define TEP_BIAS_LOAD_ITERATION_COUNT_BIT_IDX 0                // SYSTEM_CONTROL_LAST_ADDR
#define TEP_FLASH_PROGRAM_CONTROL_SEL_SLAVE_BIT_IDX 0          // SYSTEM_CONTROL_LAST_ADDR
#define TEP_FLASH_PROGRAM_FX3_ACCESS_LOCKDOWN_VALUE_BIT_IDX 16 // SYSTEM_CONTROL_LAST_ADDR
#define DEFAULT_SERIAL_NBR 0x00000042                          // SYSTEM_CONTROL_LAST_ADDR
#define SERIAL_NBR_ADDR_IN_FLASH 0x258000                      // SYSTEM_CONTROL_LAST_ADDR
#define CCAM2_CONTROL_INIT 0b00                                // SYSTEM_CONTROL_LAST_ADDR
#define CCAM2_CONTROL_MASTER 0b01                              // SYSTEM_CONTROL_LAST_ADDR
#define CCAM2_CONTROL_SLAVE 0b10                               // SYSTEM_CONTROL_LAST_ADDR

#endif // METAVISION_HAL_CCAM2_REGISTER_MAPPING_H
