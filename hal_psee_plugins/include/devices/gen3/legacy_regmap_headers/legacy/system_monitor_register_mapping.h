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

#ifndef METAVISION_HAL_SYSTEM_MONITOR_REGISTER_MAPPING_H
#define METAVISION_HAL_SYSTEM_MONITOR_REGISTER_MAPPING_H

//-----------------------------------------------------------------------------
// Register Bank name is    TEMP_VCC_XADC_MONITOR
//-----------------------------------------------------------------------------

#define SYSTEM_MONITOR_XADC_REGISTER_MAPPING_BASE_ADDR 0x00000000 // regbank_address
#define SYSTEM_MONITOR_FPGA_TEMP_ADDR 0x00000000                  // regbank_address
#define SYSTEM_MONITOR_VCC_INT_ADDR 0x00000002                    // regbank_address
#define SYSTEM_MONITOR_VCC_AUX_ADDR 0x00000004                    // regbank_address
#define SYSTEM_MONITOR_VP_VN_ADDR 0x00000006                      // regbank_address
#define SYSTEM_MONITOR_VREFP_ADDR 0x00000008                      // regbank_address
#define SYSTEM_MONITOR_VREFN_ADDR 0x0000000A                      // regbank_address
#define SYSTEM_MONITOR_VCC_BRAM_ADDR 0x0000000C                   // regbank_address
#define SYSTEM_MONITOR_XADC_SUPPLY_OFFSET_ADDR 0x00000010         // regbank_address
#define SYSTEM_MONITOR_XADC_OFFSET_ADDR 0x00000012                // regbank_address
#define SYSTEM_MONITOR_XADC_GAIN_ERROR_ADDR 0x00000014            // regbank_address
#define SYSTEM_MONITOR_VAUX0_ADDR 0x00000020                      // regbank_address
#define SYSTEM_MONITOR_VAUX1_ADDR 0x00000022                      // regbank_address
#define SYSTEM_MONITOR_VAUX2_ADDR 0x00000024                      // regbank_address
#define SYSTEM_MONITOR_VAUX3_ADDR 0x00000026                      // regbank_address
#define SYSTEM_MONITOR_VAUX4_ADDR 0x00000028                      // regbank_address
#define SYSTEM_MONITOR_VAUX5_ADDR 0x0000002A                      // regbank_address
#define SYSTEM_MONITOR_VAUX6_ADDR 0x0000002C                      // regbank_address
#define SYSTEM_MONITOR_VAUX7_ADDR 0x0000002E                      // regbank_address
#define SYSTEM_MONITOR_VAUX8_ADDR 0x00000030                      // regbank_address
#define SYSTEM_MONITOR_VAUX9_ADDR 0x00000032                      // regbank_address
#define SYSTEM_MONITOR_VAUX10_ADDR 0x00000034                     // regbank_address
#define SYSTEM_MONITOR_VAUX11_ADDR 0x00000036                     // regbank_address
#define SYSTEM_MONITOR_VAUX12_ADDR 0x00000038                     // regbank_address
#define SYSTEM_MONITOR_VAUX13_ADDR 0x0000003A                     // regbank_address
#define SYSTEM_MONITOR_VAUX14_ADDR 0x0000003C                     // regbank_address
#define SYSTEM_MONITOR_VAUX15_ADDR 0x0000003E                     // regbank_address
#define SYSTEM_MONITOR_MAX_TEMP_ADDR 0x00000040                   // regbank_address
#define SYSTEM_MONITOR_MAX_VCC_INT_ADDR 0x00000042                // regbank_address
#define SYSTEM_MONITOR_MAX_VCC_AUX_ADDR 0x00000044                // regbank_address
#define SYSTEM_MONITOR_MAX_VCC_BRAM_ADDR 0x00000046               // regbank_address
#define SYSTEM_MONITOR_MIN_TEMP_ADDR 0x00000048                   // regbank_address
#define SYSTEM_MONITOR_MIN_VCC_INT_ADDR 0x0000004A                // regbank_address
#define SYSTEM_MONITOR_MIN_VCC_AUX_ADDR 0x0000004C                // regbank_address
#define SYSTEM_MONITOR_MIN_VCC_BRAM_ADDR 0x0000004E               // regbank_address
#define SYSTEM_MONITOR_XADC_FLAGS_ADDR 0x0000007E                 // regbank_address
#define SYSTEM_MONITOR_XADC_CONF_REG0_ADDR 0x00000080             // regbank_address
#define SYSTEM_MONITOR_XADC_CONF_REG1_ADDR 0x00000082             // regbank_address
#define SYSTEM_MONITOR_XADC_CONF_REG2_ADDR 0x00000084             // regbank_address
#define SYSTEM_MONITOR_XADC_SEQ_REG0_ADDR 0x00000090              // regbank_address
#define SYSTEM_MONITOR_XADC_SEQ_REG1_ADDR 0x00000092              // regbank_address
#define SYSTEM_MONITOR_XADC_SEQ_REG2_ADDR 0x00000094              // regbank_address
#define SYSTEM_MONITOR_XADC_SEQ_REG3_ADDR 0x00000096              // regbank_address
#define SYSTEM_MONITOR_XADC_SEQ_REG4_ADDR 0x00000098              // regbank_address
#define SYSTEM_MONITOR_XADC_SEQ_REG5_ADDR 0x0000009A              // regbank_address
#define SYSTEM_MONITOR_XADC_SEQ_REG6_ADDR 0x0000009C              // regbank_address
#define SYSTEM_MONITOR_XADC_SEQ_REG7_ADDR 0x0000009E              // regbank_address
#define SYSTEM_MONITOR_XADC_ALARM_THR_REG0_ADDR 0x000000A0        // regbank_address
#define SYSTEM_MONITOR_XADC_ALARM_THR_REG1_ADDR 0x000000A2        // regbank_address
#define SYSTEM_MONITOR_XADC_ALARM_THR_REG2_ADDR 0x000000A4        // regbank_address
#define SYSTEM_MONITOR_XADC_ALARM_THR_REG3_ADDR 0x000000A6        // regbank_address
#define SYSTEM_MONITOR_XADC_ALARM_THR_REG4_ADDR 0x000000A8        // regbank_address
#define SYSTEM_MONITOR_XADC_ALARM_THR_REG5_ADDR 0x000000AA        // regbank_address
#define SYSTEM_MONITOR_XADC_ALARM_THR_REG6_ADDR 0x000000AC        // regbank_address
#define SYSTEM_MONITOR_XADC_ALARM_THR_REG7_ADDR 0x000000AE        // regbank_address
#define SYSTEM_MONITOR_XADC_ALARM_THR_REG8_ADDR 0x000000B0        // regbank_address
#define SYSTEM_MONITOR_XADC_ALARM_THR_REG9_ADDR 0x000000B2        // regbank_address
#define SYSTEM_MONITOR_XADC_ALARM_THR_REG10_ADDR 0x000000B4       // regbank_address
#define SYSTEM_MONITOR_XADC_ALARM_THR_REG11_ADDR 0x000000B6       // regbank_address
#define SYSTEM_MONITOR_XADC_ALARM_THR_REG12_ADDR 0x000000B8       // regbank_address
#define SYSTEM_MONITOR_XADC_ALARM_THR_REG13_ADDR 0x000000BA       // regbank_address
#define SYSTEM_MONITOR_XADC_ALARM_THR_REG14_ADDR 0x000000BC       // regbank_address
#define SYSTEM_MONITOR_XADC_ALARM_THR_REG15_ADDR 0x000000BE       // regbank_address

#endif // METAVISION_HAL_SYSTEM_MONITOR_REGISTER_MAPPING_H
