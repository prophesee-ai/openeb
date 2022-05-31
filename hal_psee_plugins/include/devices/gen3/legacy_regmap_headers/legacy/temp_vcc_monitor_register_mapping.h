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

#ifndef METAVISION_HAL_TEMP_VCC_MONITOR_REGISTER_MAPPING_H
#define METAVISION_HAL_TEMP_VCC_MONITOR_REGISTER_MAPPING_H

//-----------------------------------------------------------------------------
// Register Bank name is    TEMP_VCC_MONITOR
//-----------------------------------------------------------------------------

#define TEMP_VCC_MONITOR_REGISTER_MAPPING_BASE_ADDR 0x00000000 // regbank_address
#define TEMP_VCC_MONITOR_EVT_ENABLE_ADDR 0x00000000            // regbank_address
#define TEMP_VCC_MONITOR_EVT_ENABLE_DEFAULT 0x00000000         // TEMP_VCC_MONITOR_EVT_ENABLE_ADDR
#define TEMP_VCC_MONITOR_EVT_ENABLE_MASK 0x00013F1F            // TEMP_VCC_MONITOR_EVT_ENABLE_ADDR
#define CFG_ENABLE_ALL_EVT_BIT 0                               // TEMP_VCC_MONITOR_EVT_ENABLE_ADDR
#define CFG_ENABLE_TEMP_EVT_BIT 1                              // TEMP_VCC_MONITOR_EVT_ENABLE_ADDR
#define CFG_ENABLE_VCC_INT_EVT_BIT 2                           // TEMP_VCC_MONITOR_EVT_ENABLE_ADDR
#define CFG_ENABLE_VCC_AUX_EVT_BIT 3                           // TEMP_VCC_MONITOR_EVT_ENABLE_ADDR
#define CFG_ENABLE_VCC_BRAM_EVT_BIT 4                          // TEMP_VCC_MONITOR_EVT_ENABLE_ADDR
#define CFG_ENABLE_ALL_ALARM_BIT 8                             // TEMP_VCC_MONITOR_EVT_ENABLE_ADDR
#define CFG_ENABLE_OVER_TEMP_ALARM_BIT 9                       // TEMP_VCC_MONITOR_EVT_ENABLE_ADDR
#define CFG_ENABLE_USER_TEMP_ALARM_BIT 10                      // TEMP_VCC_MONITOR_EVT_ENABLE_ADDR
#define CFG_ENABLE_VCC_INT_ALARM_BIT 11                        // TEMP_VCC_MONITOR_EVT_ENABLE_ADDR
#define CFG_ENABLE_VCC_AUX_ALARM_BIT 12                        // TEMP_VCC_MONITOR_EVT_ENABLE_ADDR
#define CFG_ENABLE_VCC_BRAM_ALARM_BIT 13                       // TEMP_VCC_MONITOR_EVT_ENABLE_ADDR
#define CFG_ENABLE_SYSTEM_POWER_DOWN_BIT 16                    // TEMP_VCC_MONITOR_EVT_ENABLE_ADDR
#define TEMP_VCC_MONITOR_EVT_PERIOD_ADDR 0x00000004            // regbank_address
#define TEMP_VCC_MONITOR_EVT_PERIOD_WIDTH 24                   // TEMP_VCC_MONITOR_EVT_PERIOD_ADDR
#define TEMP_VCC_MONITOR_EVT_PERIOD_DEFAULT                         std_logic_vector(to_unsigned(100000, // TEMP_VCC_MONITOR_EVT_PERIOD_ADDR
#define TEMP_VCC_MONITOR_EVT_PERIOD_MASK 0x0FFFFFFF            // TEMP_VCC_MONITOR_EVT_PERIOD_ADDR
#define TEMP_VCC_MONITOR_EXT_TEMP_CONTROL_ADDR 0x00000008      // regbank_address
#define TEMP_VCC_MONITOR_EXT_TEMP_CONTROL_DEFAULT 0x00000000   // TEMP_VCC_MONITOR_EXT_TEMP_CONTROL_ADDR
#define TEMP_VCC_MONITOR_EXT_TEMP_CONTROL_MASK 0x00000007      // TEMP_VCC_MONITOR_EXT_TEMP_CONTROL_ADDR
#define STATUS_SYS_POWER_DOWN_BIT 0                            // TEMP_VCC_MONITOR_EXT_TEMP_CONTROL_ADDR
#define EXT_TEMP_MONITOR_EN_BIT 1                              // TEMP_VCC_MONITOR_EXT_TEMP_CONTROL_ADDR
#define EXT_TEMP_MONITOR_SPI_EN_BIT 2                          // TEMP_VCC_MONITOR_EXT_TEMP_CONTROL_ADDR
#define TEMP_VCC_MONITOR_EVK_EXT_TEMP_VALUE_ADDR 0x0000000C    // regbank_address
#define TEMP_VCC_MONITOR_EVK_EXT_TEMP_VALUE_DEFAULT 0x00000000 // TEMP_VCC_MONITOR_EVK_EXT_TEMP_VALUE_ADDR
#define TEMP_VCC_MONITOR_EVK_EXT_TEMP_VALUE_MASK 0x003FFFFF    // TEMP_VCC_MONITOR_EVK_EXT_TEMP_VALUE_ADDR
#define TEMP_VCC_MONITOR_REGISTER_MAPPING_LAST_ADDR 0x0000000C // regbank_address

#endif // METAVISION_HAL_TEMP_VCC_MONITOR_REGISTER_MAPPING_H
