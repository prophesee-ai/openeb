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

#ifndef METAVISION_HAL_ATIS_IF_MONITOR_REGISTER_MAPPING_H
#define METAVISION_HAL_ATIS_IF_MONITOR_REGISTER_MAPPING_H

//-----------------------------------------------------------------------------
// Register Bank name is    ATIF_IF_MONITOR
//-----------------------------------------------------------------------------

#define ATIS_IF_MONITOR_CFG_ENABLE_ADDR 0x00000000      // regbank_address
#define ATIS_IF_MONITOR_CFG_IDLE_TIME_ADDR 0x00000004   // regbank_address
#define ATIS_IF_MONITOR_CFG_TIMEOUT_THR_ADDR 0x00000008 // regbank_address
#define CFG_ENABLE_ALL_EVT_BIT 0                        // ATIS_IF_MONITOR_CFG_TIMEOUT_THR_ADDR
#define CFG_ENABLE_TD_IDLE_TIME_EVT_BIT 1               // ATIS_IF_MONITOR_CFG_TIMEOUT_THR_ADDR
#define CFG_ENABLE_TD_IDLE_TIMEOUT_EVT_BIT 2            // ATIS_IF_MONITOR_CFG_TIMEOUT_THR_ADDR
#define CFG_ENABLE_APS_IDLE_TIME_EVT_BIT 3              // ATIS_IF_MONITOR_CFG_TIMEOUT_THR_ADDR
#define CFG_ENABLE_APS_IDLE_TIMEOUT_EVT_BIT 4           // ATIS_IF_MONITOR_CFG_TIMEOUT_THR_ADDR
#define CFG_ENABLE_GLOBAL_ILLUMINATION_EVT_BIT 5        // ATIS_IF_MONITOR_CFG_TIMEOUT_THR_ADDR

#endif // METAVISION_HAL_ATIS_IF_MONITOR_REGISTER_MAPPING_H
