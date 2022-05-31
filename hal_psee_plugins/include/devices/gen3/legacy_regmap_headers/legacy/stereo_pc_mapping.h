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

#ifndef METAVISION_HAL_STEREO_PC_MAPPING_H
#define METAVISION_HAL_STEREO_PC_MAPPING_H

#include "boards/utils/config_registers_map.h"
#include "devices/gen3/legacy_regmap_headers/legacy/atis_if_monitor_register_mapping.h"
#include "devices/gen3/legacy_regmap_headers/legacy/ccam2_register_mapping.h"
#include "devices/gen3/legacy_regmap_headers/legacy/ext_trigger_monitor_register_mapping.h"
#include "devices/gen3/legacy_regmap_headers/legacy/fx3_host_if_register_mapping.h"
#include "devices/gen3/legacy_regmap_headers/legacy/ibgen_register_mapping.h"
#include "devices/gen3/legacy_regmap_headers/legacy/roi_register_mapping.h"
#include "devices/gen3/legacy_regmap_headers/legacy/sensor_ctrl_register_mapping.h"
#include "devices/gen3/legacy_regmap_headers/legacy/system_config_register_mapping.h"
#include "devices/gen3/legacy_regmap_headers/legacy/temp_vcc_monitor_register_mapping.h"
#include "devices/gen3/legacy_regmap_headers/legacy/temp_vcc_monitor_xadc_register_mapping.h"

// clang-format off

#define   CCAM3_BASE_ADDRESS                                          0x00000000 

#define   CCAM3_SYS_REG_BASE_ADDR                                     0x00000000 
//Mapping Bank CCAM2_SYSTEM_CONTROL from base address : 0x00000000 - Prefix : CCAM3_ 
#define   CCAM3_SYSTEM_CONTROL_ADDR                                   0x00000000 
#define   CCAM3_ATIS_CONTROL_ADDR                                     0x00000000 
#define   CCAM3_ATIS_BIASROI_UPDATE_VALUE0_ADDR                       0x00000002 
#define   CCAM3_ATIS_BIASROI_UPDATE_VALUE1_ADDR                       0x00000004 
#define   CCAM3_ATIS_BIAS_UPDATE_VALUE2_ADDR                          0x00000006 
#define   CCAM3_CCAM2_CONTROL_ADDR                                    0x00000008 
#define   CCAM3_TRIGGERS_ADDR                                         0x0000000a 
#define   CCAM3_SYSTEM_STATUS_ADDR                                    0x0000000c 
#define   CCAM3_FOUT_LSB_STATUS_ADDR                                  0x0000000e 
#define   CCAM3_FOUT_MSB_STATUS_ADDR                                  0x00000010 
#define   CCAM3_FIFO_WRCOUNT_STATUS_ADDR                              0x00000012 
#define   CCAM3_FIFO_CHECKPIX_STATUS_ADDR                             0x00000014 
#define   CCAM3_TLAST_REARMUS_ADDR                                    0x00000016 
#define   CCAM3_OVERFLOW_HITCOUNT_ADDR                                0x00000018 
#define   CCAM3_CCAM2_MODE_ADDR                                       0x0000001a 
#define   CCAM3_SERIAL_LSB_ADDR                                       0x0000001c 
#define   CCAM3_SERIAL_MSB_ADDR                                       0x00000020 
#define   CCAM3_NOTIFY_PACKETCOUNT_ADDR                               0x00000022 
#define   CCAM3_SNFETCH_FADDR_LSB_ADDR                                0x00000024 
#define   CCAM3_SNFETCH_FADDR_MSB_ADDR                                0x00000026 
#define   CCAM3_SNFETCH_RDATA_LSB_ADDR                                0x00000028 
#define   CCAM3_SNFETCH_RDATA_MSB_ADDR                                0x0000002a 
#define   CCAM3_SNFETCH_READ_ITER_ADDR                                0x0000002c 
#define   CCAM3_SNFETCH_TIME_COUNT_ADDR                               0x0000002e 
#define   CCAM3_BIAS_LOAD_ITERATION_COUNT_ADDR                        0x00000030 
#define   CCAM3_FLASH_PROGRAM_SEL_SLAVE_ADDR                          0x00000032 
#define   CCAM3_OUT_OF_FOV_FILTER_WIDTH_ADDR                          0x00000034 
#define   CCAM3_OUT_OF_FOV_FILTER_HEIGHT_ADDR                         0x00000036 
#define   CCAM3_SYSTEM_CONTROL_LAST_ADDR                              0x00000036 


#define   CCAM3_SYSTEM_MONITOR_BASE_ADDR                              0x00000040 

#define   CCAM3_TEMP_VCC_MONITOR_XADC_BASE_ADDR                       0x00000040 
//Mapping Bank TEMP_VCC_MONITOR_XADC from base address : 0x00000040 - Prefix : CCAM3_ 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_REGISTER_MAPPING_BASE_ADDR      0x00000040 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_TEMP_ADDR                       0x00000040 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VCC_INT_ADDR                    0x00000042 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VCC_AUX_ADDR                    0x00000044 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VP_VN_ADDR                      0x00000046 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VREFP_ADDR                      0x00000048 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VREFN_ADDR                      0x0000004a 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VCC_BRAM_ADDR                   0x0000004c 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_SUPPLY_OFFSET_ADDR              0x00000050 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_OFFSET_ADDR                     0x00000052 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_GAIN_ERROR_ADDR                 0x00000054 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VAUX0_ADDR                      0x00000060 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VAUX1_ADDR                      0x00000062 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VAUX2_ADDR                      0x00000064 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VAUX3_ADDR                      0x00000066 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VAUX4_ADDR                      0x00000068 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VAUX5_ADDR                      0x0000006a 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VAUX6_ADDR                      0x0000006c 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VAUX7_ADDR                      0x0000006e 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VAUX8_ADDR                      0x00000070 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VAUX9_ADDR                      0x00000072 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VAUX10_ADDR                     0x00000074 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VAUX11_ADDR                     0x00000076 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VAUX12_ADDR                     0x00000078 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VAUX13_ADDR                     0x0000007a 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VAUX14_ADDR                     0x0000007c 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_VAUX15_ADDR                     0x0000007e 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_MAX_TEMP_ADDR                   0x00000080 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_MAX_VCC_INT_ADDR                0x00000082 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_MAX_VCC_AUX_ADDR                0x00000084 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_MAX_VCC_BRAM_ADDR               0x00000086 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_MIN_TEMP_ADDR                   0x00000088 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_MIN_VCC_INT_ADDR                0x0000008a 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_MIN_VCC_AUX_ADDR                0x0000008c 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_MIN_VCC_BRAM_ADDR               0x0000008e 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_FLAGS_ADDR                      0x000000be 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_CONF_REG0_ADDR                  0x000000c0 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_CONF_REG1_ADDR                  0x000000c2 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_CONF_REG2_ADDR                  0x000000c4 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_SEQ_REG0_ADDR                   0x000000d0 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_SEQ_REG1_ADDR                   0x000000d2 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_SEQ_REG2_ADDR                   0x000000d4 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_SEQ_REG3_ADDR                   0x000000d6 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_SEQ_REG4_ADDR                   0x000000d8 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_SEQ_REG5_ADDR                   0x000000da 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_SEQ_REG6_ADDR                   0x000000dc 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_SEQ_REG7_ADDR                   0x000000de 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_ALARM_THR_REG0_ADDR             0x000000e0 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_ALARM_THR_REG1_ADDR             0x000000e2 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_ALARM_THR_REG2_ADDR             0x000000e4 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_ALARM_THR_REG3_ADDR             0x000000e6 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_ALARM_THR_REG4_ADDR             0x000000e8 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_ALARM_THR_REG5_ADDR             0x000000ea 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_ALARM_THR_REG6_ADDR             0x000000ec 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_ALARM_THR_REG7_ADDR             0x000000ee 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_ALARM_THR_REG8_ADDR             0x000000f0 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_ALARM_THR_REG9_ADDR             0x000000f2 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_ALARM_THR_REG10_ADDR            0x000000f4 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_ALARM_THR_REG11_ADDR            0x000000f6 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_ALARM_THR_REG12_ADDR            0x000000f8 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_ALARM_THR_REG13_ADDR            0x000000fa 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_ALARM_THR_REG14_ADDR            0x000000fc 
#define   CCAM3_TEMP_VCC_MONITOR_XADC_ALARM_THR_REG15_ADDR            0x000000fe 


#define   CCAM3_TEMP_VCC_MONITOR_BASE_ADDR                            0x00000140 
//Mapping Bank TEMP_VCC_MONITOR from base address : 0x00000140 - Prefix : CCAM3_ 
#define   CCAM3_TEMP_VCC_MONITOR_REGISTER_MAPPING_BASE_ADDR           0x00000140 
#define   CCAM3_TEMP_VCC_MONITOR_EVT_ENABLE_ADDR                      0x00000140 
#define   CCAM3_TEMP_VCC_MONITOR_EVT_PERIOD_ADDR                      0x00000144 
#define   CCAM3_TEMP_VCC_MONITOR_EXT_TEMP_CONTROL_ADDR                0x00000148 
#define   CCAM3_TEMP_VCC_MONITOR_EVK_EXT_TEMP_VALUE_ADDR              0x0000014c 
#define   CCAM3_TEMP_VCC_MONITOR_REGISTER_MAPPING_LAST_ADDR           0x0000014c 


#define   CCAM3_ATIS_IF_MONITOR_BASE_ADDR                             0x00000180 
//Mapping Bank ATIF_IF_MONITOR from base address : 0x00000180 - Prefix : CCAM3_ 
#define   CCAM3_ATIS_IF_MONITOR_CFG_ENABLE_ADDR                       0x00000180 
#define   CCAM3_ATIS_IF_MONITOR_CFG_IDLE_TIME_ADDR                    0x00000184 
#define   CCAM3_ATIS_IF_MONITOR_CFG_TIMEOUT_THR_ADDR                  0x00000188 


#define   CCAM3_EXT_TRIGGER_MONITOR_BASE_ADDR                         0x000001a0 
//Mapping Bank EXT_TRIGGER_MONITOR from base address : 0x000001a0 - Prefix : CCAM3_ 
#define   CCAM3_SYSTEM_MONITOR_EXT_TRIGGERS_ENABLE_ADDR               0x000001a0 


#define   CCAM3_SENSOR_IF_BASE_ADDR                                   0x00000200 

#define   CCAM3_SISLEY_SPI_BASE_ADDR                                  0x00000200 

#define   CCAM3_SISLEY_CTRL_BASE_ADDR                                 0x00000200 
//Mapping Bank SISLEY_SENSOR_IF_CTRL from base address : 0x00000200 - Prefix : CCAM3_ 
#define   CCAM3_SISLEY_SENSOR_CTRL_BASE_ADDRESS                       0x00000200 
#define   CCAM3_SISLEY_SENSOR_CTRL_START_ADDR                         0x00000200 
#define   CCAM3_SISLEY_SENSOR_CTRL_LAST_ADDR                          0x0000021c 
#define   CCAM3_SISLEY_SENSOR_GLOBAL_CTRL_ADDR                        0x00000200 
#define   CCAM3_SISLEY_SENSOR_ROI_CTRL_ADDR                           0x00000204 
#define   CCAM3_SISLEY_SENSOR_READOUT_CTRL_ADDR                       0x00000208 
#define   CCAM3_SISLEY_SENSOR_TESTBUS_CTRL_ADDR                       0x0000020c 
#define   CCAM3_SISLEY_SENSOR_CLKSYNC_CTRL_ADDR                       0x00000210 
#define   CCAM3_SISLEY_SENSOR_LIFO_CTRL_ADDR                          0x00000214 
#define   CCAM3_SISLEY_SENSOR_CHIP_ID_ADDR                            0x00000218 
#define   CCAM3_SISLEY_SENSOR_SPARE_CTRL_ADDR                         0x0000021c 


#define   CCAM3_SISLEY_IBGEN_BASE_ADDR                                0x00000300 
//Mapping Bank SISLEY_SENSOR_IF_IBGEN from base address : 0x00000300 - Prefix : CCAM3_ 
#define   CCAM3_SISLEY_IBGEN_BASE_ADDRESS                             0x00000300 
#define   CCAM3_SISLEY_IBGEN_START_ADDR                               0x00000300 
#define   CCAM3_SISLEY_IBGEN_LAST_ADDR                                0x00000368 
#define   CCAM3_SISLEY_IBGEN_VECTOR_0_ADDR                            0x00000300 
#define   CCAM3_SISLEY_IBGEN_VECTOR_1_ADDR                            0x00000304 
#define   CCAM3_SISLEY_IBGEN_VECTOR_2_ADDR                            0x00000308 
#define   CCAM3_SISLEY_IBGEN_VECTOR_3_ADDR                            0x0000030c 
#define   CCAM3_SISLEY_IBGEN_VECTOR_4_ADDR                            0x00000310 
#define   CCAM3_SISLEY_IBGEN_VECTOR_5_ADDR                            0x00000314 
#define   CCAM3_SISLEY_IBGEN_VECTOR_6_ADDR                            0x00000318 
#define   CCAM3_SISLEY_IBGEN_VECTOR_7_ADDR                            0x0000031c 
#define   CCAM3_SISLEY_IBGEN_VECTOR_8_ADDR                            0x00000320 
#define   CCAM3_SISLEY_IBGEN_VECTOR_9_ADDR                            0x00000324 
#define   CCAM3_SISLEY_IBGEN_VECTOR_10_ADDR                           0x00000328 
#define   CCAM3_SISLEY_IBGEN_VECTOR_11_ADDR                           0x0000032c 
#define   CCAM3_SISLEY_IBGEN_VECTOR_12_ADDR                           0x00000330 
#define   CCAM3_SISLEY_IBGEN_VECTOR_13_ADDR                           0x00000334 
#define   CCAM3_SISLEY_IBGEN_VECTOR_14_ADDR                           0x00000338 
#define   CCAM3_SISLEY_IBGEN_VECTOR_15_ADDR                           0x0000033c 
#define   CCAM3_SISLEY_IBGEN_VECTOR_16_ADDR                           0x00000340 
#define   CCAM3_SISLEY_IBGEN_VECTOR_17_ADDR                           0x00000344 
#define   CCAM3_SISLEY_IBGEN_VECTOR_18_ADDR                           0x00000348 
#define   CCAM3_SISLEY_IBGEN_VECTOR_19_ADDR                           0x0000034c 
#define   CCAM3_SISLEY_IBGEN_VECTOR_20_ADDR                           0x00000350 
#define   CCAM3_SISLEY_IBGEN_VECTOR_21_ADDR                           0x00000354 
#define   CCAM3_SISLEY_IBGEN_VECTOR_22_ADDR                           0x00000358 
#define   CCAM3_SISLEY_IBGEN_VECTOR_23_ADDR                           0x0000035c 
#define   CCAM3_SISLEY_IBGEN_VECTOR_24_ADDR                           0x00000360 
#define   CCAM3_SISLEY_IBGEN_VECTOR_25_ADDR                           0x00000364 
#define   CCAM3_SISLEY_IBGEN_VECTOR_26_ADDR                           0x00000368 


#define   CCAM3_SISLEY_ROI_BASE_ADDR                                  0x00000400 
//Mapping Bank SISLEY_SENSOR_IF_ROI from base address : 0x00000400 - Prefix : CCAM3_ 
#define   CCAM3_SISLEY_ROI_BASE_ADDRESS                               0x00000400 
#define   CCAM3_SISLEY_ROI_START_ADDR                                 0x00000400 
#define   CCAM3_SISLEY_ROI_LAST_ADDR                                  0x0000073c 
#define   CCAM3_SISLEY_ROI_TD_X_START_ADDR                            0x00000400 
#define   CCAM3_SISLEY_ROI_TD_X_LAST_ADDR                             0x00000450 
#define   CCAM3_SISLEY_ROI_TD_X_0_ADDR                                0x00000400 
#define   CCAM3_SISLEY_ROI_TD_X_1_ADDR                                0x00000404 
#define   CCAM3_SISLEY_ROI_TD_X_2_ADDR                                0x00000408 
#define   CCAM3_SISLEY_ROI_TD_X_3_ADDR                                0x0000040c 
#define   CCAM3_SISLEY_ROI_TD_X_4_ADDR                                0x00000410 
#define   CCAM3_SISLEY_ROI_TD_X_5_ADDR                                0x00000414 
#define   CCAM3_SISLEY_ROI_TD_X_6_ADDR                                0x00000418 
#define   CCAM3_SISLEY_ROI_TD_X_7_ADDR                                0x0000041c 
#define   CCAM3_SISLEY_ROI_TD_X_8_ADDR                                0x00000420 
#define   CCAM3_SISLEY_ROI_TD_X_9_ADDR                                0x00000424 
#define   CCAM3_SISLEY_ROI_TD_X_10_ADDR                               0x00000428 
#define   CCAM3_SISLEY_ROI_TD_X_11_ADDR                               0x0000042c 
#define   CCAM3_SISLEY_ROI_TD_X_12_ADDR                               0x00000430 
#define   CCAM3_SISLEY_ROI_TD_X_13_ADDR                               0x00000434 
#define   CCAM3_SISLEY_ROI_TD_X_14_ADDR                               0x00000438 
#define   CCAM3_SISLEY_ROI_TD_X_15_ADDR                               0x0000043c 
#define   CCAM3_SISLEY_ROI_TD_X_16_ADDR                               0x00000440 
#define   CCAM3_SISLEY_ROI_TD_X_17_ADDR                               0x00000444 
#define   CCAM3_SISLEY_ROI_TD_X_18_ADDR                               0x00000448 
#define   CCAM3_SISLEY_ROI_TD_X_19_ADDR                               0x0000044c 
#define   CCAM3_SISLEY_ROI_TD_X_20_ADDR                               0x00000450 
#define   CCAM3_SISLEY_ROI_TD_Y_START_ADDR                            0x00000500 
#define   CCAM3_SISLEY_ROI_TD_Y_LAST_ADDR                             0x0000053c 
#define   CCAM3_SISLEY_ROI_TD_Y_0_ADDR                                0x00000500 
#define   CCAM3_SISLEY_ROI_TD_Y_1_ADDR                                0x00000504 
#define   CCAM3_SISLEY_ROI_TD_Y_2_ADDR                                0x00000508 
#define   CCAM3_SISLEY_ROI_TD_Y_3_ADDR                                0x0000050c 
#define   CCAM3_SISLEY_ROI_TD_Y_4_ADDR                                0x00000510 
#define   CCAM3_SISLEY_ROI_TD_Y_5_ADDR                                0x00000514 
#define   CCAM3_SISLEY_ROI_TD_Y_6_ADDR                                0x00000518 
#define   CCAM3_SISLEY_ROI_TD_Y_7_ADDR                                0x0000051c 
#define   CCAM3_SISLEY_ROI_TD_Y_8_ADDR                                0x00000520 
#define   CCAM3_SISLEY_ROI_TD_Y_9_ADDR                                0x00000524 
#define   CCAM3_SISLEY_ROI_TD_Y_10_ADDR                               0x00000528 
#define   CCAM3_SISLEY_ROI_TD_Y_11_ADDR                               0x0000052c 
#define   CCAM3_SISLEY_ROI_TD_Y_12_ADDR                               0x00000530 
#define   CCAM3_SISLEY_ROI_TD_Y_13_ADDR                               0x00000534 
#define   CCAM3_SISLEY_ROI_TD_Y_14_ADDR                               0x00000538 
#define   CCAM3_SISLEY_ROI_TD_Y_15_ADDR                               0x0000053c 
#define   CCAM3_SISLEY_ROI_EM_X_START_ADDR                            0x00000600 
#define   CCAM3_SISLEY_ROI_EM_X_LAST_ADDR                             0x00000650 
#define   CCAM3_SISLEY_ROI_EM_X_0_ADDR                                0x00000600 
#define   CCAM3_SISLEY_ROI_EM_X_1_ADDR                                0x00000604 
#define   CCAM3_SISLEY_ROI_EM_X_2_ADDR                                0x00000608 
#define   CCAM3_SISLEY_ROI_EM_X_3_ADDR                                0x0000060c 
#define   CCAM3_SISLEY_ROI_EM_X_4_ADDR                                0x00000610 
#define   CCAM3_SISLEY_ROI_EM_X_5_ADDR                                0x00000614 
#define   CCAM3_SISLEY_ROI_EM_X_6_ADDR                                0x00000618 
#define   CCAM3_SISLEY_ROI_EM_X_7_ADDR                                0x0000061c 
#define   CCAM3_SISLEY_ROI_EM_X_8_ADDR                                0x00000620 
#define   CCAM3_SISLEY_ROI_EM_X_9_ADDR                                0x00000624 
#define   CCAM3_SISLEY_ROI_EM_X_10_ADDR                               0x00000628 
#define   CCAM3_SISLEY_ROI_EM_X_11_ADDR                               0x0000062c 
#define   CCAM3_SISLEY_ROI_EM_X_12_ADDR                               0x00000630 
#define   CCAM3_SISLEY_ROI_EM_X_13_ADDR                               0x00000634 
#define   CCAM3_SISLEY_ROI_EM_X_14_ADDR                               0x00000638 
#define   CCAM3_SISLEY_ROI_EM_X_15_ADDR                               0x0000063c 
#define   CCAM3_SISLEY_ROI_EM_X_16_ADDR                               0x00000640 
#define   CCAM3_SISLEY_ROI_EM_X_17_ADDR                               0x00000644 
#define   CCAM3_SISLEY_ROI_EM_X_18_ADDR                               0x00000648 
#define   CCAM3_SISLEY_ROI_EM_X_19_ADDR                               0x0000064c 
#define   CCAM3_SISLEY_ROI_EM_X_20_ADDR                               0x00000650 
#define   CCAM3_SISLEY_ROI_EM_Y_START_ADDR                            0x00000700 
#define   CCAM3_SISLEY_ROI_EM_Y_LAST_ADDR                             0x0000073c 
#define   CCAM3_SISLEY_ROI_EM_Y_0_ADDR                                0x00000700 
#define   CCAM3_SISLEY_ROI_EM_Y_1_ADDR                                0x00000704 
#define   CCAM3_SISLEY_ROI_EM_Y_2_ADDR                                0x00000708 
#define   CCAM3_SISLEY_ROI_EM_Y_3_ADDR                                0x0000070c 
#define   CCAM3_SISLEY_ROI_EM_Y_4_ADDR                                0x00000710 
#define   CCAM3_SISLEY_ROI_EM_Y_5_ADDR                                0x00000714 
#define   CCAM3_SISLEY_ROI_EM_Y_6_ADDR                                0x00000718 
#define   CCAM3_SISLEY_ROI_EM_Y_7_ADDR                                0x0000071c 
#define   CCAM3_SISLEY_ROI_EM_Y_8_ADDR                                0x00000720 
#define   CCAM3_SISLEY_ROI_EM_Y_9_ADDR                                0x00000724 
#define   CCAM3_SISLEY_ROI_EM_Y_10_ADDR                               0x00000728 
#define   CCAM3_SISLEY_ROI_EM_Y_11_ADDR                               0x0000072c 
#define   CCAM3_SISLEY_ROI_EM_Y_12_ADDR                               0x00000730 
#define   CCAM3_SISLEY_ROI_EM_Y_13_ADDR                               0x00000734 
#define   CCAM3_SISLEY_ROI_EM_Y_14_ADDR                               0x00000738 
#define   CCAM3_SISLEY_ROI_EM_Y_15_ADDR                               0x0000073c 

#define CCAM3_SISLEY_IF_BASE_ADDR                                     0x00000740
#define CCAM3_SISLEY_IF_LAST_ADDR                                     0x00000750
#define CCAM3_SISLEY_IF_SIZE                                          0x000000C0

#define CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_ADDR                     0x00000740
#define CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_ENABLE_BIT_IDX           0
#define CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_ENABLE_WIDTH             1
#define CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_ENABLE_DEFAULT           0x00000000
#define CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_TYPE_BIT_IDX             4
#define CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_TYPE_WIDTH               1
#define CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_TYPE_DEFAULT             0x00000000
#define CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_PIXEL_TYPE_BIT_IDX       8
#define CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_PIXEL_TYPE_WIDTH         1
#define CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_PIXEL_TYPE_DEFAULT       0x00000000
#define CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_PIXEL_POLARITY_BIT_IDX   12
#define CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_PIXEL_POLARITY_WIDTH     1
#define CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_PIXEL_POLARITY_DEFAULT   0x00000000

#define CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_ADDR                    0x00000744
#define CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_VALID_RATIO_BIT_IDX     0
#define CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_VALID_RATIO_WIDTH       10
#define CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_VALID_RATIO_DEFAULT     0x00000000
#define CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_LENGTH_BIT_IDX          16
#define CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_LENGTH_WIDTH            16
#define CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_LENGTH_DEFAULT          0x00000000

#define CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_ADDR                    0x00000748
#define CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_VALID_RATIO_BIT_IDX     0
#define CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_VALID_RATIO_WIDTH       10
#define CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_VALID_RATIO_DEFAULT     0x00000000
#define CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_LENGTH_BIT_IDX          16
#define CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_LENGTH_WIDTH            16
#define CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_LENGTH_DEFAULT          0x00000000


#define   STEREO_SYSTEM_CONFIG_BASE_ADDR                              0x00000800 
//Mapping Bank SYSTEM_CONFIG from base address : 0x00000800 - Prefix : CCAM3_ 
#define   CCAM3_SYSTEM_CONFIG_BASE_ADDR                               0x00000800 
#define   CCAM3_SYSTEM_CONFIG_ADDR                                    0x00000800 
#define   CCAM3_SYSTEM_CONFIG_ID_ADDR                                 0x00000800 
#define   CCAM3_SYSTEM_CONFIG_VERSION_ADDR                            0x00000804 
#define   CCAM3_SYSTEM_CONFIG_BUILD_DATE_ADDR                         0x00000808 
#define   CCAM3_SYSTEM_CONFIG_VERSION_CONTROL_ID_ADDR                 0x0000080c 
#define   CCAM3_SYSTEM_CONFIG_LAST_ADDR                               0x0000080c 


#define   STEREO_FX3_HOST_IF_BASE_ADDR                                0x00001400 
//Mapping Bank FX3_HOST_IF from base address : 0x00001400 - Prefix : CCAM3_ 
#define   CCAM3_FX3_HOST_IF_REGISTER_MAPPING_BASE_ADDR                0x00001400 
#define   CCAM3_FX3_HOST_IF_PKT_END_ENABLE_ADDR                       0x00001400 
#define   CCAM3_FX3_HOST_IF_PKT_END_INTERVAL_US_ADDR                  0x00001404 
#define   CCAM3_FX3_HOST_IF_PKT_END_DATA_COUNT_ADDR                   0x00001408 
#define   CCAM3_REG00_ADDR_BIT_ADDR                                   0x00001400 
#define   CCAM3_REG01_ADDR_BIT_ADDR                                   0x00001404 
#define   CCAM3_REG02_ADDR_BIT_ADDR                                   0x00001408 
#define   CCAM3_REG03_ADDR_BIT_ADDR                                   0x0000140c 
#define   CCAM3_REG04_ADDR_BIT_ADDR                                   0x00001410 
#define   CCAM3_REG05_ADDR_BIT_ADDR                                   0x00001414 
#define   CCAM3_REG06_ADDR_BIT_ADDR                                   0x00001418 
#define   CCAM3_REG07_ADDR_BIT_ADDR                                   0x0000141c 
#define   CCAM3_REG08_ADDR_BIT_ADDR                                   0x00001420 
#define   CCAM3_REG09_ADDR_BIT_ADDR                                   0x00001424 
#define   CCAM3_REG10_ADDR_BIT_ADDR                                   0x00001428 
#define   CCAM3_REG11_ADDR_BIT_ADDR                                   0x0000142c 
#define   CCAM3_REG12_ADDR_BIT_ADDR                                   0x00001430 
#define   CCAM3_REG13_ADDR_BIT_ADDR                                   0x00001434 
#define   CCAM3_REG14_ADDR_BIT_ADDR                                   0x00001438 
#define   CCAM3_REG15_ADDR_BIT_ADDR                                   0x0000143c 
#define   CCAM3_FX3_HOST_IF_LAST_ADDR                                 0x0000143c 

#define   CCAM4_MIPI_HOST_IF_BASE_ADDR                                0x00001500
//Mapping Bank MIPI_HOST_IF from base address : 0x00001500 - Prefix : CCAM4_
#define   CCAM4_MIPI_HOST_IF_BASE_ADDR                                0x00001500
#define   CCAM4_MIPI_HOST_IF_CONTROL_ADDR                             0x00001500
#define   CCAM4_MIPI_HOST_IF_DATA_IDENTIFIER_ADDR                     0x00001504
#define   CCAM4_MIPI_HOST_IF_FRAME_PERIOD_ADDR                        0x00001508
#define   CCAM4_MIPI_HOST_IF_PACKET_TIMEOUT_ADDR                      0x0000150c
#define   CCAM4_MIPI_HOST_IF_PACKET_SIZE_ADDR                         0x00001510
#define   CCAM4_MIPI_HOST_IF_START_TIME_ADDR                          0x00001514
#define   CCAM4_MIPI_HOST_IF_START_FRAME_TIME_ADDR                    0x00001518
#define   CCAM4_MIPI_HOST_IF_END_FRAME_TIME_ADDR                      0x0000151c
#define   CCAM4_MIPI_HOST_IF_INTER_FRAME_TIME_ADDR                    0x00001520
#define   CCAM4_MIPI_HOST_IF_INTER_PACKET_TIME_ADDR                   0x00001524
#define   CCAM4_MIPI_HOST_IF_END_ADDR                                 0x00001528

#define   CCAM4_MIPI_AVAILABILITY_ADDR                                0x00001800

// clang-format on

#endif // METAVISION_HAL_STEREO_PC_MAPPING_H
