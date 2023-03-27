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

#ifndef METAVISION_HAL_IMX636_EVK2_REGISTERMAP_H
#define METAVISION_HAL_IMX636_EVK2_REGISTERMAP_H

#include "metavision/psee_hw_layer/utils/regmap_data.h"

static RegmapElement Imx636Evk2RegisterMap[] = {
    // clang-format off

    {R, {"SYSTEM_CONTROL/GLOBAL_CONTROL", 0x0000}},
    {F, {"MODE", 0, 2, 0x0}},
    {F, {"FORMAT", 2, 2, 0x2}},
    {A, {"raw", 0x0}},
    {A, {"2.0", 0x2}},
    {A, {"3.0", 0x3}},
    {F, {"CCAM_ID", 4, 2, 0x0}},
    {F, {"OUTPUT_FORMAT", 6, 2, 0x2}},
    {A, {"raw", 0x0}},
    {A, {"2.0", 0x2}},
    {A, {"3.0", 0x3}},

    {R, {"SYSTEM_CONTROL/CLK_CONTROL", 0x0004}},
    {F, {"CORE_EN", 0, 1, 0x0}},
    {F, {"CORE_SOFT_RST", 1, 1, 0x0}},
    {F, {"CORE_REG_BANK_RST", 2, 1, 0x0}},
    {F, {"SENSOR_IF_EN", 4, 1, 0x0}},
    {F, {"SENSOR_IF_SOFT_RST", 5, 1, 0x0}},
    {F, {"SENSOR_IF_REG_BANK_RST", 6, 1, 0x0}},
    {F, {"HOST_IF_EN", 8, 1, 0x0}},
    {F, {"HOST_IF_SOFT_RST", 9, 1, 0x0}},
    {F, {"HOST_IF_REG_BANK_RST", 10, 1, 0x0}},
    {F, {"GLOBAL_RST", 16, 1, 0x0}},

    {R, {"SYSTEM_CONTROL/TIME_BASE_CONTROL", 0x0008}},
    {F, {"ENABLE", 0, 1, 0x0}},
    {F, {"EXT_SYNC_MODE", 1, 1, 0x0}},
    {F, {"EXT_SYNC_ENABLE", 2, 1, 0x0}},
    {F, {"EXT_SYNC_MASTER", 3, 1, 0x0}},
    {F, {"EXT_SYNC_MASTER_SEL", 4, 1, 0x0}},
    {F, {"ENABLE_EXT_SYNC", 5, 1, 0x0}},
    {F, {"ENABLE_CAM_SYNC", 6, 1, 0x0}},

    {R, {"SYSTEM_CONTROL/TIME_BASE_RESOLUTION", 0x000C}},
    {F, {"CLKS", 0, 16, 0x64}},

    {R, {"SYSTEM_CONTROL/IMU_CONTROL", 0x0010}},
    {F, {"ENABLE", 0, 1, 0x0}},

    {R, {"SYSTEM_CONTROL/EVT_DATA_FORMATTER_CONTROL", 0x0014}},
    {F, {"ENABLE", 0, 1, 0x0}},
    {F, {"BYPASS", 1, 1, 0x0}},

    {R, {"SYSTEM_CONTROL/EVT_MERGE_CONTROL", 0x0018}},
    {F, {"ENABLE", 0, 1, 0x0}},
    {F, {"BYPASS", 1, 1, 0x0}},
    {F, {"SOURCE", 2, 1, 0x0}},

    {R, {"SYSTEM_CONTROL/TH_RECOVERY_CONTROL", 0x0044}},
    {F, {"ENABLE", 0, 1, 0x0}},
    {F, {"BYPASS", 1, 1, 0x0}},

    {R, {"SYSTEM_CONTROL/TS_CHECKER_CONTROL", 0x0048}},
    {F, {"BYPASS", 0, 1, 0x0}},
    {F, {"THRESHOLD", 1, 24, 0x186A0}},

    {R, {"SYSTEM_CONTROL/TS_CHECKER_EVT_UI_CNT", 0x004C}},
    {F, {"TIME_LOW", 0, 16, 0x0}},
    {F, {"TIME_HIGH", 16, 16, 0x0}},

    {R, {"SYSTEM_CONTROL/TS_CHECKER_EVT_BROKEN_CNT", 0x0050}},
    {F, {"TIME_LOW", 0, 16, 0x0}},
    {F, {"TIME_HIGH", 16, 16, 0x0}},

    {R, {"SYSTEM_CONTROL/TS_CHECKER_EVT_UK_ERR_CNT", 0x0054}},
    {F, {"TIME_LOW", 0, 16, 0x0}},
    {F, {"TIME_HIGH", 16, 16, 0x0}},

    {R, {"SYSTEM_CONTROL/BOARD_CONTROL_STATUS", 0x0058}},
    {F, {"ENET_PWDN", 0, 1, 0x0}},
    {F, {"PAV_15W", 1, 1, 0x0}},
    {F, {"PAV_4P5W", 2, 1, 0x0}},
    {F, {"PAV_7P5W", 3, 1, 0x0}},
    {F, {"USB_EN_READ_PAV", 4, 1, 0x0}},
    {F, {"USB3_EN_REF_CLK", 5, 1, 0x1}},
    {F, {"USBCC_MUX_EN_N", 6, 1, 0x0}},
    {F, {"VMON_ALERT", 7, 1, 0x0}},
    {F, {"VMON_I2C_EN_LVLSHFT", 8, 1, 0x0}},
    {F, {"VERSION", 10, 2, 0x0}},
    {F, {"USB2PHY_RESETB", 12, 1, 0x1}},
    {F, {"ENET_RESET_N", 13, 1, 0x0}},
    {F, {"LDO_2V5_1V0_EN", 14, 1, 0x0}},
    {F, {"VMON_PU_TO_1V8", 15, 1, 0x0}},
    {F, {"USB_C_OUT1", 16, 1, 0x0}},
    {F, {"USB_C_OUT2", 17, 1, 0x0}},

    {R, {"SYSTEM_CONTROL/IO_CONTROL", 0x005C}},
    {F, {"SYNC_IN", 0, 1, 0x0}},
    {F, {"TRIG_IN", 1, 1, 0x0}},
    {F, {"SYNC_OUT_EN_FLT_CHK", 4, 1, 0x0}},
    {F, {"SYNC_OUT_EN_HSIDE", 5, 1, 0x0}},
    {F, {"SYNC_OUT_FAULT_ALERT", 6, 1, 0x0}},
    {F, {"SYNC_OUT", 7, 1, 0x0}},
    {F, {"SYNC_OUT_MODE", 8, 4, 0x0}},

    {R, {"SYSTEM_CONTROL/OUT_TH_RECOVERY_CONTROL", 0x0060}},
    {F, {"ENABLE", 0, 1, 0x1}},
    {F, {"BYPASS", 1, 1, 0x1}},

    {R, {"SYSTEM_CONTROL/OUT_TS_CHECKER_CONTROL", 0x0064}},
    {F, {"BYPASS", 0, 1, 0x1}},
    {F, {"THRESHOLD", 1, 24, 0x186A0}},

    {R, {"SYSTEM_CONTROL/OUT_TS_CHECKER_EVT_UI_CNT", 0x0068}},
    {F, {"TIME_LOW", 0, 16, 0x0}},
    {F, {"TIME_HIGH", 16, 16, 0x0}},

    {R, {"SYSTEM_CONTROL/OUT_TS_CHECKER_EVT_BROKEN_CNT", 0x006C}},
    {F, {"TIME_LOW", 0, 16, 0x0}},
    {F, {"TIME_HIGH", 16, 16, 0x0}},

    {R, {"SYSTEM_CONTROL/OUT_TS_CHECKER_EVT_UK_ERR_CNT", 0x0070}},
    {F, {"TIME_LOW", 0, 16, 0x0}},
    {F, {"TIME_HIGH", 16, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/TEMP", 0x0200}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VCC_INT", 0x0202}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VCC_AUX", 0x0204}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VP_VN", 0x0206}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VREFP", 0x0208}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VREFN", 0x020A}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VCC_BRAM", 0x020C}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/SUPPLY_OFFSET", 0x0210}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/OFFSET", 0x0212}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/GAIN_ERROR", 0x0214}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VAUX0", 0x0220}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VAUX1", 0x0222}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VAUX2", 0x0224}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VAUX3", 0x0226}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VAUX4", 0x0228}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VAUX5", 0x022A}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VAUX6", 0x022C}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VAUX7", 0x022E}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VAUX8", 0x0230}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VAUX9", 0x0232}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VAUX10", 0x0234}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VAUX11", 0x0236}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VAUX12", 0x0238}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VAUX13", 0x023A}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VAUX14", 0x023C}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/VAUX15", 0x023E}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/MAX_TEMP", 0x0240}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/MAX_VCC_INT", 0x0242}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/MAX_VCC_AUX", 0x0244}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/MAX_VCC_BRAM", 0x0246}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/MIN_TEMP", 0x0248}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/MIN_VCC_INT", 0x024A}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/MIN_VCC_AUX", 0x024C}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/MIN_VCC_BRAM", 0x024E}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/FLAGS", 0x027E}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/CONF_REG0", 0x0280}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/CONF_REG1", 0x0282}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/CONF_REG2", 0x0284}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/SEQ_REG0", 0x0290}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/SEQ_REG1", 0x0292}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/SEQ_REG2", 0x0294}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/SEQ_REG3", 0x0296}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/SEQ_REG4", 0x0298}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/SEQ_REG5", 0x029A}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/SEQ_REG6", 0x029C}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/SEQ_REG7", 0x029E}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/ALARM_THR_REG0", 0x02A0}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/ALARM_THR_REG1", 0x02A2}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/ALARM_THR_REG2", 0x02A4}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/ALARM_THR_REG3", 0x02A6}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/ALARM_THR_REG4", 0x02A8}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/ALARM_THR_REG5", 0x02AA}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/ALARM_THR_REG6", 0x02AC}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/ALARM_THR_REG7", 0x02AE}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/ALARM_THR_REG8", 0x02B0}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/ALARM_THR_REG9", 0x02B2}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/ALARM_THR_REG10", 0x02B4}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/ALARM_THR_REG11", 0x02B6}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/ALARM_THR_REG12", 0x02B8}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/ALARM_THR_REG13", 0x02BA}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/ALARM_THR_REG14", 0x02BC}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR_XADC/ALARM_THR_REG15", 0x02BE}},
    {F, {"VALUE", 0, 16, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR/EVT_ENABLE", 0x0300}},
    {F, {"ALL_EVT", 0, 1, 0x0}},
    {F, {"TEMP_EVT", 1, 1, 0x0}},
    {F, {"VCC_INT_EVT", 2, 1, 0x0}},
    {F, {"VCC_AUX_EVT", 3, 1, 0x0}},
    {F, {"VCC_BRAM_EVT", 4, 1, 0x0}},
    {F, {"ALL_ALARM", 8, 1, 0x0}},
    {F, {"OVER_TEMP_ALARM", 9, 1, 0x0}},
    {F, {"USER_TEMP_ALARM", 10, 1, 0x0}},
    {F, {"VCC_INT_ALARM", 11, 1, 0x0}},
    {F, {"VCC_AUX_ALARM", 12, 1, 0x0}},
    {F, {"VCC_BRAM_ALARM", 13, 1, 0x0}},
    {F, {"SYSTEM_POWER_DOWN", 16, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR/EVT_PERIOD", 0x0304}},
    {F, {"VALUE", 0, 24, 0x186A0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR/EXT_TEMP_CONTROL", 0x0308}},
    {F, {"STATUS_SYS_POWER_DOWN", 0, 1, 0x0}},
    {F, {"EXT_TEMP_MONITOR_EN", 1, 1, 0x0}},
    {F, {"EXT_TEMP_MONITOR_SPI_EN", 2, 1, 0x0}},
    {F, {"REMOTE_TEMP_MONITOR_EN", 3, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR/EVK_EXT_TEMP_VALUE", 0x030C}},
    {F, {"VALUE", 0, 22, 0x0}},

    {R, {"SYSTEM_MONITOR/TEMP_VCC_MONITOR/REMOTE_TEMP_ADDR", 0x0310}},
    {F, {"VALUE", 0, 24, 0x2030C}},

    {R, {"SYSTEM_MONITOR/ATIS_IF_MONITOR/CFG_ENABLE", 0x0340}},
    {F, {"ALL_EVT", 0, 1, 0x0}},
    {F, {"TD_IDLE_TIME_EVT", 1, 1, 0x0}},
    {F, {"TD_IDLE_TIMEOUT_EVT", 2, 1, 0x0}},
    {F, {"APS_IDLE_TIME_EVT", 3, 1, 0x0}},
    {F, {"APS_IDLE_TIMEOUT_EVT", 4, 1, 0x0}},
    {F, {"GLOBAL_ILLUMINATION_EVT", 5, 1, 0x0}},
    {F, {"EM_TRIGGER_SEQ_EVT", 6, 1, 0x0}},
    {F, {"REFRACTORY_CLOCK_EVT", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/ATIS_IF_MONITOR/CFG_IDLE_TIME_THR", 0x0344}},
    {F, {"VALUE", 0, 26, 0x2710}},

    {R, {"SYSTEM_MONITOR/ATIS_IF_MONITOR/CFG_IDLE_TIMEOUT_THR", 0x0348}},
    {F, {"VALUE", 0, 26, 0x2710}},

    {R, {"SYSTEM_MONITOR/ATIS_IF_MONITOR/STAT_GLOBAL_ILLUMINATION", 0x034C}},
    {F, {"DATA", 0, 26, 0x0}},
    {F, {"VALID", 31, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/ATIS_IF_MONITOR/STAT_REFRACTORY_CLOCK", 0x0350}},
    {F, {"DATA", 0, 24, 0x0}},

    {R, {"SYSTEM_MONITOR/EXT_TRIGGERS/ENABLE", 0x0360}},
    {F, {"TRIGGER_0", 0, 1, 0x0}},
    {F, {"TRIGGER_1", 1, 1, 0x0}},
    {F, {"TRIGGER_2", 2, 1, 0x0}},
    {F, {"TRIGGER_3", 3, 1, 0x0}},
    {F, {"TRIGGER_4", 4, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/EXT_TRIGGERS/OUT_ENABLE", 0x0364}},
    {F, {"VALUE", 0, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/EXT_TRIGGERS/OUT_PULSE_PERIOD", 0x0368}},
    {F, {"", 0, 32, 0x64}},

    {R, {"SYSTEM_MONITOR/EXT_TRIGGERS/OUT_PULSE_WIDTH", 0x036C}},
    {F, {"", 0, 32, 0x1}},

    {R, {"SYSTEM_MONITOR/EXT_TRIGGERS/OUT_REGISTER_MODE", 0x0370}},
    {F, {"ENABLE", 0, 1, 0x0}},
    {F, {"VALUE", 1, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/WHO_AM_I", 0x0400}},
    {F, {"VALUE", 0, 8, 0xEA}},

    {R, {"SYSTEM_MONITOR/IMU/GYRO_SMPLRT_DIV", 0x0400}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_MST_ODR_CONFIG", 0x0400}},
    {F, {"I2C_MST_ODR_CONFIG", 0, 4, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/GYRO_CONFIG_1", 0x0404}},
    {F, {"GYRO_FCHOICE", 0, 1, 0x1}},
    {F, {"GYRO_FS_SEL", 1, 2, 0x0}},
    {F, {"GYRO_DLPFCFG", 3, 3, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_MST_CTRL", 0x0404}},
    {F, {"I2C_MST_CLK", 0, 4, 0x0}},
    {F, {"I2C_MST_P_NSR", 4, 1, 0x0}},
    {F, {"MULT_MST_EN", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/SELF_TEST_X_GYRO", 0x0408}},
    {F, {"XG_ST_DATA", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/GYRO_CONFIG_2", 0x0408}},
    {F, {"GYRO_AVGCFG", 0, 3, 0x0}},
    {F, {"ZGYRO_CTEN", 3, 1, 0x0}},
    {F, {"YGYRO_CTEN", 4, 1, 0x0}},
    {F, {"XGYRO_CTEN", 5, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_MST_DELAY_CTRL", 0x0408}},
    {F, {"I2C_SLV0_DELAY_EN", 0, 1, 0x0}},
    {F, {"I2C_SLV1_DELAY_EN", 1, 1, 0x0}},
    {F, {"I2C_SLV2_DELAY_EN", 2, 1, 0x0}},
    {F, {"I2C_SLV3_DELAY_EN", 3, 1, 0x0}},
    {F, {"I2C_SLV4_DELAY_EN", 4, 1, 0x0}},
    {F, {"DELAY_ES_SHADOW", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/USER_CTRL", 0x040C}},
    {F, {"I2C_MST_RST", 1, 1, 0x0}},
    {F, {"SRAM_RST", 2, 1, 0x0}},
    {F, {"DMP_RST", 3, 1, 0x0}},
    {F, {"I2C_IF_DIS", 4, 1, 0x0}},
    {F, {"I2C_MST_EN", 5, 1, 0x0}},
    {F, {"FIFO_EN", 6, 1, 0x0}},
    {F, {"DMP_EN", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/SELF_TEST_Y_GYRO", 0x040C}},
    {F, {"YG_ST_DATA", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/XG_OFFS_USRH", 0x040C}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV0_ADDR", 0x040C}},
    {F, {"I2C_ID_0", 0, 7, 0x0}},
    {F, {"I2C_SLV0_RNW", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/SELF_TEST_Z_GYRO", 0x0410}},
    {F, {"ZG_ST_DATA", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/XG_OFFS_USRL", 0x0410}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV0_REG", 0x0410}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/LP_CONFIG", 0x0414}},
    {F, {"GYRO_CYCLE", 4, 1, 0x0}},
    {F, {"ACCEL_CYCLE", 5, 1, 0x0}},
    {F, {"I2C_MST_CYCLE", 6, 1, 0x1}},

    {R, {"SYSTEM_MONITOR/IMU/YG_OFFS_USRH", 0x0414}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV0_CTRL", 0x0414}},
    {F, {"I2C_SLV0_LENG", 0, 4, 0x0}},
    {F, {"I2C_SLV0_GRP", 4, 1, 0x0}},
    {F, {"I2C_SLV0_REG_DIS", 5, 1, 0x0}},
    {F, {"I2C_SLV0_BYTE_SW", 6, 1, 0x0}},
    {F, {"I2C_SLV0_EN", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/PWR_MGMT_1", 0x0418}},
    {F, {"CLKSEL", 0, 3, 0x0}},
    {F, {"TEMP_DIS", 3, 1, 0x0}},
    {F, {"LP_EN", 5, 1, 0x0}},
    {F, {"SLEEP", 6, 1, 0x0}},
    {F, {"DEVICE_RESET", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/YG_OFFS_USRL", 0x0418}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV0_DO", 0x0418}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/PWR_MGMT_2", 0x041C}},
    {F, {"DISABLE_GYRO", 0, 3, 0x6}},
    {F, {"DISABLE_ACCEL", 3, 3, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ZG_OFFS_USRH", 0x041C}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV1_ADDR", 0x041C}},
    {F, {"I2C_ID_1", 0, 7, 0x0}},
    {F, {"I2C_SLV1_RNW", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ZG_OFFS_USRL", 0x0420}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV1_REG", 0x0420}},
    {F, {"I2C_SLV1_REG", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ODR_ALIGN_EN", 0x0424}},
    {F, {"ODR_ALIGN_EN", 0, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV1_CTRL", 0x0424}},
    {F, {"I2C_SLV1_LENG", 0, 4, 0x0}},
    {F, {"I2C_SLV1_GRP", 4, 1, 0x0}},
    {F, {"I2C_SLV1_REG_DIS", 5, 1, 0x0}},
    {F, {"I2C_SLV1_BYTE_SW", 6, 1, 0x0}},
    {F, {"I2C_SLV1_EN", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV1_DO", 0x0428}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV2_ADDR", 0x042C}},
    {F, {"I2C_ID_2", 0, 7, 0x0}},
    {F, {"I2C_SLV2_RNW", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV2_REG", 0x0430}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV2_CTRL", 0x0434}},
    {F, {"I2C_SLV2_LENG", 0, 4, 0x0}},
    {F, {"I2C_SLV2_GRP", 4, 1, 0x0}},
    {F, {"I2C_SLV2_REG_DIS", 5, 1, 0x0}},
    {F, {"I2C_SLV2_BYTE_SW", 6, 1, 0x0}},
    {F, {"I2C_SLV2_EN", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/SELF_TEST_X_ACCEL", 0x0438}},
    {F, {"XA_ST_DATA", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV2_DO", 0x0438}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/INT_PIN_CFG", 0x043C}},
    {F, {"BYPASS_EN", 1, 1, 0x0}},
    {F, {"FSYNC_INT_MODE_EN", 2, 1, 0x0}},
    {F, {"ACTL_FSYNC", 3, 1, 0x0}},
    {F, {"INT_ANYRD_2CLEAR", 4, 1, 0x0}},
    {F, {"INT1_LATCH_INT_EN", 5, 1, 0x0}},
    {F, {"INT1_OPEN", 6, 1, 0x0}},
    {F, {"INT1_ACTL", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/SELF_TEST_Y_ACCEL", 0x043C}},
    {F, {"YA_ST_DATA", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV3_ADDR", 0x043C}},
    {F, {"I2C_ID_3", 0, 7, 0x0}},
    {F, {"I2C_SLV3_RNW", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/INT_ENABLE", 0x0440}},
    {F, {"I2C_MST_INT_EN", 0, 1, 0x0}},
    {F, {"DMP_INT1_EN", 1, 1, 0x0}},
    {F, {"PLL_RDY_EN", 2, 1, 0x0}},
    {F, {"WOM_INT_EN", 3, 1, 0x0}},
    {F, {"REG_WOF_EN", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/SELF_TEST_Z_ACCEL", 0x0440}},
    {F, {"ZA_ST_DATA", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ACCEL_SMPLRT_DIV_1", 0x0440}},
    {F, {"VALUE", 0, 4, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV3_REG", 0x0440}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/INT_ENABLE_1", 0x0444}},
    {F, {"RAW_DATA_0_RDY_EN", 0, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ACCEL_SMPLRT_DIV_2", 0x0444}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV3_CTRL", 0x0444}},
    {F, {"I2C_SLV3_LENG", 0, 4, 0x0}},
    {F, {"I2C_SLV3_GRP", 4, 1, 0x0}},
    {F, {"I2C_SLV3_REG_DIS", 5, 1, 0x0}},
    {F, {"I2C_SLV3_BYTE_SW", 6, 1, 0x0}},
    {F, {"I2C_SLV3_EN", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/INT_ENABLE_2", 0x0448}},
    {F, {"FIFO_OVERFLOW_EN", 0, 5, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ACCEL_INTEL_CTRL", 0x0448}},
    {F, {"ACCEL_INTEL_MODE_INT", 0, 1, 0x0}},
    {F, {"ACCEL_INTEL_EN", 1, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV3_DO", 0x0448}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/INT_ENABLE_3", 0x044C}},
    {F, {"FIFO_WM_EN", 0, 5, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ACCEL_WOM_THR", 0x044C}},
    {F, {"WOM_THRESHOLD", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV4_ADDR", 0x044C}},
    {F, {"I2C_ID_4", 0, 7, 0x0}},
    {F, {"I2C_SLV4_RNW", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/XA_OFFS_H", 0x0450}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ACCEL_CONFIG", 0x0450}},
    {F, {"ACCEL_FCHOICE", 0, 1, 0x1}},
    {F, {"ACCEL_FS_SEL", 1, 2, 0x0}},
    {F, {"ACCEL_DLPFCFG", 3, 3, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV4_REG", 0x0450}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/XA_OFFS_L", 0x0454}},
    {F, {"VALUE", 1, 7, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ACCEL_CONFIG_2", 0x0454}},
    {F, {"DEC3_CFG", 0, 2, 0x0}},
    {F, {"AZ_ST_EN_REG", 2, 1, 0x0}},
    {F, {"AY_ST_EN_REG", 3, 1, 0x0}},
    {F, {"AX_ST_EN_REG", 4, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV4_CTRL", 0x0454}},
    {F, {"I2C_SLV4_DLY", 0, 5, 0x0}},
    {F, {"I2C_SLV4_REG_DIS", 5, 1, 0x0}},
    {F, {"I2C_SLV4_BYTE_SW", 6, 1, 0x0}},
    {F, {"I2C_SLV4_EN", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV4_DO", 0x0458}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_MST_STATUS", 0x045C}},
    {F, {"I2C_SLV0_NACK", 0, 1, 0x0}},
    {F, {"I2C_SLV1_NACK", 1, 1, 0x0}},
    {F, {"I2C_SLV2_NACK", 2, 1, 0x0}},
    {F, {"I2C_SLV3_NACK", 3, 1, 0x0}},
    {F, {"I2C_SLV4_NACK", 4, 1, 0x0}},
    {F, {"I2C_LOST_ARB", 5, 1, 0x0}},
    {F, {"I2C_SLV4_DONE", 6, 1, 0x0}},
    {F, {"PASS_THROUGH", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/YA_OFFS_H", 0x045C}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/I2C_SLV4_DI", 0x045C}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/YA_OFFS_L", 0x0460}},
    {F, {"VALUE", 1, 7, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/INT_STATUS", 0x0464}},
    {F, {"I2C_MST_INT", 0, 1, 0x0}},
    {F, {"DMP_INT1", 1, 1, 0x0}},
    {F, {"PLL_RDY_INT", 2, 1, 0x0}},
    {F, {"WOM_INT", 3, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/INT_STATUS_1", 0x0468}},
    {F, {"RAW_DATA_0_RDY_INT", 0, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ZA_OFFS_H", 0x0468}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/INT_STATUS_2", 0x046C}},
    {F, {"FIFO_OVERFLOW_INT", 0, 5, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ZA_OFFS_L", 0x046C}},
    {F, {"VALUE", 1, 7, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/INT_STATUS_3", 0x0470}},
    {F, {"FIFO_WM_INT", 0, 5, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/DELAY_TIMEH", 0x04A0}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/TIMEBASE_CORRECTION_PLL", 0x04A0}},
    {F, {"TBC_PLL", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/DELAY_TIMEL", 0x04A4}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ACCEL_XOUT_H", 0x04B4}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ACCEL_XOUT_L", 0x04B8}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ACCEL_YOUT_H", 0x04BC}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ACCEL_YOUT_L", 0x04C0}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ACCEL_ZOUT_H", 0x04C4}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/ACCEL_ZOUT_L", 0x04C8}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/GYRO_XOUT_H", 0x04CC}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/GYRO_XOUT_L", 0x04D0}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/GYRO_YOUT_H", 0x04D4}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/GYRO_YOUT_L", 0x04D8}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/GYRO_ZOUT_H", 0x04DC}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/GYRO_ZOUT_L", 0x04E0}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/TEMP_OUT_H", 0x04E4}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/TEMP_OUT_L", 0x04E8}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_00", 0x04EC}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_01", 0x04F0}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_02", 0x04F4}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_03", 0x04F8}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_04", 0x04FC}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_05", 0x0500}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_06", 0x0504}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_07", 0x0508}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_08", 0x050C}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_09", 0x0510}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_10", 0x0514}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_11", 0x0518}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_12", 0x051C}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_13", 0x0520}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_14", 0x0524}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_15", 0x0528}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_16", 0x052C}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_17", 0x0530}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_18", 0x0534}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_19", 0x0538}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_20", 0x053C}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_21", 0x0540}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_22", 0x0544}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/EXT_SLV_SENS_DATA_23", 0x0548}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/FSYNC_CONFIG", 0x0548}},
    {F, {"EXT_SYNC_SET", 0, 4, 0x0}},
    {F, {"WOF_EDGE_INT", 4, 1, 0x0}},
    {F, {"WOF_DEGLITCH_EN", 5, 1, 0x0}},
    {F, {"DELAY_TIME_EN", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/TEMP_CONFIG", 0x054C}},
    {F, {"TEMP_DLPFCFG", 0, 3, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/MOD_CTRL_USR", 0x0550}},
    {F, {"REG_LP_DMP_EN", 0, 1, 0x1}},

    {R, {"SYSTEM_MONITOR/IMU/FIFO_EN_1", 0x0598}},
    {F, {"SLV_0_FIFO_EN", 0, 1, 0x0}},
    {F, {"SLV_1_FIFO_EN", 1, 1, 0x0}},
    {F, {"SLV_2_FIFO_EN", 2, 1, 0x0}},
    {F, {"SLV_3_FIFO_EN", 3, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/FIFO_EN_2", 0x059C}},
    {F, {"TEMP_FIFO_EN", 0, 1, 0x0}},
    {F, {"GYRO_X_FIFO_EN", 1, 1, 0x0}},
    {F, {"GYRO_Y_FIFO_EN", 2, 1, 0x0}},
    {F, {"GYRO_Z_FIFO_EN", 3, 1, 0x0}},
    {F, {"ACCEL_FIFO_EN", 4, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/FIFO_RST", 0x05A0}},
    {F, {"FIFO_RESET", 0, 5, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/FIFO_MODE", 0x05A4}},
    {F, {"FIFO_MODE", 0, 5, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/FIFO_COUNTH", 0x05C0}},
    {F, {"FIFO_CNT", 0, 5, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/FIFO_COUNTL", 0x05C4}},
    {F, {"FIFO_CNT", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/FIFO_R_W", 0x05C8}},
    {F, {"VALUE", 0, 8, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/DATA_RDY_STATUS", 0x05D0}},
    {F, {"RAW_DATA_RDY", 0, 4, 0x0}},
    {F, {"WOF_STATUS", 7, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/FIFO_CFG", 0x05D8}},
    {F, {"FIFO_CFG", 0, 1, 0x0}},

    {R, {"SYSTEM_MONITOR/IMU/REG_BANK_SEL", 0x05FC}},
    {F, {"USER_BANK", 4, 2, 0x0}},

    {R, {"SYSTEM_MONITOR/CONTROL/EVT_MERGE_CONTROL", 0x0600}},
    {F, {"ENABLE", 0, 1, 0x1}},
    {F, {"BYPASS", 1, 1, 0x0}},
    {F, {"SOURCE", 2, 2, 0x0}},

    {R, {"SYSTEM_CONFIG/ID", 0x0800}},
    {F, {"VALUE", 0, 8, 0x32}},

    {R, {"SYSTEM_CONFIG/VERSION", 0x0804}},
    {F, {"MICRO", 0, 8, 0x0}},
    {F, {"MINOR", 8, 8, 0x0}},
    {F, {"MAJOR", 16, 8, 0x0}},

    {R, {"SYSTEM_CONFIG/BUILD_DATE", 0x0808}},
    {F, {"VALUE", 0, 32, 0x0}},

    {R, {"SYSTEM_CONFIG/VERSION_CONTROL_ID", 0x080C}},
    {F, {"VALUE", 0, 32, 0x0}},

    {R, {"PS_HOST_IF/AXI_DMA_PACKETIZER/CONTROL", 0x2000}},
    {F, {"BYPASS", 0, 1, 0x0}},
    {F, {"ENABLE_COUNTER_PATTERN", 1, 1, 0x0}},

    {R, {"PS_HOST_IF/AXI_DMA_PACKETIZER/PACKET_LENGTH", 0x2004}},
    {F, {"VALUE", 0, 32, 0x400}},

    {R, {"PS_HOST_IF/AXIL_BRIDGE/CONTROL", 0x2100}},
    {F, {"BUS_ERROR_EN", 0, 1, 0x0}},

    {R, {"PS_HOST_IF/AXIL_BRIDGE/ERROR_STATUS", 0x2104}},
    {F, {"CODE", 0, 16, 0x0}},
    {F, {"COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/IMX636/roi_ctrl", 0x100004}},
    {F, {"roi_td_en", 1, 1, 0x0}},
    {F, {"roi_td_shadow_trigger", 5, 1, 0x0}},
    {F, {"td_roi_roni_n_en", 6, 1, 0x1}},
    {F, {"Reserved_8", 8, 1, 0x0}},
    {F, {"px_td_rstn", 10, 1, 0x0}},
    {F, {"Reserved_17_11", 11, 7, 0xA}},
    {F, {"Reserved_25", 25, 1, 0x0}},
    {F, {"Reserved_29_28", 28, 2, 0x3}},
    {F, {"Reserved_31_30", 30, 2, 0x3}},

    {R, {"SENSOR_IF/IMX636/lifo_ctrl", 0x10000C}},
    {F, {"lifo_en", 0, 1, 0x0}},
    {F, {"lifo_out_en", 1, 1, 0x0}},
    {F, {"lifo_cnt_en", 2, 1, 0x0}},
    {F, {"Reserved_31_3", 3, 29, 0x0}},

    {R, {"SENSOR_IF/IMX636/lifo_status", 0x100010}},
    {F, {"lifo_ton", 0, 29, 0x0}},
    {F, {"lifo_ton_valid", 29, 1, 0x0}},
    {F, {"Reserved_30", 30, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/Reserved_0014", 0x100014}},
    {F, {"Reserved_31_0", 0, 32, 0xA0401806}},

    {R, {"SENSOR_IF/IMX636/spare0", 0x100018}},
    {F, {"Reserved_19_0", 0, 20, 0x0}},
    {F, {"gcd_rstn", 20, 1, 0x0}},
    {F, {"Reserved_31_21", 21, 11, 0x0}},

    {R, {"SENSOR_IF/IMX636/refractory_ctrl", 0x100020}},
    {F, {"refr_counter", 0, 28, 0x0}},
    {F, {"refr_valid", 28, 1, 0x0}},
    {F, {"Reserved_29", 29, 1, 0x0}},
    {F, {"refr_cnt_en", 30, 1, 0x0}},
    {F, {"refr_en", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/roi_win_ctrl", 0x100034}},
    {F, {"roi_master_en", 0, 1, 0x0}},
    {F, {"roi_win_done", 1, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/roi_win_start_addr", 0x100038}},
    {F, {"roi_win_start_x", 0, 11, 0x0}},
    {F, {"roi_win_start_y", 16, 10, 0x0}},

    {R, {"SENSOR_IF/IMX636/roi_win_end_addr", 0x10003C}},
    {F, {"roi_win_end_x", 0, 11, 0x4FF}},
    {F, {"roi_win_end_y", 16, 10, 0x2CF}},

    {R, {"SENSOR_IF/IMX636/dig_pad2_ctrl", 0x100044}},
    {F, {"Reserved_15_0", 0, 16, 0xFCCF}},
    {F, {"pad_sync", 16, 4, 0xF}},
    {F, {"Reserved_31_20", 20, 12, 0xCCF}},

    {R, {"SENSOR_IF/IMX636/adc_control", 0x10004C}},
    {F, {"adc_en", 0, 1, 0x0}},
    {F, {"adc_clk_en", 1, 1, 0x0}},
    {F, {"adc_start", 2, 1, 0x0}},
    {F, {"Reserved_31_3", 3, 29, 0xEC8}},

    {R, {"SENSOR_IF/IMX636/adc_status", 0x100050}},
    {F, {"adc_dac_dyn", 0, 10, 0x0}},
    {F, {"Reserved_10", 10, 1, 0x0}},
    {F, {"adc_done_dyn", 11, 1, 0x0}},
    {F, {"Reserved_31_12", 12, 20, 0x0}},

    {R, {"SENSOR_IF/IMX636/adc_misc_ctrl", 0x100054}},
    {F, {"Reserved_0", 0, 1, 0x0}},
    {F, {"adc_buf_cal_en", 1, 1, 0x0}},
    {F, {"Reserved_9_2", 2, 8, 0x84}},
    {F, {"adc_rng", 10, 2, 0x0}},
    {F, {"adc_temp", 12, 1, 0x0}},
    {F, {"Reserved_14_13", 13, 2, 0x0}},

    {R, {"SENSOR_IF/IMX636/temp_ctrl", 0x10005C}},
    {F, {"temp_buf_cal_en", 0, 1, 0x0}},
    {F, {"temp_buf_en", 1, 1, 0x0}},
    {F, {"Reserved_31_2", 2, 30, 0x20}},

    {R, {"SENSOR_IF/IMX636/iph_mirr_ctrl", 0x100074}},
    {F, {"iph_mirr_en", 0, 1, 0x0}},
    {F, {"iph_mirr_amp_en", 1, 1, 0x1}},
    {F, {"Reserved_31_2", 2, 30, 0x0}},

    {R, {"SENSOR_IF/IMX636/gcd_ctrl1", 0x100078}},
    {F, {"gcd_en", 0, 1, 0x0}},
    {F, {"gcd_diffamp_en", 1, 1, 0x0}},
    {F, {"gcd_lpf_en", 2, 1, 0x0}},
    {F, {"Reserved_31_3", 3, 29, 0x8003BE9}},

    {R, {"SENSOR_IF/IMX636/reqy_qmon_ctrl", 0x100088}},
    {F, {"reqy_qmon_en", 0, 1, 0x0}},
    {F, {"reqy_qmon_rstn", 1, 1, 0x0}},
    {F, {"Reserved_3_2", 2, 2, 0x0}},
    {F, {"reqy_qmon_interrupt_en", 4, 1, 0x0}},
    {F, {"reqy_qmon_trip_ctl", 10, 10, 0x0}},
    {F, {"Reserved_31_16", 20, 12, 0x0}},

    {R, {"SENSOR_IF/IMX636/reqy_qmon_status", 0x10008C}},
    {F, {"Reserved_15_0", 0, 16, 0x0}},
    {F, {"reqy_qmon_sum_irq", 16, 10, 0x0}},
    {F, {"reqy_qmon_trip_irq", 26, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/gcd_shadow_ctrl", 0x100090}},
    {F, {"Reserved_0", 0, 1, 0x0}},
    {F, {"gcd_irq_sw_override", 1, 1, 0x0}},
    {F, {"gcd_reset_on_copy", 2, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/gcd_shadow_status", 0x100094}},
    {F, {"gcd_shadow_valid", 0, 1, 0x0}},
    {F, {"Reserved_31_1", 1, 31, 0x0}},

    {R, {"SENSOR_IF/IMX636/gcd_shadow_counter", 0x100098}},
    {F, {"gcd_shadow_cnt_off", 0, 16, 0x0}},
    {F, {"gcd_shadow_cnt_on", 16, 16, 0x0}},

    {R, {"SENSOR_IF/IMX636/stop_sequence_control", 0x1000C8}},
    {F, {"stop_sequence_start", 0, 1, 0x0}},
    {F, {"Reserved_15_8", 8, 8, 0x1}},

    {R, {"SENSOR_IF/IMX636/bias/bias_fo", 0x101004}},
    {F, {"idac_ctl", 0, 8, 0x0}},
    {F, {"Reserved_27_8", 8, 20, 0x3A1E8}},
    {F, {"single_transfer", 28, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/bias/bias_hpf", 0x10100C}},
    {F, {"idac_ctl", 0, 8, 0x0}},
    {F, {"Reserved_27_8", 8, 20, 0x3A1FF}},
    {F, {"single_transfer", 28, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/bias/bias_diff_on", 0x101010}},
    {F, {"idac_ctl", 0, 8, 0x0}},
    {F, {"Reserved_27_8", 8, 20, 0x1A163}},
    {F, {"single_transfer", 28, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/bias/bias_diff", 0x101014}},
    {F, {"idac_ctl", 0, 8, 0x4D}},
    {F, {"Reserved_27_8", 8, 20, 0x1A150}},
    {F, {"single_transfer", 28, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/bias/bias_diff_off", 0x101018}},
    {F, {"idac_ctl", 0, 8, 0x0}},
    {F, {"Reserved_27_8", 8, 20, 0x1A137}},
    {F, {"single_transfer", 28, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/bias/bias_refr", 0x101020}},
    {F, {"idac_ctl", 0, 8, 0x14}},
    {F, {"Reserved_27_8", 8, 20, 0x38296}},
    {F, {"single_transfer", 28, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/bias/bgen_ctrl", 0x101100}},
    {F, {"burst_transfer", 0, 1, 0x0}},
    {F, {"Reserved_2_1", 1, 2, 0x0}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x00", 0x102000}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x01", 0x102004}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x02", 0x102008}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x03", 0x10200C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x04", 0x102010}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x05", 0x102014}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x06", 0x102018}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x07", 0x10201C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x08", 0x102020}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x09", 0x102024}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x10", 0x102028}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x11", 0x10202C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x12", 0x102030}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x13", 0x102034}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x14", 0x102038}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x15", 0x10203C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x16", 0x102040}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x17", 0x102044}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x18", 0x102048}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x19", 0x10204C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x20", 0x102050}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x21", 0x102054}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x22", 0x102058}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x23", 0x10205C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x24", 0x102060}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x25", 0x102064}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x26", 0x102068}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x27", 0x10206C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x28", 0x102070}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x29", 0x102074}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x30", 0x102078}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x31", 0x10207C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x32", 0x102080}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x33", 0x102084}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x34", 0x102088}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x35", 0x10208C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x36", 0x102090}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x37", 0x102094}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x38", 0x102098}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_x39", 0x10209C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y00", 0x104000}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y01", 0x104004}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y02", 0x104008}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y03", 0x10400C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y04", 0x104010}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y05", 0x104014}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y06", 0x104018}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y07", 0x10401C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y08", 0x104020}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y09", 0x104024}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y10", 0x104028}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y11", 0x10402C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y12", 0x104030}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y13", 0x104034}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y14", 0x104038}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y15", 0x10403C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y16", 0x104040}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y17", 0x104044}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y18", 0x104048}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y19", 0x10404C}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y20", 0x104050}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y21", 0x104054}},
    {F, {"effective", 0, 32, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFFFFFF}},

    {R, {"SENSOR_IF/IMX636/roi/td_roi_y22", 0x104058}},
    {F, {"effective", 0, 16, 0x0}},
    {A, {"enable", 0x0}},
    {A, {"disable", 0xFFFF}},
    {F, {"Reserved_16", 16, 1, 0x1}},
    {F, {"Reserved_17", 17, 1, 0x1}},
    {F, {"Reserved_19_18", 18, 2, 0x3}},
    {F, {"Reserved_21_20", 20, 2, 0x3}},
    {F, {"Reserved_22", 22, 1, 0x1}},
    {F, {"Reserved_23", 23, 1, 0x1}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6000", 0x106000}},
    {F, {"Reserved_1_0", 0, 2, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/in_drop_rate_control", 0x106004}},
    {F, {"cfg_event_delay_fifo_en", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"Reserved_10_2", 2, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/reference_period", 0x106008}},
    {F, {"erc_reference_period", 0, 10, 0x80}},

    {R, {"SENSOR_IF/IMX636/erc/td_target_event_rate", 0x10600C}},
    {F, {"target_event_rate", 0, 22, 0x80}},

    {R, {"SENSOR_IF/IMX636/erc/erc_enable", 0x106028}},
    {F, {"erc_en", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"Reserved_2", 2, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_602C", 0x10602C}},
    {F, {"Reserved_0", 0, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_dropping_control", 0x106050}},
    {F, {"t_dropping_en", 0, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/h_dropping_control", 0x106060}},
    {F, {"h_dropping_en", 0, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/v_dropping_control", 0x106070}},
    {F, {"v_dropping_en", 0, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/h_drop_lut_00", 0x106080}},
    {F, {"hlut00", 0, 5, 0x0}},
    {F, {"hlut01", 8, 5, 0x0}},
    {F, {"hlut02", 16, 5, 0x0}},
    {F, {"hlut03", 24, 5, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/h_drop_lut_01", 0x106084}},
    {F, {"hlut04", 0, 5, 0x0}},
    {F, {"hlut05", 8, 5, 0x0}},
    {F, {"hlut06", 16, 5, 0x0}},
    {F, {"hlut07", 24, 5, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/h_drop_lut_02", 0x106088}},
    {F, {"hlut08", 0, 5, 0x0}},
    {F, {"hlut09", 8, 5, 0x0}},
    {F, {"hlut10", 16, 5, 0x0}},
    {F, {"hlut11", 24, 5, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/h_drop_lut_03", 0x10608C}},
    {F, {"hlut12", 0, 5, 0x0}},
    {F, {"hlut13", 8, 5, 0x0}},
    {F, {"hlut14", 16, 5, 0x0}},
    {F, {"hlut15", 24, 5, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/h_drop_lut_04", 0x106090}},
    {F, {"hlut16", 0, 5, 0x0}},
    {F, {"hlut17", 8, 5, 0x0}},
    {F, {"hlut18", 16, 5, 0x0}},
    {F, {"hlut19", 24, 5, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/h_drop_lut_05", 0x106094}},
    {F, {"hlut20", 0, 5, 0x0}},
    {F, {"hlut21", 8, 5, 0x0}},
    {F, {"hlut22", 16, 5, 0x0}},
    {F, {"hlut23", 24, 5, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/h_drop_lut_06", 0x106098}},
    {F, {"hlut24", 0, 5, 0x0}},
    {F, {"hlut25", 8, 5, 0x0}},
    {F, {"hlut26", 16, 5, 0x0}},
    {F, {"hlut27", 24, 5, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/h_drop_lut_07", 0x10609C}},
    {F, {"hlut28", 0, 5, 0x0}},
    {F, {"hlut29", 8, 5, 0x0}},
    {F, {"hlut30", 16, 5, 0x0}},
    {F, {"hlut31", 24, 5, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/v_drop_lut_00", 0x1060C0}},
    {F, {"vlut00", 0, 5, 0x0}},
    {F, {"vlut01", 8, 5, 0x0}},
    {F, {"vlut02", 16, 5, 0x0}},
    {F, {"vlut03", 24, 5, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/v_drop_lut_01", 0x1060C4}},
    {F, {"vlut04", 0, 5, 0x0}},
    {F, {"vlut05", 8, 5, 0x0}},
    {F, {"vlut06", 16, 5, 0x0}},
    {F, {"vlut07", 24, 5, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/v_drop_lut_02", 0x1060C8}},
    {F, {"vlut08", 0, 5, 0x0}},
    {F, {"vlut09", 8, 5, 0x0}},
    {F, {"vlut10", 16, 5, 0x0}},
    {F, {"vlut11", 24, 5, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/v_drop_lut_03", 0x1060CC}},
    {F, {"vlut12", 0, 5, 0x0}},
    {F, {"vlut13", 8, 5, 0x0}},
    {F, {"vlut14", 16, 5, 0x0}},
    {F, {"vlut15", 24, 5, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/v_drop_lut_04", 0x1060D0}},
    {F, {"vlut16", 0, 5, 0x0}},
    {F, {"vlut17", 8, 5, 0x0}},
    {F, {"vlut18", 16, 5, 0x0}},
    {F, {"vlut19", 24, 5, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/v_drop_lut_05", 0x1060D4}},
    {F, {"vlut20", 0, 5, 0x0}},
    {F, {"vlut21", 8, 5, 0x0}},
    {F, {"vlut22", 16, 5, 0x0}},
    {F, {"vlut23", 24, 5, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/v_drop_lut_06", 0x1060D8}},
    {F, {"vlut24", 0, 5, 0x0}},
    {F, {"vlut25", 8, 5, 0x0}},
    {F, {"vlut26", 16, 5, 0x0}},
    {F, {"vlut27", 24, 5, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/v_drop_lut_07", 0x1060DC}},
    {F, {"vlut28", 0, 5, 0x0}},
    {F, {"vlut29", 8, 5, 0x0}},
    {F, {"vlut30", 16, 5, 0x0}},
    {F, {"vlut31", 24, 5, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_00", 0x106400}},
    {F, {"tlut000", 0, 9, 0x0}},
    {F, {"tlut001", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_01", 0x106404}},
    {F, {"tlut002", 0, 9, 0x0}},
    {F, {"tlut003", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_02", 0x106408}},
    {F, {"tlut004", 0, 9, 0x0}},
    {F, {"tlut005", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_03", 0x10640C}},
    {F, {"tlut006", 0, 9, 0x0}},
    {F, {"tlut007", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_04", 0x106410}},
    {F, {"tlut008", 0, 9, 0x0}},
    {F, {"tlut009", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_05", 0x106414}},
    {F, {"tlut010", 0, 9, 0x0}},
    {F, {"tlut011", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_06", 0x106418}},
    {F, {"tlut012", 0, 9, 0x0}},
    {F, {"tlut013", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_07", 0x10641C}},
    {F, {"tlut014", 0, 9, 0x0}},
    {F, {"tlut015", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_08", 0x106420}},
    {F, {"tlut016", 0, 9, 0x0}},
    {F, {"tlut017", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_09", 0x106424}},
    {F, {"tlut018", 0, 9, 0x0}},
    {F, {"tlut019", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_10", 0x106428}},
    {F, {"tlut020", 0, 9, 0x0}},
    {F, {"tlut021", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_11", 0x10642C}},
    {F, {"tlut022", 0, 9, 0x0}},
    {F, {"tlut023", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_12", 0x106430}},
    {F, {"tlut024", 0, 9, 0x0}},
    {F, {"tlut025", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_13", 0x106434}},
    {F, {"tlut026", 0, 9, 0x0}},
    {F, {"tlut027", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_14", 0x106438}},
    {F, {"tlut028", 0, 9, 0x0}},
    {F, {"tlut029", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_15", 0x10643C}},
    {F, {"tlut030", 0, 9, 0x0}},
    {F, {"tlut031", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_16", 0x106440}},
    {F, {"tlut032", 0, 9, 0x0}},
    {F, {"tlut033", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_17", 0x106444}},
    {F, {"tlut034", 0, 9, 0x0}},
    {F, {"tlut035", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_18", 0x106448}},
    {F, {"tlut036", 0, 9, 0x0}},
    {F, {"tlut037", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_19", 0x10644C}},
    {F, {"tlut038", 0, 9, 0x0}},
    {F, {"tlut039", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_20", 0x106450}},
    {F, {"tlut040", 0, 9, 0x0}},
    {F, {"tlut041", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_21", 0x106454}},
    {F, {"tlut042", 0, 9, 0x0}},
    {F, {"tlut043", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_22", 0x106458}},
    {F, {"tlut044", 0, 9, 0x0}},
    {F, {"tlut045", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_23", 0x10645C}},
    {F, {"tlut046", 0, 9, 0x0}},
    {F, {"tlut047", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_24", 0x106460}},
    {F, {"tlut048", 0, 9, 0x0}},
    {F, {"tlut049", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_25", 0x106464}},
    {F, {"tlut050", 0, 9, 0x0}},
    {F, {"tlut051", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_26", 0x106468}},
    {F, {"tlut052", 0, 9, 0x0}},
    {F, {"tlut053", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_27", 0x10646C}},
    {F, {"tlut054", 0, 9, 0x0}},
    {F, {"tlut055", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_28", 0x106470}},
    {F, {"tlut056", 0, 9, 0x0}},
    {F, {"tlut057", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_29", 0x106474}},
    {F, {"tlut058", 0, 9, 0x0}},
    {F, {"tlut059", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_30", 0x106478}},
    {F, {"tlut060", 0, 9, 0x0}},
    {F, {"tlut061", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_31", 0x10647C}},
    {F, {"tlut062", 0, 9, 0x0}},
    {F, {"tlut063", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_32", 0x106480}},
    {F, {"tlut064", 0, 9, 0x0}},
    {F, {"tlut065", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_33", 0x106484}},
    {F, {"tlut066", 0, 9, 0x0}},
    {F, {"tlut067", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_34", 0x106488}},
    {F, {"tlut068", 0, 9, 0x0}},
    {F, {"tlut069", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_35", 0x10648C}},
    {F, {"tlut070", 0, 9, 0x0}},
    {F, {"tlut071", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_36", 0x106490}},
    {F, {"tlut072", 0, 9, 0x0}},
    {F, {"tlut073", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_37", 0x106494}},
    {F, {"tlut074", 0, 9, 0x0}},
    {F, {"tlut075", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_38", 0x106498}},
    {F, {"tlut076", 0, 9, 0x0}},
    {F, {"tlut077", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_39", 0x10649C}},
    {F, {"tlut078", 0, 9, 0x0}},
    {F, {"tlut079", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_40", 0x1064A0}},
    {F, {"tlut080", 0, 9, 0x0}},
    {F, {"tlut081", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_41", 0x1064A4}},
    {F, {"tlut082", 0, 9, 0x0}},
    {F, {"tlut083", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_42", 0x1064A8}},
    {F, {"tlut084", 0, 9, 0x0}},
    {F, {"tlut085", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_43", 0x1064AC}},
    {F, {"tlut086", 0, 9, 0x0}},
    {F, {"tlut087", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_44", 0x1064B0}},
    {F, {"tlut088", 0, 9, 0x0}},
    {F, {"tlut089", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_45", 0x1064B4}},
    {F, {"tlut090", 0, 9, 0x0}},
    {F, {"tlut091", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_46", 0x1064B8}},
    {F, {"tlut092", 0, 9, 0x0}},
    {F, {"tlut093", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_47", 0x1064BC}},
    {F, {"tlut094", 0, 9, 0x0}},
    {F, {"tlut095", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_48", 0x1064C0}},
    {F, {"tlut096", 0, 9, 0x0}},
    {F, {"tlut097", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_49", 0x1064C4}},
    {F, {"tlut098", 0, 9, 0x0}},
    {F, {"tlut099", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_50", 0x1064C8}},
    {F, {"tlut100", 0, 9, 0x0}},
    {F, {"tlut101", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_51", 0x1064CC}},
    {F, {"tlut102", 0, 9, 0x0}},
    {F, {"tlut103", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_52", 0x1064D0}},
    {F, {"tlut104", 0, 9, 0x0}},
    {F, {"tlut105", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_53", 0x1064D4}},
    {F, {"tlut106", 0, 9, 0x0}},
    {F, {"tlut107", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_54", 0x1064D8}},
    {F, {"tlut108", 0, 9, 0x0}},
    {F, {"tlut109", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_55", 0x1064DC}},
    {F, {"tlut110", 0, 9, 0x0}},
    {F, {"tlut111", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_56", 0x1064E0}},
    {F, {"tlut112", 0, 9, 0x0}},
    {F, {"tlut113", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_57", 0x1064E4}},
    {F, {"tlut114", 0, 9, 0x0}},
    {F, {"tlut115", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_58", 0x1064E8}},
    {F, {"tlut116", 0, 9, 0x0}},
    {F, {"tlut117", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_59", 0x1064EC}},
    {F, {"tlut118", 0, 9, 0x0}},
    {F, {"tlut119", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_60", 0x1064F0}},
    {F, {"tlut120", 0, 9, 0x0}},
    {F, {"tlut121", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_61", 0x1064F4}},
    {F, {"tlut122", 0, 9, 0x0}},
    {F, {"tlut123", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_62", 0x1064F8}},
    {F, {"tlut124", 0, 9, 0x0}},
    {F, {"tlut125", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_63", 0x1064FC}},
    {F, {"tlut126", 0, 9, 0x0}},
    {F, {"tlut127", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_64", 0x106500}},
    {F, {"tlut128", 0, 9, 0x0}},
    {F, {"tlut129", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_65", 0x106504}},
    {F, {"tlut130", 0, 9, 0x0}},
    {F, {"tlut131", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_66", 0x106508}},
    {F, {"tlut132", 0, 9, 0x0}},
    {F, {"tlut133", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_67", 0x10650C}},
    {F, {"tlut134", 0, 9, 0x0}},
    {F, {"tlut135", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_68", 0x106510}},
    {F, {"tlut136", 0, 9, 0x0}},
    {F, {"tlut137", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_69", 0x106514}},
    {F, {"tlut138", 0, 9, 0x0}},
    {F, {"tlut139", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_70", 0x106518}},
    {F, {"tlut140", 0, 9, 0x0}},
    {F, {"tlut141", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_71", 0x10651C}},
    {F, {"tlut142", 0, 9, 0x0}},
    {F, {"tlut143", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_72", 0x106520}},
    {F, {"tlut144", 0, 9, 0x0}},
    {F, {"tlut145", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_73", 0x106524}},
    {F, {"tlut146", 0, 9, 0x0}},
    {F, {"tlut147", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_74", 0x106528}},
    {F, {"tlut148", 0, 9, 0x0}},
    {F, {"tlut149", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_75", 0x10652C}},
    {F, {"tlut150", 0, 9, 0x0}},
    {F, {"tlut151", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_76", 0x106530}},
    {F, {"tlut152", 0, 9, 0x0}},
    {F, {"tlut153", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_77", 0x106534}},
    {F, {"tlut154", 0, 9, 0x0}},
    {F, {"tlut155", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_78", 0x106538}},
    {F, {"tlut156", 0, 9, 0x0}},
    {F, {"tlut157", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_79", 0x10653C}},
    {F, {"tlut158", 0, 9, 0x0}},
    {F, {"tlut159", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_80", 0x106540}},
    {F, {"tlut160", 0, 9, 0x0}},
    {F, {"tlut161", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_81", 0x106544}},
    {F, {"tlut162", 0, 9, 0x0}},
    {F, {"tlut163", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_82", 0x106548}},
    {F, {"tlut164", 0, 9, 0x0}},
    {F, {"tlut165", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_83", 0x10654C}},
    {F, {"tlut166", 0, 9, 0x0}},
    {F, {"tlut167", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_84", 0x106550}},
    {F, {"tlut168", 0, 9, 0x0}},
    {F, {"tlut169", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_85", 0x106554}},
    {F, {"tlut170", 0, 9, 0x0}},
    {F, {"tlut171", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_86", 0x106558}},
    {F, {"tlut172", 0, 9, 0x0}},
    {F, {"tlut173", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_87", 0x10655C}},
    {F, {"tlut174", 0, 9, 0x0}},
    {F, {"tlut175", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_88", 0x106560}},
    {F, {"tlut176", 0, 9, 0x0}},
    {F, {"tlut177", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_89", 0x106564}},
    {F, {"tlut178", 0, 9, 0x0}},
    {F, {"tlut179", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_90", 0x106568}},
    {F, {"tlut180", 0, 9, 0x0}},
    {F, {"tlut181", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_91", 0x10656C}},
    {F, {"tlut182", 0, 9, 0x0}},
    {F, {"tlut183", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_92", 0x106570}},
    {F, {"tlut184", 0, 9, 0x0}},
    {F, {"tlut185", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_93", 0x106574}},
    {F, {"tlut186", 0, 9, 0x0}},
    {F, {"tlut187", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_94", 0x106578}},
    {F, {"tlut188", 0, 9, 0x0}},
    {F, {"tlut189", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_95", 0x10657C}},
    {F, {"tlut190", 0, 9, 0x0}},
    {F, {"tlut191", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_96", 0x106580}},
    {F, {"tlut192", 0, 9, 0x0}},
    {F, {"tlut193", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_97", 0x106584}},
    {F, {"tlut194", 0, 9, 0x0}},
    {F, {"tlut195", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_98", 0x106588}},
    {F, {"tlut196", 0, 9, 0x0}},
    {F, {"tlut197", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_99", 0x10658C}},
    {F, {"tlut198", 0, 9, 0x0}},
    {F, {"tlut199", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_100", 0x106590}},
    {F, {"tlut200", 0, 9, 0x0}},
    {F, {"tlut201", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_101", 0x106594}},
    {F, {"tlut202", 0, 9, 0x0}},
    {F, {"tlut203", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_102", 0x106598}},
    {F, {"tlut204", 0, 9, 0x0}},
    {F, {"tlut205", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_103", 0x10659C}},
    {F, {"tlut206", 0, 9, 0x0}},
    {F, {"tlut207", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_104", 0x1065A0}},
    {F, {"tlut208", 0, 9, 0x0}},
    {F, {"tlut209", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_105", 0x1065A4}},
    {F, {"tlut210", 0, 9, 0x0}},
    {F, {"tlut211", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_106", 0x1065A8}},
    {F, {"tlut212", 0, 9, 0x0}},
    {F, {"tlut213", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_107", 0x1065AC}},
    {F, {"tlut214", 0, 9, 0x0}},
    {F, {"tlut215", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_108", 0x1065B0}},
    {F, {"tlut216", 0, 9, 0x0}},
    {F, {"tlut217", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_109", 0x1065B4}},
    {F, {"tlut218", 0, 9, 0x0}},
    {F, {"tlut219", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_110", 0x1065B8}},
    {F, {"tlut220", 0, 9, 0x0}},
    {F, {"tlut221", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_111", 0x1065BC}},
    {F, {"tlut222", 0, 9, 0x0}},
    {F, {"tlut223", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_112", 0x1065C0}},
    {F, {"tlut224", 0, 9, 0x0}},
    {F, {"tlut225", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_113", 0x1065C4}},
    {F, {"tlut226", 0, 9, 0x0}},
    {F, {"tlut227", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_114", 0x1065C8}},
    {F, {"tlut228", 0, 9, 0x0}},
    {F, {"tlut229", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_115", 0x1065CC}},
    {F, {"tlut230", 0, 9, 0x0}},
    {F, {"tlut231", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_116", 0x1065D0}},
    {F, {"tlut232", 0, 9, 0x0}},
    {F, {"tlut233", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_117", 0x1065D4}},
    {F, {"tlut234", 0, 9, 0x0}},
    {F, {"tlut235", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_118", 0x1065D8}},
    {F, {"tlut236", 0, 9, 0x0}},
    {F, {"tlut237", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_119", 0x1065DC}},
    {F, {"tlut238", 0, 9, 0x0}},
    {F, {"tlut239", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_120", 0x1065E0}},
    {F, {"tlut240", 0, 9, 0x0}},
    {F, {"tlut241", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_121", 0x1065E4}},
    {F, {"tlut242", 0, 9, 0x0}},
    {F, {"tlut243", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_122", 0x1065E8}},
    {F, {"tlut244", 0, 9, 0x0}},
    {F, {"tlut245", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_123", 0x1065EC}},
    {F, {"tlut246", 0, 9, 0x0}},
    {F, {"tlut247", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_124", 0x1065F0}},
    {F, {"tlut248", 0, 9, 0x0}},
    {F, {"tlut249", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_125", 0x1065F4}},
    {F, {"tlut250", 0, 9, 0x0}},
    {F, {"tlut251", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_126", 0x1065F8}},
    {F, {"tlut252", 0, 9, 0x0}},
    {F, {"tlut253", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_127", 0x1065FC}},
    {F, {"tlut254", 0, 9, 0x0}},
    {F, {"tlut255", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_128", 0x106600}},
    {F, {"tlut256", 0, 9, 0x0}},
    {F, {"tlut257", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_129", 0x106604}},
    {F, {"tlut258", 0, 9, 0x0}},
    {F, {"tlut259", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_130", 0x106608}},
    {F, {"tlut260", 0, 9, 0x0}},
    {F, {"tlut261", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_131", 0x10660C}},
    {F, {"tlut262", 0, 9, 0x0}},
    {F, {"tlut263", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_132", 0x106610}},
    {F, {"tlut264", 0, 9, 0x0}},
    {F, {"tlut265", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_133", 0x106614}},
    {F, {"tlut266", 0, 9, 0x0}},
    {F, {"tlut267", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_134", 0x106618}},
    {F, {"tlut268", 0, 9, 0x0}},
    {F, {"tlut269", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_135", 0x10661C}},
    {F, {"tlut270", 0, 9, 0x0}},
    {F, {"tlut271", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_136", 0x106620}},
    {F, {"tlut272", 0, 9, 0x0}},
    {F, {"tlut273", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_137", 0x106624}},
    {F, {"tlut274", 0, 9, 0x0}},
    {F, {"tlut275", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_138", 0x106628}},
    {F, {"tlut276", 0, 9, 0x0}},
    {F, {"tlut277", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_139", 0x10662C}},
    {F, {"tlut278", 0, 9, 0x0}},
    {F, {"tlut279", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_140", 0x106630}},
    {F, {"tlut280", 0, 9, 0x0}},
    {F, {"tlut281", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_141", 0x106634}},
    {F, {"tlut282", 0, 9, 0x0}},
    {F, {"tlut283", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_142", 0x106638}},
    {F, {"tlut284", 0, 9, 0x0}},
    {F, {"tlut285", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_143", 0x10663C}},
    {F, {"tlut286", 0, 9, 0x0}},
    {F, {"tlut287", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_144", 0x106640}},
    {F, {"tlut288", 0, 9, 0x0}},
    {F, {"tlut289", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_145", 0x106644}},
    {F, {"tlut290", 0, 9, 0x0}},
    {F, {"tlut291", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_146", 0x106648}},
    {F, {"tlut292", 0, 9, 0x0}},
    {F, {"tlut293", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_147", 0x10664C}},
    {F, {"tlut294", 0, 9, 0x0}},
    {F, {"tlut295", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_148", 0x106650}},
    {F, {"tlut296", 0, 9, 0x0}},
    {F, {"tlut297", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_149", 0x106654}},
    {F, {"tlut298", 0, 9, 0x0}},
    {F, {"tlut299", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_150", 0x106658}},
    {F, {"tlut300", 0, 9, 0x0}},
    {F, {"tlut301", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_151", 0x10665C}},
    {F, {"tlut302", 0, 9, 0x0}},
    {F, {"tlut303", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_152", 0x106660}},
    {F, {"tlut304", 0, 9, 0x0}},
    {F, {"tlut305", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_153", 0x106664}},
    {F, {"tlut306", 0, 9, 0x0}},
    {F, {"tlut307", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_154", 0x106668}},
    {F, {"tlut308", 0, 9, 0x0}},
    {F, {"tlut309", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_155", 0x10666C}},
    {F, {"tlut310", 0, 9, 0x0}},
    {F, {"tlut311", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_156", 0x106670}},
    {F, {"tlut312", 0, 9, 0x0}},
    {F, {"tlut313", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_157", 0x106674}},
    {F, {"tlut314", 0, 9, 0x0}},
    {F, {"tlut315", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_158", 0x106678}},
    {F, {"tlut316", 0, 9, 0x0}},
    {F, {"tlut317", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_159", 0x10667C}},
    {F, {"tlut318", 0, 9, 0x0}},
    {F, {"tlut319", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_160", 0x106680}},
    {F, {"tlut320", 0, 9, 0x0}},
    {F, {"tlut321", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_161", 0x106684}},
    {F, {"tlut322", 0, 9, 0x0}},
    {F, {"tlut323", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_162", 0x106688}},
    {F, {"tlut324", 0, 9, 0x0}},
    {F, {"tlut325", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_163", 0x10668C}},
    {F, {"tlut326", 0, 9, 0x0}},
    {F, {"tlut327", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_164", 0x106690}},
    {F, {"tlut328", 0, 9, 0x0}},
    {F, {"tlut329", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_165", 0x106694}},
    {F, {"tlut330", 0, 9, 0x0}},
    {F, {"tlut331", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_166", 0x106698}},
    {F, {"tlut332", 0, 9, 0x0}},
    {F, {"tlut333", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_167", 0x10669C}},
    {F, {"tlut334", 0, 9, 0x0}},
    {F, {"tlut335", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_168", 0x1066A0}},
    {F, {"tlut336", 0, 9, 0x0}},
    {F, {"tlut337", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_169", 0x1066A4}},
    {F, {"tlut338", 0, 9, 0x0}},
    {F, {"tlut339", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_170", 0x1066A8}},
    {F, {"tlut340", 0, 9, 0x0}},
    {F, {"tlut341", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_171", 0x1066AC}},
    {F, {"tlut342", 0, 9, 0x0}},
    {F, {"tlut343", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_172", 0x1066B0}},
    {F, {"tlut344", 0, 9, 0x0}},
    {F, {"tlut345", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_173", 0x1066B4}},
    {F, {"tlut346", 0, 9, 0x0}},
    {F, {"tlut347", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_174", 0x1066B8}},
    {F, {"tlut348", 0, 9, 0x0}},
    {F, {"tlut349", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_175", 0x1066BC}},
    {F, {"tlut350", 0, 9, 0x0}},
    {F, {"tlut351", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_176", 0x1066C0}},
    {F, {"tlut352", 0, 9, 0x0}},
    {F, {"tlut353", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_177", 0x1066C4}},
    {F, {"tlut354", 0, 9, 0x0}},
    {F, {"tlut355", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_178", 0x1066C8}},
    {F, {"tlut356", 0, 9, 0x0}},
    {F, {"tlut357", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_179", 0x1066CC}},
    {F, {"tlut358", 0, 9, 0x0}},
    {F, {"tlut359", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_180", 0x1066D0}},
    {F, {"tlut360", 0, 9, 0x0}},
    {F, {"tlut361", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_181", 0x1066D4}},
    {F, {"tlut362", 0, 9, 0x0}},
    {F, {"tlut363", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_182", 0x1066D8}},
    {F, {"tlut364", 0, 9, 0x0}},
    {F, {"tlut365", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_183", 0x1066DC}},
    {F, {"tlut366", 0, 9, 0x0}},
    {F, {"tlut367", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_184", 0x1066E0}},
    {F, {"tlut368", 0, 9, 0x0}},
    {F, {"tlut369", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_185", 0x1066E4}},
    {F, {"tlut370", 0, 9, 0x0}},
    {F, {"tlut371", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_186", 0x1066E8}},
    {F, {"tlut372", 0, 9, 0x0}},
    {F, {"tlut373", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_187", 0x1066EC}},
    {F, {"tlut374", 0, 9, 0x0}},
    {F, {"tlut375", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_188", 0x1066F0}},
    {F, {"tlut376", 0, 9, 0x0}},
    {F, {"tlut377", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_189", 0x1066F4}},
    {F, {"tlut378", 0, 9, 0x0}},
    {F, {"tlut379", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_190", 0x1066F8}},
    {F, {"tlut380", 0, 9, 0x0}},
    {F, {"tlut381", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_191", 0x1066FC}},
    {F, {"tlut382", 0, 9, 0x0}},
    {F, {"tlut383", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_192", 0x106700}},
    {F, {"tlut384", 0, 9, 0x0}},
    {F, {"tlut385", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_193", 0x106704}},
    {F, {"tlut386", 0, 9, 0x0}},
    {F, {"tlut387", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_194", 0x106708}},
    {F, {"tlut388", 0, 9, 0x0}},
    {F, {"tlut389", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_195", 0x10670C}},
    {F, {"tlut390", 0, 9, 0x0}},
    {F, {"tlut391", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_196", 0x106710}},
    {F, {"tlut392", 0, 9, 0x0}},
    {F, {"tlut393", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_197", 0x106714}},
    {F, {"tlut394", 0, 9, 0x0}},
    {F, {"tlut395", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_198", 0x106718}},
    {F, {"tlut396", 0, 9, 0x0}},
    {F, {"tlut397", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_199", 0x10671C}},
    {F, {"tlut398", 0, 9, 0x0}},
    {F, {"tlut399", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_200", 0x106720}},
    {F, {"tlut400", 0, 9, 0x0}},
    {F, {"tlut401", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_201", 0x106724}},
    {F, {"tlut402", 0, 9, 0x0}},
    {F, {"tlut403", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_202", 0x106728}},
    {F, {"tlut404", 0, 9, 0x0}},
    {F, {"tlut405", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_203", 0x10672C}},
    {F, {"tlut406", 0, 9, 0x0}},
    {F, {"tlut407", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_204", 0x106730}},
    {F, {"tlut408", 0, 9, 0x0}},
    {F, {"tlut409", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_205", 0x106734}},
    {F, {"tlut410", 0, 9, 0x0}},
    {F, {"tlut411", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_206", 0x106738}},
    {F, {"tlut412", 0, 9, 0x0}},
    {F, {"tlut413", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_207", 0x10673C}},
    {F, {"tlut414", 0, 9, 0x0}},
    {F, {"tlut415", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_208", 0x106740}},
    {F, {"tlut416", 0, 9, 0x0}},
    {F, {"tlut417", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_209", 0x106744}},
    {F, {"tlut418", 0, 9, 0x0}},
    {F, {"tlut419", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_210", 0x106748}},
    {F, {"tlut420", 0, 9, 0x0}},
    {F, {"tlut421", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_211", 0x10674C}},
    {F, {"tlut422", 0, 9, 0x0}},
    {F, {"tlut423", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_212", 0x106750}},
    {F, {"tlut424", 0, 9, 0x0}},
    {F, {"tlut425", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_213", 0x106754}},
    {F, {"tlut426", 0, 9, 0x0}},
    {F, {"tlut427", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_214", 0x106758}},
    {F, {"tlut428", 0, 9, 0x0}},
    {F, {"tlut429", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_215", 0x10675C}},
    {F, {"tlut430", 0, 9, 0x0}},
    {F, {"tlut431", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_216", 0x106760}},
    {F, {"tlut432", 0, 9, 0x0}},
    {F, {"tlut433", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_217", 0x106764}},
    {F, {"tlut434", 0, 9, 0x0}},
    {F, {"tlut435", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_218", 0x106768}},
    {F, {"tlut436", 0, 9, 0x0}},
    {F, {"tlut437", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_219", 0x10676C}},
    {F, {"tlut438", 0, 9, 0x0}},
    {F, {"tlut439", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_220", 0x106770}},
    {F, {"tlut440", 0, 9, 0x0}},
    {F, {"tlut441", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_221", 0x106774}},
    {F, {"tlut442", 0, 9, 0x0}},
    {F, {"tlut443", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_222", 0x106778}},
    {F, {"tlut444", 0, 9, 0x0}},
    {F, {"tlut445", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_223", 0x10677C}},
    {F, {"tlut446", 0, 9, 0x0}},
    {F, {"tlut447", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_224", 0x106780}},
    {F, {"tlut448", 0, 9, 0x0}},
    {F, {"tlut449", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_225", 0x106784}},
    {F, {"tlut450", 0, 9, 0x0}},
    {F, {"tlut451", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_226", 0x106788}},
    {F, {"tlut452", 0, 9, 0x0}},
    {F, {"tlut453", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_227", 0x10678C}},
    {F, {"tlut454", 0, 9, 0x0}},
    {F, {"tlut455", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_228", 0x106790}},
    {F, {"tlut456", 0, 9, 0x0}},
    {F, {"tlut457", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_229", 0x106794}},
    {F, {"tlut458", 0, 9, 0x0}},
    {F, {"tlut459", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_230", 0x106798}},
    {F, {"tlut460", 0, 9, 0x0}},
    {F, {"tlut461", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_231", 0x10679C}},
    {F, {"tlut462", 0, 9, 0x0}},
    {F, {"tlut463", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_232", 0x1067A0}},
    {F, {"tlut464", 0, 9, 0x0}},
    {F, {"tlut465", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_233", 0x1067A4}},
    {F, {"tlut466", 0, 9, 0x0}},
    {F, {"tlut467", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_234", 0x1067A8}},
    {F, {"tlut468", 0, 9, 0x0}},
    {F, {"tlut469", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_235", 0x1067AC}},
    {F, {"tlut470", 0, 9, 0x0}},
    {F, {"tlut471", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_236", 0x1067B0}},
    {F, {"tlut472", 0, 9, 0x0}},
    {F, {"tlut473", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_237", 0x1067B4}},
    {F, {"tlut474", 0, 9, 0x0}},
    {F, {"tlut475", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_238", 0x1067B8}},
    {F, {"tlut476", 0, 9, 0x0}},
    {F, {"tlut477", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_239", 0x1067BC}},
    {F, {"tlut478", 0, 9, 0x0}},
    {F, {"tlut479", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_240", 0x1067C0}},
    {F, {"tlut480", 0, 9, 0x0}},
    {F, {"tlut481", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_241", 0x1067C4}},
    {F, {"tlut482", 0, 9, 0x0}},
    {F, {"tlut483", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_242", 0x1067C8}},
    {F, {"tlut484", 0, 9, 0x0}},
    {F, {"tlut485", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_243", 0x1067CC}},
    {F, {"tlut486", 0, 9, 0x0}},
    {F, {"tlut487", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_244", 0x1067D0}},
    {F, {"tlut488", 0, 9, 0x0}},
    {F, {"tlut489", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_245", 0x1067D4}},
    {F, {"tlut490", 0, 9, 0x0}},
    {F, {"tlut491", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_246", 0x1067D8}},
    {F, {"tlut492", 0, 9, 0x0}},
    {F, {"tlut493", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_247", 0x1067DC}},
    {F, {"tlut494", 0, 9, 0x0}},
    {F, {"tlut495", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_248", 0x1067E0}},
    {F, {"tlut496", 0, 9, 0x0}},
    {F, {"tlut497", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_249", 0x1067E4}},
    {F, {"tlut498", 0, 9, 0x0}},
    {F, {"tlut499", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_250", 0x1067E8}},
    {F, {"tlut500", 0, 9, 0x0}},
    {F, {"tlut501", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_251", 0x1067EC}},
    {F, {"tlut502", 0, 9, 0x0}},
    {F, {"tlut503", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_252", 0x1067F0}},
    {F, {"tlut504", 0, 9, 0x0}},
    {F, {"tlut505", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_253", 0x1067F4}},
    {F, {"tlut506", 0, 9, 0x0}},
    {F, {"tlut507", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_254", 0x1067F8}},
    {F, {"tlut508", 0, 9, 0x0}},
    {F, {"tlut509", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/t_drop_lut_255", 0x1067FC}},
    {F, {"tlut510", 0, 9, 0x0}},
    {F, {"tlut511", 16, 9, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6800", 0x106800}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6804", 0x106804}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6808", 0x106808}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_680C", 0x10680C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6810", 0x106810}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6814", 0x106814}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6818", 0x106818}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_681C", 0x10681C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6820", 0x106820}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6824", 0x106824}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6828", 0x106828}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_682C", 0x10682C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6830", 0x106830}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6834", 0x106834}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6838", 0x106838}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_683C", 0x10683C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6840", 0x106840}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6844", 0x106844}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6848", 0x106848}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_684C", 0x10684C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6850", 0x106850}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6854", 0x106854}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6858", 0x106858}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_685C", 0x10685C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6860", 0x106860}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6864", 0x106864}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6868", 0x106868}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_686C", 0x10686C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6870", 0x106870}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6874", 0x106874}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6878", 0x106878}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_687C", 0x10687C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6880", 0x106880}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6884", 0x106884}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6888", 0x106888}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_688C", 0x10688C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6890", 0x106890}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6894", 0x106894}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6898", 0x106898}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_689C", 0x10689C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68A0", 0x1068A0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68A4", 0x1068A4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68A8", 0x1068A8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68AC", 0x1068AC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68B0", 0x1068B0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68B4", 0x1068B4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68B8", 0x1068B8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68BC", 0x1068BC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68C0", 0x1068C0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68C4", 0x1068C4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68C8", 0x1068C8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68CC", 0x1068CC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68D0", 0x1068D0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68D4", 0x1068D4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68D8", 0x1068D8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68DC", 0x1068DC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68E0", 0x1068E0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68E4", 0x1068E4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68E8", 0x1068E8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68EC", 0x1068EC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68F0", 0x1068F0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68F4", 0x1068F4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68F8", 0x1068F8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_68FC", 0x1068FC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6900", 0x106900}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6904", 0x106904}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6908", 0x106908}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_690C", 0x10690C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6910", 0x106910}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6914", 0x106914}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6918", 0x106918}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_691C", 0x10691C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6920", 0x106920}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6924", 0x106924}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6928", 0x106928}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_692C", 0x10692C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6930", 0x106930}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6934", 0x106934}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6938", 0x106938}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_693C", 0x10693C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6940", 0x106940}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6944", 0x106944}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6948", 0x106948}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_694C", 0x10694C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6950", 0x106950}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6954", 0x106954}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6958", 0x106958}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_695C", 0x10695C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6960", 0x106960}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6964", 0x106964}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6968", 0x106968}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_696C", 0x10696C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6970", 0x106970}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6974", 0x106974}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6978", 0x106978}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_697C", 0x10697C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6980", 0x106980}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6984", 0x106984}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6988", 0x106988}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_698C", 0x10698C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6990", 0x106990}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6994", 0x106994}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6998", 0x106998}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_699C", 0x10699C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69A0", 0x1069A0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69A4", 0x1069A4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69A8", 0x1069A8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69AC", 0x1069AC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69B0", 0x1069B0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69B4", 0x1069B4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69B8", 0x1069B8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69BC", 0x1069BC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69C0", 0x1069C0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69C4", 0x1069C4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69C8", 0x1069C8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69CC", 0x1069CC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69D0", 0x1069D0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69D4", 0x1069D4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69D8", 0x1069D8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69DC", 0x1069DC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69E0", 0x1069E0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69E4", 0x1069E4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69E8", 0x1069E8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69EC", 0x1069EC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69F0", 0x1069F0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69F4", 0x1069F4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69F8", 0x1069F8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_69FC", 0x1069FC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A00", 0x106A00}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A04", 0x106A04}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A08", 0x106A08}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A0C", 0x106A0C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A10", 0x106A10}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A14", 0x106A14}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A18", 0x106A18}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A1C", 0x106A1C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A20", 0x106A20}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A24", 0x106A24}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A28", 0x106A28}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A2C", 0x106A2C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A30", 0x106A30}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A34", 0x106A34}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A38", 0x106A38}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A3C", 0x106A3C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A40", 0x106A40}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A44", 0x106A44}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A48", 0x106A48}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A4C", 0x106A4C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A50", 0x106A50}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A54", 0x106A54}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A58", 0x106A58}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A5C", 0x106A5C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A60", 0x106A60}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A64", 0x106A64}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A68", 0x106A68}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A6C", 0x106A6C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A70", 0x106A70}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A74", 0x106A74}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A78", 0x106A78}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A7C", 0x106A7C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A80", 0x106A80}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A84", 0x106A84}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A88", 0x106A88}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A8C", 0x106A8C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A90", 0x106A90}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A94", 0x106A94}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A98", 0x106A98}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6A9C", 0x106A9C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AA0", 0x106AA0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AA4", 0x106AA4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AA8", 0x106AA8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AAC", 0x106AAC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AB0", 0x106AB0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AB4", 0x106AB4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AB8", 0x106AB8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6ABC", 0x106ABC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AC0", 0x106AC0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AC4", 0x106AC4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AC8", 0x106AC8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6ACC", 0x106ACC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AD0", 0x106AD0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AD4", 0x106AD4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AD8", 0x106AD8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6ADC", 0x106ADC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AE0", 0x106AE0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AE4", 0x106AE4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AE8", 0x106AE8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AEC", 0x106AEC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AF0", 0x106AF0}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AF4", 0x106AF4}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AF8", 0x106AF8}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6AFC", 0x106AFC}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B00", 0x106B00}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B04", 0x106B04}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B08", 0x106B08}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B0C", 0x106B0C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B10", 0x106B10}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B14", 0x106B14}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B18", 0x106B18}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B1C", 0x106B1C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B20", 0x106B20}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B24", 0x106B24}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B28", 0x106B28}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B2C", 0x106B2C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B30", 0x106B30}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B34", 0x106B34}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B38", 0x106B38}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B3C", 0x106B3C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B40", 0x106B40}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B44", 0x106B44}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B48", 0x106B48}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B4C", 0x106B4C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B50", 0x106B50}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B54", 0x106B54}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B58", 0x106B58}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B5C", 0x106B5C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B60", 0x106B60}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B64", 0x106B64}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B68", 0x106B68}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B6C", 0x106B6C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B70", 0x106B70}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B74", 0x106B74}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B78", 0x106B78}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B7C", 0x106B7C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B80", 0x106B80}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B84", 0x106B84}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B88", 0x106B88}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B8C", 0x106B8C}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B90", 0x106B90}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/erc/Reserved_6B94", 0x106B94}},
    {F, {"Reserved_5_0", 0, 6, 0x0}},
    {F, {"Reserved_13_8", 8, 6, 0x0}},
    {F, {"Reserved_21_16", 16, 6, 0x0}},
    {F, {"Reserved_29_24", 24, 6, 0x0}},

    {R, {"SENSOR_IF/IMX636/edf/pipeline_control", 0x107000}},
    {F, {"Reserved_0", 0, 1, 0x1}},
    {F, {"format", 1, 1, 0x0}},
    {F, {"Reserved_2", 2, 1, 0x0}},
    {F, {"Reserved_3", 3, 1, 0x0}},
    {F, {"Reserved_4", 4, 1, 0x0}},
    {F, {"Reserved_31_16", 16, 16, 0xFFFF}},

    {R, {"SENSOR_IF/IMX636/edf/Reserved_7004", 0x107004}},
    {F, {"Reserved_10", 10, 1, 0x1}},

    {R, {"SENSOR_IF/IMX636/eoi/Reserved_8000", 0x108000}},
    {F, {"Reserved_7_6", 6, 2, 0x2}},

    {R, {"SENSOR_IF/IMX636/ro/readout_ctrl", 0x109000}},
    {F, {"Reserved_0", 0, 1, 0x0}},
    {F, {"ro_td_self_test_en", 1, 1, 0x0}},
    {F, {"Reserved_3", 3, 1, 0x1}},
    {F, {"Reserved_4", 4, 1, 0x0}},
    {F, {"ro_inv_pol_td", 5, 1, 0x0}},
    {F, {"Reserved_7_6", 6, 2, 0x0}},
    {F, {"Reserved_31_8", 8, 24, 0x2}},

    {R, {"SENSOR_IF/IMX636/ro/ro_fsm_ctrl", 0x109004}},
    {F, {"readout_wait", 0, 16, 0x1E}},
    {F, {"Reserved_31_16", 16, 16, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/time_base_ctrl", 0x109008}},
    {F, {"time_base_enable", 0, 1, 0x0}},
    {F, {"time_base_mode", 1, 1, 0x0}},
    {F, {"external_mode", 2, 1, 0x0}},
    {F, {"external_mode_enable", 3, 1, 0x0}},
    {F, {"Reserved_10_4", 4, 7, 0x64}},

    {R, {"SENSOR_IF/IMX636/ro/dig_ctrl", 0x10900C}},
    {F, {"dig_crop_enable", 0, 3, 0x0}},
    {F, {"dig_crop_reset_orig", 4, 1, 0x0}},
    {F, {"Reserved_31_5", 5, 27, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/dig_start_pos", 0x109010}},
    {F, {"dig_crop_start_x", 0, 11, 0x0}},
    {F, {"dig_crop_start_y", 16, 10, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/dig_end_pos", 0x109014}},
    {F, {"dig_crop_end_x", 0, 11, 0x0}},
    {F, {"dig_crop_end_y", 16, 10, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/ro_ctrl", 0x109028}},
    {F, {"area_cnt_en", 0, 1, 0x0}},
    {F, {"output_disable", 1, 1, 0x0}},
    {F, {"keep_th", 2, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_x0_addr", 0x10902C}},
    {F, {"x0_addr", 0, 11, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_x1_addr", 0x109030}},
    {F, {"x1_addr", 0, 11, 0x140}},

    {R, {"SENSOR_IF/IMX636/ro/area_x2_addr", 0x109034}},
    {F, {"x2_addr", 0, 11, 0x280}},

    {R, {"SENSOR_IF/IMX636/ro/area_x3_addr", 0x109038}},
    {F, {"x3_addr", 0, 11, 0x3C0}},

    {R, {"SENSOR_IF/IMX636/ro/area_x4_addr", 0x10903C}},
    {F, {"x4_addr", 0, 11, 0x500}},

    {R, {"SENSOR_IF/IMX636/ro/area_y0_addr", 0x109040}},
    {F, {"y0_addr", 0, 11, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_y1_addr", 0x109044}},
    {F, {"y1_addr", 0, 11, 0xB4}},

    {R, {"SENSOR_IF/IMX636/ro/area_y2_addr", 0x109048}},
    {F, {"y2_addr", 0, 11, 0x168}},

    {R, {"SENSOR_IF/IMX636/ro/area_y3_addr", 0x10904C}},
    {F, {"y3_addr", 0, 11, 0x21C}},

    {R, {"SENSOR_IF/IMX636/ro/area_y4_addr", 0x109050}},
    {F, {"y4_addr", 0, 11, 0x2D0}},

    {R, {"SENSOR_IF/IMX636/ro/counter_ctrl", 0x109054}},
    {F, {"count_en", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"Reserved_2", 2, 1, 0x1}},

    {R, {"SENSOR_IF/IMX636/ro/counter_timer_threshold", 0x109058}},
    {F, {"timer_threshold", 0, 32, 0x3E8}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_00", 0x109100}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_01", 0x109104}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_02", 0x109108}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_03", 0x10910C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_04", 0x109110}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_05", 0x109114}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_06", 0x109118}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_07", 0x10911C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_08", 0x109120}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_09", 0x109124}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_10", 0x109128}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_11", 0x10912C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_12", 0x109130}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_13", 0x109134}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_14", 0x109138}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_15", 0x10913C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_16", 0x109140}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_17", 0x109144}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_18", 0x109148}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_19", 0x10914C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_20", 0x109150}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_21", 0x109154}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_22", 0x109158}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_23", 0x10915C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_24", 0x109160}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_25", 0x109164}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_26", 0x109168}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_27", 0x10916C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_28", 0x109170}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_29", 0x109174}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_30", 0x109178}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_31", 0x10917C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_32", 0x109180}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_33", 0x109184}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_34", 0x109188}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_35", 0x10918C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_36", 0x109190}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_37", 0x109194}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_38", 0x109198}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_39", 0x10919C}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_40", 0x1091A0}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_41", 0x1091A4}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_42", 0x1091A8}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_43", 0x1091AC}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_44", 0x1091B0}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_45", 0x1091B4}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_46", 0x1091B8}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_47", 0x1091BC}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_48", 0x1091C0}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_49", 0x1091C4}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_50", 0x1091C8}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_51", 0x1091CC}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_52", 0x1091D0}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_53", 0x1091D4}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_54", 0x1091D8}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_55", 0x1091DC}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_56", 0x1091E0}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_57", 0x1091E4}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_58", 0x1091E8}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_59", 0x1091EC}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_60", 0x1091F0}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_61", 0x1091F4}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_62", 0x1091F8}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/digital_mask_pixel_63", 0x1091FC}},
    {F, {"x", 0, 11, 0x0}},
    {F, {"y", 16, 11, 0x0}},
    {F, {"valid", 31, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_cnt00", 0x109200}},
    {F, {"area_cnt_val_00", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_cnt01", 0x109204}},
    {F, {"area_cnt_val_01", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_cnt02", 0x109208}},
    {F, {"area_cnt_val_02", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_cnt03", 0x10920C}},
    {F, {"area_cnt_val_03", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_cnt04", 0x109210}},
    {F, {"area_cnt_val_04", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_cnt05", 0x109214}},
    {F, {"area_cnt_val_05", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_cnt06", 0x109218}},
    {F, {"area_cnt_val_06", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_cnt07", 0x10921C}},
    {F, {"area_cnt_val_07", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_cnt08", 0x109220}},
    {F, {"area_cnt_val_08", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_cnt09", 0x109224}},
    {F, {"area_cnt_val_09", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_cnt10", 0x109228}},
    {F, {"area_cnt_val_10", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_cnt11", 0x10922C}},
    {F, {"area_cnt_val_11", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_cnt12", 0x109230}},
    {F, {"area_cnt_val_12", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_cnt13", 0x109234}},
    {F, {"area_cnt_val_13", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_cnt14", 0x109238}},
    {F, {"area_cnt_val_14", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/area_cnt15", 0x10923C}},
    {F, {"area_cnt_val_15", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/ro/evt_vector_cnt_val", 0x109244}},
    {F, {"evt_vector_cnt_val", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/mipi_csi/mipi_control", 0x10B000}},
    {F, {"mipi_csi_enable", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"Reserved_2", 2, 1, 0x0}},
    {F, {"mipi_data_lane1", 3, 1, 0x1}},
    {F, {"mipi_data_lane2", 4, 1, 0x1}},
    {F, {"mipi_packet_timeout_enable", 5, 1, 0x0}},
    {F, {"line_blanking_clk_disable", 6, 1, 0x1}},
    {F, {"Reserved_7", 7, 1, 0x0}},
    {F, {"line_blanking_en", 8, 1, 0x1}},
    {F, {"frame_blanking_en", 9, 1, 0x0}},
    {F, {"Reserved_31_10", 10, 22, 0x0}},

    {R, {"SENSOR_IF/IMX636/mipi_csi/mipi_packet_size", 0x10B020}},
    {F, {"mipi_packet_size", 0, 15, 0x2000}},

    {R, {"SENSOR_IF/IMX636/mipi_csi/mipi_packet_timeout", 0x10B024}},
    {F, {"mipi_packet_timeout", 0, 16, 0x40}},

    {R, {"SENSOR_IF/IMX636/mipi_csi/mipi_frame_period", 0x10B028}},
    {F, {"mipi_frame_period", 4, 12, 0x7D}},

    {R, {"SENSOR_IF/IMX636/mipi_csi/mipi_line_blanking", 0x10B02C}},
    {F, {"mipi_line_blanking", 0, 8, 0xA}},

    {R, {"SENSOR_IF/IMX636/mipi_csi/mipi_frame_blanking", 0x10B030}},
    {F, {"mipi_frame_blanking", 0, 16, 0x0}},

    {R, {"SENSOR_IF/IMX636/afk/pipeline_control", 0x10C000}},
    {F, {"Reserved_0", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"afk_bypass", 2, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/afk/param", 0x10C004}},
    {F, {"counter_low", 0, 3, 0x4}},
    {F, {"counter_high", 3, 3, 0x6}},
    {F, {"invert", 6, 1, 0x0}},
    {F, {"drop_disable", 7, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/afk/filter_period", 0x10C008}},
    {F, {"min_cutoff_period", 0, 8, 0xF}},
    {F, {"max_cutoff_period", 8, 8, 0x9C}},
    {F, {"inverted_duty_cycle", 16, 4, 0x8}},

    {R, {"SENSOR_IF/IMX636/afk/invalidation", 0x10C0C0}},
    {F, {"dt_fifo_wait_time", 0, 12, 0x5A0}},
    {F, {"Reserved_23_12", 12, 12, 0x5A}},
    {F, {"Reserved_27_24", 24, 4, 0xA}},
    {F, {"Reserved_28", 28, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/afk/initialization", 0x10C0C4}},
    {F, {"afk_req_init", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"afk_flag_init_done", 2, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/afk/shadow_ctrl", 0x10C0D4}},
    {F, {"timer_en", 0, 1, 0x0}},
    {F, {"Reserved_31_1", 1, 31, 0x2}},

    {R, {"SENSOR_IF/IMX636/afk/shadow_timer_threshold", 0x10C0D8}},
    {F, {"timer_threshold", 0, 32, 0x3E8}},

    {R, {"SENSOR_IF/IMX636/afk/shadow_status", 0x10C0DC}},
    {F, {"shadow_valid", 0, 1, 0x0}},
    {F, {"shadow_overrun", 1, 1, 0x0}},
    {F, {"Reserved_31_2", 2, 30, 0x0}},

    {R, {"SENSOR_IF/IMX636/afk/total_evt_count", 0x10C0E0}},
    {F, {"total_evt_count", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/afk/flicker_evt_count", 0x10C0E4}},
    {F, {"flicker_evt_count", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/afk/vector_evt_count", 0x10C0E8}},
    {F, {"vector_evt_count", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/stc/pipeline_control", 0x10D000}},
    {F, {"Reserved_0", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"stc_trail_bypass", 2, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/stc/stc_param", 0x10D004}},
    {F, {"stc_enable", 0, 1, 0x0}},
    {F, {"stc_threshold", 1, 19, 0x2710}},
    {F, {"disable_stc_cut_trail", 24, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/stc/trail_param", 0x10D008}},
    {F, {"trail_enable", 0, 1, 0x0}},
    {F, {"trail_threshold", 1, 19, 0x186A0}},

    {R, {"SENSOR_IF/IMX636/stc/timestamping", 0x10D00C}},
    {F, {"prescaler", 0, 5, 0xD}},
    {F, {"multiplier", 5, 4, 0x1}},
    {F, {"Reserved_9", 9, 1, 0x1}},
    {F, {"enable_last_ts_update_at_every_event", 16, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/stc/invalidation", 0x10D0C0}},
    {F, {"dt_fifo_wait_time", 0, 12, 0x4}},
    {F, {"dt_fifo_timeout", 12, 12, 0x118}},
    {F, {"Reserved_27_24", 24, 4, 0xA}},
    {F, {"Reserved_28", 28, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/stc/initialization", 0x10D0C4}},
    {F, {"stc_req_init", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"stc_flag_init_done", 2, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/stc/shadow_ctrl", 0x10D0D4}},
    {F, {"timer_en", 0, 1, 0x0}},
    {F, {"Reserved_31_1", 1, 31, 0x2}},

    {R, {"SENSOR_IF/IMX636/stc/shadow_timer_threshold", 0x10D0D8}},
    {F, {"timer_threshold", 0, 32, 0x3E8}},

    {R, {"SENSOR_IF/IMX636/stc/shadow_status", 0x10D0DC}},
    {F, {"shadow_valid", 0, 1, 0x0}},
    {F, {"shadow_overrun", 1, 1, 0x0}},
    {F, {"Reserved_31_2", 2, 30, 0x0}},

    {R, {"SENSOR_IF/IMX636/stc/total_evt_count", 0x10D0E0}},
    {F, {"total_evt_count", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/stc/stc_evt_count", 0x10D0E4}},
    {F, {"stc_evt_count", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/stc/trail_evt_count", 0x10D0E8}},
    {F, {"trail_evt_count", 0, 32, 0x0}},

    {R, {"IMX636/stc/output_vector_count", 0x10D0EC}},
    {F, {"output_vector_count", 0, 32, 0x0}},

    {R, {"SENSOR_IF/IMX636/slvs/slvs_control", 0x10E000}},
    {F, {"slvs_llp_enable", 0, 1, 0x0}},
    {F, {"Reserved_1", 1, 1, 0x0}},
    {F, {"Reserved_2", 2, 1, 0x1}},
    {F, {"slvs_packet_timeout_enable", 5, 1, 0x0}},
    {F, {"slvs_line_blanking_en", 8, 1, 0x1}},
    {F, {"slvs_frame_blanking_en", 9, 1, 0x0}},

    {R, {"SENSOR_IF/IMX636/slvs/slvs_packet_size", 0x10E020}},
    {F, {"slvs_packet_size", 0, 14, 0x1000}},

    {R, {"SENSOR_IF/IMX636/slvs/slvs_packet_timeout", 0x10E024}},
    {F, {"slvs_packet_timeout", 0, 16, 0x40}},

    {R, {"SENSOR_IF/IMX636/slvs/slvs_line_blanking", 0x10E02C}},
    {F, {"slvs_line_blanking", 0, 8, 0xA}},

    {R, {"SENSOR_IF/IMX636/slvs/slvs_frame_blanking", 0x10E030}},
    {F, {"slvs_frame_blanking", 0, 16, 0x0}},

    {R, {"SENSOR_IF/IMX636/slvs/slvs_phy_logic_ctrl_00", 0x10E150}},
    {F, {"oportsel", 0, 2, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/CORE_CONFIG", 0x700000}},
    {F, {"CORE_ENABLE", 0, 1, 0x1}},
    {F, {"SOFT_RESET", 1, 1, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/PROTOCOL_CONFIG", 0x700004}},
    {F, {"ACTIVE_LANES", 0, 2, 0x1}},
    {F, {"MAX_LANES", 3, 2, 0x1}},

    {R, {"SENSOR_IF/MIPI_RX/CORE_STAT", 0x700010}},
    {F, {"SOFT_RESET", 0, 1, 0x0}},
    {F, {"STREAM_LINE_BUFFER_FULL", 1, 1, 0x0}},
    {F, {"SHORT_PACKET_FIFO_NOT_EMPTY", 2, 1, 0x0}},
    {F, {"SHORT_PACKET_FIFO_FULL", 3, 1, 0x0}},
    {F, {"PACKET_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/GLOBAL_IRQ_ENABLE", 0x700020}},
    {F, {"", 0, 1, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/IRQ_STAT", 0x700024}},
    {F, {"VC0_ERR_FRAME_DATA", 0, 1, 0x0}},
    {F, {"VC0_ERR_FRAME_SYNC", 1, 1, 0x0}},
    {F, {"VC1_ERR_FRAME_DATA", 2, 1, 0x0}},
    {F, {"VC1_ERR_FRAME_SYNC", 3, 1, 0x0}},
    {F, {"VC2_ERR_FRAME_DATA", 4, 1, 0x0}},
    {F, {"VC2_ERR_FRAME_SYNC", 5, 1, 0x0}},
    {F, {"VC3_ERR_FRAME_DATA", 6, 1, 0x0}},
    {F, {"VC3_ERR_FRAME_SYNC", 7, 1, 0x0}},
    {F, {"ERR_ID", 8, 1, 0x0}},
    {F, {"ERR_CRC", 9, 1, 0x0}},
    {F, {"ERR_ECC_CORRECTED", 10, 1, 0x0}},
    {F, {"ERR_ECC_DOUBLE", 11, 1, 0x0}},
    {F, {"ERR_SOT_SYNC_HS", 12, 1, 0x0}},
    {F, {"ERR_SOT_HS", 13, 1, 0x0}},
    {F, {"STOP_STATE", 17, 1, 0x0}},
    {F, {"ST_LINE_BUF_FULL", 18, 1, 0x0}},
    {F, {"SHT_PKT_FIFO_N_EMPTY", 19, 1, 0x0}},
    {F, {"SHT_PKT_FIFO_FULL", 20, 1, 0x0}},
    {F, {"LANE_CONFIG_ERR", 21, 1, 0x0}},
    {F, {"WC_CORRUPTION", 22, 1, 0x0}},
    {F, {"RX_SKEWCALHS", 29, 1, 0x0}},
    {F, {"VCX FRAME ERROR", 30, 1, 0x0}},
    {F, {"FRAME_RECEIVED", 31, 1, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/IRQ_ENABLE", 0x700028}},
    {F, {"VC0_ERR_FRAME_DATA", 0, 1, 0x0}},
    {F, {"VC0_ERR_FRAME_SYNC", 1, 1, 0x0}},
    {F, {"VC1_ERR_FRAME_DATA", 2, 1, 0x0}},
    {F, {"VC1_ERR_FRAME_SYNC", 3, 1, 0x0}},
    {F, {"VC2_ERR_FRAME_DATA", 4, 1, 0x0}},
    {F, {"VC2_ERR_FRAME_SYNC", 5, 1, 0x0}},
    {F, {"VC3_ERR_FRAME_DATA", 6, 1, 0x0}},
    {F, {"VC3_ERR_FRAME_SYNC", 7, 1, 0x0}},
    {F, {"ERR_ID", 8, 1, 0x0}},
    {F, {"ERR_CRC", 9, 1, 0x0}},
    {F, {"ERR_ECC_CORRECTED", 10, 1, 0x0}},
    {F, {"ERR_ECC_DOUBLE", 11, 1, 0x0}},
    {F, {"ERR_SOT_SYNC_HS", 12, 1, 0x0}},
    {F, {"ERR_SOT_HS", 13, 1, 0x0}},
    {F, {"STOP_STATE", 17, 1, 0x0}},
    {F, {"ST_LINE_BUF_FULL", 18, 1, 0x0}},
    {F, {"SHT_PKT_FIFO_N_EMPTY", 19, 1, 0x0}},
    {F, {"SHT_PKT_FIFO_FULL", 20, 1, 0x0}},
    {F, {"LANE_CONFIG_ERR", 21, 1, 0x0}},
    {F, {"WC_CORRUPTION", 22, 1, 0x0}},
    {F, {"RX_SKEWCALHS", 29, 1, 0x0}},
    {F, {"FRAME_RECEIVED", 31, 1, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/GENERIC_SHT_PKT", 0x700030}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},
    {F, {"VIRTUAL_CHANNEL", 6, 2, 0x0}},
    {F, {"DATA", 8, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VCX_FRAME_ERROR", 0x700034}},
    {F, {"FRAME_LEVEL_ERR_VC4", 0, 1, 0x0}},
    {F, {"FRAME_SYNC_ERR_VC4", 1, 1, 0x0}},
    {F, {"FRAME_LEVEL_ERR_VC5", 2, 1, 0x0}},
    {F, {"FRAME_SYNC_ERR_VC5", 3, 1, 0x0}},
    {F, {"FRAME_LEVEL_ERR_VC6", 4, 1, 0x0}},
    {F, {"FRAME_SYNC_ERR_VC6", 5, 1, 0x0}},
    {F, {"FRAME_LEVEL_ERR_VC7", 6, 1, 0x0}},
    {F, {"FRAME_SYNC_ERR_VC7", 7, 1, 0x0}},
    {F, {"FRAME_LEVEL_ERR_VC8", 8, 1, 0x0}},
    {F, {"FRAME_SYNC_ERR_VC8", 9, 1, 0x0}},
    {F, {"FRAME_LEVEL_ERR_VC9", 10, 1, 0x0}},
    {F, {"FRAME_SYNC_ERR_VC9", 11, 1, 0x0}},
    {F, {"FRAME_LEVEL_ERR_VC10", 12, 1, 0x0}},
    {F, {"FRAME_SYNC_ERR_VC10", 13, 1, 0x0}},
    {F, {"FRAME_LEVEL_ERR_VC11", 14, 1, 0x0}},
    {F, {"FRAME_SYNC_ERR_VC11", 15, 1, 0x0}},
    {F, {"FRAME_LEVEL_ERR_VC12", 16, 1, 0x0}},
    {F, {"FRAME_SYNC_ERR_VC12", 17, 1, 0x0}},
    {F, {"FRAME_LEVEL_ERR_VC13", 18, 1, 0x0}},
    {F, {"FRAME_SYNC_ERR_VC13", 19, 1, 0x0}},
    {F, {"FRAME_LEVEL_ERR_VC14", 20, 1, 0x0}},
    {F, {"FRAME_SYNC_ERR_VC14", 21, 1, 0x0}},
    {F, {"FRAME_LEVEL_ERR_VC15", 22, 1, 0x0}},
    {F, {"FRAME_SYNC_ERR_VC15", 23, 1, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/CLK_LANE_INFO", 0x70003C}},
    {F, {"RESERVED", 0, 1, 0x0}},
    {F, {"STOP_STATE", 1, 1, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/LANE0_INFO", 0x700040}},
    {F, {"SOT_SYNC_ERR", 0, 1, 0x0}},
    {F, {"SOT_ERR", 1, 1, 0x0}},
    {F, {"STOP_STATE", 5, 1, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/LANE1_INFO", 0x700044}},
    {F, {"SOT_SYNC_ERR", 0, 1, 0x0}},
    {F, {"SOT_ERR", 1, 1, 0x0}},
    {F, {"STOP_STATE", 5, 1, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/LANE2_INFO", 0x700048}},
    {F, {"SOT_SYNC_ERR", 0, 1, 0x0}},
    {F, {"SOT_ERR", 1, 1, 0x0}},
    {F, {"STOP_STATE", 5, 1, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/LANE3_INFO", 0x70004C}},
    {F, {"SOT_SYNC_ERR", 0, 1, 0x0}},
    {F, {"SOT_ERR", 1, 1, 0x0}},
    {F, {"STOP_STATE", 5, 1, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC0_IMAGE_INFO_1", 0x700060}},
    {F, {"BYTE_COUNT", 0, 16, 0x0}},
    {F, {"LINE_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC0_IMAGE_INFO_2", 0x700064}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC1_IMAGE_INFO_1", 0x700068}},
    {F, {"BYTE_COUNT", 0, 16, 0x0}},
    {F, {"LINE_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC1_IMAGE_INFO_2", 0x70006C}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC2_IMAGE_INFO_1", 0x700070}},
    {F, {"BYTE_COUNT", 0, 16, 0x0}},
    {F, {"LINE_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC2_IMAGE_INFO_2", 0x700074}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC3_IMAGE_INFO_1", 0x700078}},
    {F, {"BYTE_COUNT", 0, 16, 0x0}},
    {F, {"LINE_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC3_IMAGE_INFO_2", 0x70007C}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC4_IMAGE_INFO_1", 0x700080}},
    {F, {"BYTE_COUNT", 0, 16, 0x0}},
    {F, {"LINE_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC4_IMAGE_INFO_2", 0x700084}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC5_IMAGE_INFO_1", 0x700088}},
    {F, {"BYTE_COUNT", 0, 16, 0x0}},
    {F, {"LINE_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC5_IMAGE_INFO_2", 0x70008C}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC6_IMAGE_INFO_1", 0x700090}},
    {F, {"BYTE_COUNT", 0, 16, 0x0}},
    {F, {"LINE_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC6_IMAGE_INFO_2", 0x700094}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC7_IMAGE_INFO_1", 0x700098}},
    {F, {"BYTE_COUNT", 0, 16, 0x0}},
    {F, {"LINE_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC7_IMAGE_INFO_2", 0x70009C}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC8_IMAGE_INFO_1", 0x7000A0}},
    {F, {"BYTE_COUNT", 0, 16, 0x0}},
    {F, {"LINE_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC8_IMAGE_INFO_2", 0x7000A4}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC9_IMAGE_INFO_1", 0x7000A8}},
    {F, {"BYTE_COUNT", 0, 16, 0x0}},
    {F, {"LINE_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC9_IMAGE_INFO_2", 0x7000AC}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC10_IMAGE_INFO_1", 0x7000B0}},
    {F, {"BYTE_COUNT", 0, 16, 0x0}},
    {F, {"LINE_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC10_IMAGE_INFO_2", 0x7000B4}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC11_IMAGE_INFO_1", 0x7000B8}},
    {F, {"BYTE_COUNT", 0, 16, 0x0}},
    {F, {"LINE_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC11_IMAGE_INFO_2", 0x7000BC}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC12_IMAGE_INFO_1", 0x7000C0}},
    {F, {"BYTE_COUNT", 0, 16, 0x0}},
    {F, {"LINE_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC12_IMAGE_INFO_2", 0x7000C4}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC13_IMAGE_INFO_1", 0x7000C8}},
    {F, {"BYTE_COUNT", 0, 16, 0x0}},
    {F, {"LINE_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC13_IMAGE_INFO_2", 0x7000CC}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC14_IMAGE_INFO_1", 0x7000D0}},
    {F, {"BYTE_COUNT", 0, 16, 0x0}},
    {F, {"LINE_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC14_IMAGE_INFO_2", 0x7000D4}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC15_IMAGE_INFO_1", 0x7000D8}},
    {F, {"BYTE_COUNT", 0, 16, 0x0}},
    {F, {"LINE_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/VC15_IMAGE_INFO_2", 0x7000DC}},
    {F, {"DATA_TYPE", 0, 6, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/CONTROL", 0x708000}},
    {F, {"SRST", 0, 1, 0x0}},
    {F, {"DPHY_EN", 1, 1, 0x1}},

    {R, {"SENSOR_IF/MIPI_RX/IDELAY_TAP_VALUE", 0x708004}},
    {F, {"LANE_0", 0, 5, 0x0}},
    {F, {"LANE_1", 8, 5, 0x0}},
    {F, {"LANE_2", 16, 5, 0x0}},
    {F, {"LANE_3", 24, 5, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/INIT", 0x708008}},
    {F, {"INIT_VAL", 0, 32, 0x186A0}},

    {R, {"SENSOR_IF/MIPI_RX/HS_TIMEOUT", 0x708010}},
    {F, {"RX_VALUE", 0, 32, 0x10005}},

    {R, {"SENSOR_IF/MIPI_RX/ESC_TIMEOUT", 0x708014}},
    {F, {"VALUE", 0, 32, 0x6400}},

    {R, {"SENSOR_IF/MIPI_RX/CL_STATUS", 0x708018}},
    {F, {"MODE", 0, 2, 0x0}},
    {F, {"ULPS", 2, 1, 0x0}},
    {F, {"INIT_DONE", 3, 1, 0x0}},
    {F, {"STOP_STATE", 4, 1, 0x0}},
    {F, {"ERR_CONTROL", 5, 1, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/LANE0_STATUS", 0x70801C}},
    {F, {"MODE", 0, 2, 0x0}},
    {F, {"ULPS", 2, 1, 0x0}},
    {F, {"INIT_DONE", 3, 1, 0x0}},
    {F, {"HS_ABORT", 4, 1, 0x0}},
    {F, {"ESC_ABORT", 5, 1, 0x0}},
    {F, {"STOP_STATE", 6, 1, 0x0}},
    {F, {"PKT_CNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/LANE1_STATUS", 0x708020}},
    {F, {"MODE", 0, 2, 0x0}},
    {F, {"ULPS", 2, 1, 0x0}},
    {F, {"INIT_DONE", 3, 1, 0x0}},
    {F, {"HS_ABORT", 4, 1, 0x0}},
    {F, {"ESC_ABORT", 5, 1, 0x0}},
    {F, {"STOP_STATE", 6, 1, 0x0}},
    {F, {"PKT_CNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/LANE2_STATUS", 0x708024}},
    {F, {"MODE", 0, 2, 0x0}},
    {F, {"ULPS", 2, 1, 0x0}},
    {F, {"INIT_DONE", 3, 1, 0x0}},
    {F, {"HS_ABORT", 4, 1, 0x0}},
    {F, {"ESC_ABORT", 5, 1, 0x0}},
    {F, {"STOP_STATE", 6, 1, 0x0}},
    {F, {"PKT_CNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/LANE3_STATUS", 0x708028}},
    {F, {"MODE", 0, 2, 0x0}},
    {F, {"ULPS", 2, 1, 0x0}},
    {F, {"INIT_DONE", 3, 1, 0x0}},
    {F, {"HS_ABORT", 4, 1, 0x0}},
    {F, {"ESC_ABORT", 5, 1, 0x0}},
    {F, {"STOP_STATE", 6, 1, 0x0}},
    {F, {"PKT_CNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/LANE0_HS_SETTLE", 0x708030}},
    {F, {"HS_SETTLE_NS", 0, 9, 0x8D}},

    {R, {"SENSOR_IF/MIPI_RX/LANE1_HS_SETTLE", 0x708048}},
    {F, {"HS_SETTLE_NS", 0, 9, 0x8D}},

    {R, {"SENSOR_IF/MIPI_RX/LANE2_HS_SETTLE", 0x70804C}},
    {F, {"HS_SETTLE_NS", 0, 9, 0x8D}},

    {R, {"SENSOR_IF/MIPI_RX/LANE3_HS_SETTLE", 0x708050}},
    {F, {"HS_SETTLE_NS", 0, 9, 0x8D}},

    {R, {"SENSOR_IF/MIPI_RX/LANE4_HS_SETTLE", 0x708054}},
    {F, {"HS_SETTLE_NS", 0, 9, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/LANE5_HS_SETTLE", 0x708058}},
    {F, {"HS_SETTLE_NS", 0, 9, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/LANE6_HS_SETTLE", 0x70805C}},
    {F, {"HS_SETTLE_NS", 0, 9, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/LANE7_HS_SETTLE", 0x708060}},
    {F, {"HS_SETTLE_NS", 0, 9, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/LANE4_STATUS", 0x708064}},
    {F, {"MODE", 0, 2, 0x0}},
    {F, {"ULPS", 2, 1, 0x0}},
    {F, {"INIT_DONE", 3, 1, 0x0}},
    {F, {"HS_ABORT", 4, 1, 0x0}},
    {F, {"ESC_ABORT", 5, 1, 0x0}},
    {F, {"STOP_STATE", 6, 1, 0x0}},
    {F, {"PKT_CNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/LANE5_STATUS", 0x708068}},
    {F, {"MODE", 0, 2, 0x0}},
    {F, {"ULPS", 2, 1, 0x0}},
    {F, {"INIT_DONE", 3, 1, 0x0}},
    {F, {"HS_ABORT", 4, 1, 0x0}},
    {F, {"ESC_ABORT", 5, 1, 0x0}},
    {F, {"STOP_STATE", 6, 1, 0x0}},
    {F, {"PKT_CNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/LANE6_STATUS", 0x70806C}},
    {F, {"MODE", 0, 2, 0x0}},
    {F, {"ULPS", 2, 1, 0x0}},
    {F, {"INIT_DONE", 3, 1, 0x0}},
    {F, {"HS_ABORT", 4, 1, 0x0}},
    {F, {"ESC_ABORT", 5, 1, 0x0}},
    {F, {"STOP_STATE", 6, 1, 0x0}},
    {F, {"PKT_CNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/LANE7_STATUS", 0x708070}},
    {F, {"MODE", 0, 2, 0x0}},
    {F, {"ULPS", 2, 1, 0x0}},
    {F, {"INIT_DONE", 3, 1, 0x0}},
    {F, {"HS_ABORT", 4, 1, 0x0}},
    {F, {"ESC_ABORT", 5, 1, 0x0}},
    {F, {"STOP_STATE", 6, 1, 0x0}},
    {F, {"PKT_CNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/MIPI_RX/IDELAY_TAP_VALUE_L4_TO_L7", 0x708074}},
    {F, {"LANE_4", 0, 5, 0x0}},
    {F, {"LANE_5", 8, 5, 0x0}},
    {F, {"LANE_6", 16, 5, 0x0}},
    {F, {"LANE_7", 24, 5, 0x0}},

    {R, {"SENSOR_IF/GEN41_IF_CTRL/CONTROL", 0x70F000}},
    {F, {"ENABLE", 0, 1, 0x0}},
    {F, {"BYPASS", 1, 1, 0x0}},
    {F, {"LAST_CTRL_MODE", 11, 1, 0x0}},
    {F, {"HALF_WORD_SWAP", 12, 1, 0x0}},
    {F, {"AUX_PRESENCE", 16, 1, 0x0}},
    {F, {"MIPI_RX_PRESENCE", 17, 1, 0x0}},
    {F, {"SLVS_RX_PRESENCE", 18, 1, 0x0}},
    {F, {"TD_POL_INV", 20, 1, 0x0}},
    {F, {"EM_POL_INV", 21, 1, 0x0}},
    {F, {"GEN_LAST", 22, 1, 0x0}},
    {F, {"AUX_IN_MODE", 24, 1, 0x0}},
    {F, {"AUX_8BN_4B", 25, 1, 0x0}},
    {F, {"SLVS_IN_MODE", 26, 1, 0x0}},

    {R, {"SENSOR_IF/GEN41_IF_CTRL/TRIGGER", 0x70F004}},
    {F, {"AFIFO_RESET", 0, 1, 0x0}},

    {R, {"SENSOR_IF/GEN41_IF_CTRL/TEST_PATTERN_CONTROL", 0x70F008}},
    {F, {"EVT_TYPE", 0, 4, 0x0}},
    {F, {"EVT_FORMAT", 4, 2, 0x2}},
    {A, {"2.0", 0x2}},
    {A, {"3.0", 0x3}},
    {F, {"ENABLE", 8, 1, 0x0}},
    {F, {"VECTOR_MODE", 9, 1, 0x0}},

    {R, {"SENSOR_IF/GEN41_IF_CTRL/TEST_PATTERN_N_PERIOD", 0x70F00C}},
    {F, {"VALID_RATIO", 0, 10, 0x0}},
    {F, {"LENGTH", 16, 16, 0x0}},

    {R, {"SENSOR_IF/GEN41_IF_CTRL/TEST_PATTERN_P_PERIOD", 0x70F010}},
    {F, {"VALID_RATIO", 0, 10, 0x0}},
    {F, {"LENGTH", 16, 16, 0x0}},

    {R, {"SENSOR_IF/GEN41_IF_CTRL/TEST_PATTERN_VECTOR", 0x70F014}},
    {F, {"SEED_VALUE", 0, 5, 0x1F}},

    {R, {"SENSOR_IF/GEN41_IF_CTRL/OOB_FILTER_CONTROL", 0x70F018}},
    {F, {"ENABLE", 0, 1, 0x0}},

    {R, {"SENSOR_IF/GEN41_IF_CTRL/OOB_FILTER_ORIGIN", 0x70F01C}},
    {F, {"Y", 0, 11, 0x0}},
    {F, {"X", 16, 11, 0x0}},

    {R, {"SENSOR_IF/GEN41_IF_CTRL/OOB_FILTER_SIZE", 0x70F020}},
    {F, {"HEIGHT", 0, 11, 0x2CF}},
    {F, {"WIDTH", 16, 11, 0x4FF}},

    {R, {"SENSOR_IF/GEN41_IF_CTRL/CCAM5_CONTROL", 0x70F024}},
    {F, {"PSU_EN", 0, 1, 0x0}},
    {F, {"RST_N", 1, 1, 0x0}},

    {R, {"SENSOR_IF/GEN41_IF_CTRL/TRIGGER_FWD", 0x70F028}},
    {F, {"TRIGGER_ID", 0, 8, 0x0}},
    {F, {"ENABLE", 8, 1, 0x0}},

    {R, {"SENSOR_IF/GEN41_IF_CTRL/MIPI_RX_CONTROL", 0x70F02C}},
    {F, {"VIDEO_RESET", 0, 1, 0x0}},

    {R, {"SENSOR_IF/GEN41_IF_CTRL/GEN41_CTRL", 0x70F030}},
    {F, {"BOOT", 0, 1, 0x0}},
    {F, {"DFT_MODE", 1, 1, 0x0}},
    {F, {"TEST_MODE", 2, 1, 0x0}},
    {F, {"I2C_ADDR", 3, 1, 0x0}},
    {F, {"CPU_DEBUG", 4, 1, 0x0}},
    {F, {"AGPIO", 5, 2, 0x0}},
    {F, {"DGPIO", 7, 1, 0x0}},
    {F, {"ARST_N", 8, 1, 0x0}},
    {F, {"TDRST_N", 9, 1, 0x0}},
    {F, {"XCLEAR", 10, 1, 0x0}},
    {F, {"I2C_SPI_SEL", 11, 1, 0x0}},
    {F, {"TRIG_IO_DIR", 12, 1, 0x0}},
    {F, {"DGPIO_DIR", 13, 1, 0x0}},
    {F, {"DGPIO_VAL", 14, 1, 0x0}},
    {F, {"OMODE_BUF", 15, 1, 0x0}},
    {F, {"CPUSTAIT_BUF", 16, 1, 0x0}},
    {F, {"TPA_SEL_1V", 17, 1, 0x0}},

    {R, {"SENSOR_IF/GEN41_IF_CTRL/GEN41_POWER_CTRL", 0x70F034}},
    {F, {"1V8_SYS", 0, 1, 0x0}},
    {F, {"3V3_SYS", 1, 1, 0x0}},
    {F, {"VDDLSC", 2, 1, 0x0}},
    {F, {"VDDPLL", 3, 1, 0x0}},
    {F, {"VDDLIF", 4, 1, 0x0}},
    {F, {"VDDMIF", 5, 1, 0x0}},
    {F, {"VDDIO", 6, 1, 0x0}},

    {R, {"SENSOR_IF/GEN41_IF_CTRL/GEN41_CLK_CTRL", 0x70F038}},
    {F, {"ENABLE", 0, 1, 0x0}},
    {F, {"VALID", 1, 1, 0x0}},
    {F, {"VCO_HIGH_TIME", 10, 6, 0x0}},
    {F, {"VCO_LOW_TIME", 20, 6, 0x0}},

    {R, {"SENSOR_IF/GEN41_AUX_IF/IODELAY_DEC", 0x710004}},
    {F, {"DATA", 0, 8, 0x0}},
    {F, {"VALID", 8, 1, 0x0}},
    {F, {"CLK", 9, 1, 0x0}},

    {R, {"SENSOR_IF/GEN41_AUX_IF/IODELAY_INC", 0x710008}},
    {F, {"DATA", 0, 8, 0x0}},
    {F, {"VALID", 8, 1, 0x0}},
    {F, {"CLK", 9, 1, 0x0}},

    {R, {"SENSOR_IF/GEN41_AUX_IF/IODELAY_LOAD", 0x71000C}},
    {F, {"DATA", 0, 8, 0x0}},
    {F, {"VALID", 8, 1, 0x0}},
    {F, {"CLK", 9, 1, 0x0}},

    {R, {"SENSOR_IF/GEN41_AUX_IF/IODELAY_SET_VALUE_0", 0x710010}},
    {F, {"DATA_0", 0, 5, 0x0}},
    {F, {"DATA_1", 5, 5, 0x0}},
    {F, {"DATA_2", 10, 5, 0x0}},
    {F, {"DATA_3", 15, 5, 0x0}},
    {F, {"DATA_4", 20, 5, 0x0}},
    {F, {"DATA_5", 25, 5, 0x0}},

    {R, {"SENSOR_IF/GEN41_AUX_IF/IODELAY_SET_VALUE_1", 0x710014}},
    {F, {"DATA_6", 0, 5, 0x0}},
    {F, {"DATA_7", 5, 5, 0x0}},
    {F, {"VALID", 10, 5, 0x0}},
    {F, {"CLK", 15, 5, 0x0}},

    {R, {"SENSOR_IF/GEN41_AUX_IF/IODELAY_GET_VALUE_0", 0x710018}},
    {F, {"DATA_0", 0, 5, 0x0}},
    {F, {"DATA_1", 5, 5, 0x0}},
    {F, {"DATA_2", 10, 5, 0x0}},
    {F, {"DATA_3", 15, 5, 0x0}},
    {F, {"DATA_4", 20, 5, 0x0}},
    {F, {"DATA_5", 25, 5, 0x0}},

    {R, {"SENSOR_IF/GEN41_AUX_IF/IODELAY_GET_VALUE_1", 0x71001C}},
    {F, {"DATA_6", 0, 5, 0x0}},
    {F, {"DATA_7", 5, 5, 0x0}},
    {F, {"VALID", 10, 5, 0x0}},
    {F, {"CLK", 15, 5, 0x0}},

    {R, {"SENSOR_IF/GEN41_AUX_IF/SAMPLE_CLK_EDGE", 0x710020}},
    {F, {"DATA", 0, 8, 0x0}},
    {F, {"VALID", 8, 1, 0x0}},

    {R, {"SENSOR_IF/VIP/CONTROL", 0x710100}},
    {F, {"MODE", 0, 3, 0x0}},
    {F, {"RATE", 3, 10, 0x0}},
    {F, {"REPEAT", 16, 13, 0x0}},
    {F, {"DISABLE_OUTPUT", 29, 1, 0x0}},
    {F, {"FORCE_READY", 30, 1, 0x1}},

    {R, {"SENSOR_IF/VIP/TEST_STATUS", 0x710104}},
    {F, {"STATUS", 0, 1, 0x0}},
    {F, {"RESET", 1, 1, 0x0}},

    {R, {"SENSOR_IF/VIP/PATTERN_TEST_DATA", 0x710108}},
    {F, {"DATA", 0, 32, 0x0}},

    {R, {"SENSOR_IF/VIP/TEST_RESULT", 0x71010C}},
    {F, {"RESULT", 0, 32, 0x0}},

    {R, {"SENSOR_IF/VIP/VALID_COUNTER", 0x710110}},
    {F, {"COUNTER", 0, 32, 0x0}},

    {R, {"SENSOR_IF/VIP/ERROR_COUNTER", 0x710114}},
    {F, {"COUNTER", 0, 32, 0x0}},

    {R, {"SENSOR_IF/VIP/ERROR_INDEX", 0x710118}},
    {F, {"INDEX", 0, 32, 0x0}},

    {R, {"SENSOR_IF/SLVS_RX/CORE_CONFIG", 0x710200}},
    {F, {"CORE_ENABLE", 0, 1, 0x0}},
    {F, {"DATA_FORMAT", 1, 2, 0x3}},

    {R, {"SENSOR_IF/SLVS_RX/PROTOCOL_CONFIG", 0x710204}},
    {F, {"ACTIVE_LANES", 0, 2, 0x3}},
    {F, {"EN_IN_BYTE_ENDIANNESS_SWAP", 4, 1, 0x0}},
    {F, {"EN_PLD_BYTE_ORDER_SWAP", 8, 1, 0x0}},
    {F, {"EN_PLD_BYTE_ENDIANNESS_SWAP", 12, 1, 0x0}},

    {R, {"SENSOR_IF/SLVS_RX/CORE_STAT", 0x710208}},
    {F, {"SOFT_RESET", 0, 1, 0x0}},
    {F, {"OUT_BUFFER_FULL", 1, 1, 0x0}},
    {F, {"PHASE_DETECT_LOCKED", 4, 4, 0x0}},
    {F, {"ACCU_REG_LOCKED", 8, 4, 0x0}},
    {F, {"PACKET_COUNT", 16, 16, 0x0}},

    {R, {"SENSOR_IF/SLVS_RX/SYNC1_CODE_MSB", 0x71020C}},
    {F, {"VALUE", 0, 32, 0xDB0DB1DB}},

    {R, {"SENSOR_IF/SLVS_RX/SYNC1_CODE_LSB", 0x710210}},
    {F, {"VALUE", 0, 32, 0x2DB3DB4D}},

    {R, {"SENSOR_IF/SLVS_RX/SYNC2_CODE_MSB", 0x710214}},
    {F, {"VALUE", 0, 32, 0xB5DB6DB7}},

    {R, {"SENSOR_IF/SLVS_RX/SYNC2_CODE_LSB", 0x710218}},
    {F, {"VALUE", 0, 32, 0xDB8DB9DB}},

    {R, {"SENSOR_IF/SLVS_RX/SYNC3_CODE_MSB", 0x71021C}},
    {F, {"VALUE", 0, 32, 0xADBBDBCD}},

    {R, {"SENSOR_IF/SLVS_RX/SYNC3_CODE_LSB", 0x710220}},
    {F, {"VALUE", 0, 32, 0xBDDBEDBF}},

    {R, {"SENSOR_IF/SLVS_RX/SYNC4_FS_CODE_MSB", 0x710224}},
    {F, {"VALUE", 0, 32, 0x0}},

    {R, {"SENSOR_IF/SLVS_RX/SYNC4_FS_CODE_LSB", 0x710228}},
    {F, {"VALUE", 0, 32, 0x0}},

    {R, {"SENSOR_IF/SLVS_RX/SYNC4_FE_CODE_MSB", 0x71022C}},
    {F, {"VALUE", 0, 32, 0x0}},

    {R, {"SENSOR_IF/SLVS_RX/SYNC4_FE_CODE_LSB", 0x710230}},
    {F, {"VALUE", 0, 32, 0x4000}},

    {R, {"SENSOR_IF/SLVS_RX/SYNC4_SAV_CODE_MSB", 0x710234}},
    {F, {"VALUE", 0, 32, 0x0}},

    {R, {"SENSOR_IF/SLVS_RX/SYNC4_SAV_CODE_LSB", 0x710238}},
    {F, {"VALUE", 0, 32, 0x8000}},

    {R, {"SENSOR_IF/SLVS_RX/SYNC4_EAV_CODE_MSB", 0x71023C}},
    {F, {"VALUE", 0, 32, 0x0}},

    {R, {"SENSOR_IF/SLVS_RX/SYNC4_EAV_CODE_LSB", 0x710240}},
    {F, {"VALUE", 0, 32, 0xC000}},

    {R, {"SENSOR_IF/SLVS_RX/BLK_CODE_MSB", 0x710244}},
    {F, {"VALUE", 0, 32, 0xB0B1B2B3}},

    {R, {"SENSOR_IF/SLVS_RX/BLK_CODE_LSB", 0x710248}},
    {F, {"VALUE", 0, 32, 0xB4B5B6B7}},

    {R, {"SENSOR_IF/SLVS_RX/DMY_CODE_MSB", 0x71024C}},
    {F, {"VALUE", 0, 32, 0x99999999}},

    {R, {"SENSOR_IF/SLVS_RX/DMY_CODE_LSB", 0x710250}},
    {F, {"VALUE", 0, 32, 0x99999999}},

    {R, {"SENSOR_IF/SLVS_RX/IODELAY_LOAD", 0x710254}},
    {F, {"LANE_0", 0, 1, 0x0}},
    {F, {"LANE_1", 8, 1, 0x0}},
    {F, {"LANE_2", 16, 1, 0x0}},
    {F, {"LANE_3", 24, 1, 0x0}},
    {F, {"BUSY", 31, 1, 0x0}},

    {R, {"SENSOR_IF/SLVS_RX/IODELAY_SET_VALUE_0", 0x710258}},
    {F, {"LANE_0", 0, 9, 0x0}},
    {F, {"LANE_1", 16, 9, 0x0}},

    {R, {"SENSOR_IF/SLVS_RX/IODELAY_SET_VALUE_1", 0x71025C}},
    {F, {"LANE_2", 0, 9, 0x0}},
    {F, {"LANE_3", 16, 9, 0x0}},

    {R, {"SENSOR_IF/SLVS_RX/STAT_IODELAY_SET_VALUE_0", 0x710260}},
    {F, {"LANE_0", 0, 9, 0x0}},
    {F, {"LANE_1", 18, 9, 0x0}},

    {R, {"SENSOR_IF/SLVS_RX/STAT_IODELAY_SET_VALUE_1", 0x710264}},
    {F, {"LANE_2", 0, 9, 0x0}},
    {F, {"LANE_3", 18, 9, 0x0}}

    // clang-format on
};

static uint32_t Imx636Evk2RegisterMapSize = sizeof(Imx636Evk2RegisterMap) / sizeof(Imx636Evk2RegisterMap[0]);

#endif // METAVISION_HAL_IMX636_EVK2_REGISTERMAP_H
