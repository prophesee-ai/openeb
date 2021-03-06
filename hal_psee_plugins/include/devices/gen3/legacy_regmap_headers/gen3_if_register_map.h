/**********************************************************************************************
 * WARNING: THIS FILE HAS BEEN GENERATED BY REG_MAP_MANAGER TOOLS PLEASE DO NOT MODIFY IT     *
 **********************************************************************************************
 * File: gen3_if_register_map.h                                                               *
 *                                                                                            *
 * Copyright (c) 2015-2018 Prophesee. All rights reserved.                                    *
 *                                                                                            *
 * Date:           08/10/2018 at 12h05m31s                                                    *
 * Name:           gen3_if_register_map                                                       *
 * Version:        1.0                                                                        *
 * Hash:           0ea2255587212f208ba5c9e2f53ac1a3f537456497c0bfdff9f15967e558ea69           *
 **********************************************************************************************
 * WARNING: THIS FILE HAS BEEN GENERATED BY REG_MAP_MANAGER TOOLS PLEASE DO NOT MODIFY IT     *
 *********************************************************************************************/

#ifndef METAVISION_HAL_GEN3_IF_REGISTER_MAP_H
#define METAVISION_HAL_GEN3_IF_REGISTER_MAP_H

//------------------------------------------------------------------------------------------------------------
// SISLEY_IF
//------------------------------------------------------------------------------------------------------------

#define SISLEY_IF_BASE_ADDR 0x00000000
#define SISLEY_IF_LAST_ADDR 0x00000010
#define SISLEY_IF_SIZE 0x000000C0

#define SISLEY_IF_TEST_PATTERN_CONTROL_ADDR 0x00000000
#define SISLEY_IF_TEST_PATTERN_CONTROL_ENABLE_BIT_IDX 0
#define SISLEY_IF_TEST_PATTERN_CONTROL_ENABLE_WIDTH 1
#define SISLEY_IF_TEST_PATTERN_CONTROL_ENABLE_DEFAULT 0x00000000
#define SISLEY_IF_TEST_PATTERN_CONTROL_TYPE_BIT_IDX 4
#define SISLEY_IF_TEST_PATTERN_CONTROL_TYPE_WIDTH 1
#define SISLEY_IF_TEST_PATTERN_CONTROL_TYPE_DEFAULT 0x00000000
#define SISLEY_IF_TEST_PATTERN_CONTROL_PIXEL_TYPE_BIT_IDX 8
#define SISLEY_IF_TEST_PATTERN_CONTROL_PIXEL_TYPE_WIDTH 1
#define SISLEY_IF_TEST_PATTERN_CONTROL_PIXEL_TYPE_DEFAULT 0x00000000
#define SISLEY_IF_TEST_PATTERN_CONTROL_PIXEL_POLARITY_BIT_IDX 12
#define SISLEY_IF_TEST_PATTERN_CONTROL_PIXEL_POLARITY_WIDTH 1
#define SISLEY_IF_TEST_PATTERN_CONTROL_PIXEL_POLARITY_DEFAULT 0x00000000

#define SISLEY_IF_TEST_PATTERN_N_PERIOD_ADDR 0x00000004
#define SISLEY_IF_TEST_PATTERN_N_PERIOD_VALID_RATIO_BIT_IDX 0
#define SISLEY_IF_TEST_PATTERN_N_PERIOD_VALID_RATIO_WIDTH 10
#define SISLEY_IF_TEST_PATTERN_N_PERIOD_VALID_RATIO_DEFAULT 0x00000000
#define SISLEY_IF_TEST_PATTERN_N_PERIOD_LENGTH_BIT_IDX 16
#define SISLEY_IF_TEST_PATTERN_N_PERIOD_LENGTH_WIDTH 16
#define SISLEY_IF_TEST_PATTERN_N_PERIOD_LENGTH_DEFAULT 0x00000000

#define SISLEY_IF_TEST_PATTERN_P_PERIOD_ADDR 0x00000008
#define SISLEY_IF_TEST_PATTERN_P_PERIOD_VALID_RATIO_BIT_IDX 0
#define SISLEY_IF_TEST_PATTERN_P_PERIOD_VALID_RATIO_WIDTH 10
#define SISLEY_IF_TEST_PATTERN_P_PERIOD_VALID_RATIO_DEFAULT 0x00000000
#define SISLEY_IF_TEST_PATTERN_P_PERIOD_LENGTH_BIT_IDX 16
#define SISLEY_IF_TEST_PATTERN_P_PERIOD_LENGTH_WIDTH 16
#define SISLEY_IF_TEST_PATTERN_P_PERIOD_LENGTH_DEFAULT 0x00000000

#define SISLEY_IF_CONTROL_ADDR 0x0000000C
#define SISLEY_IF_CONTROL_SELF_ACK_BIT_IDX 0
#define SISLEY_IF_CONTROL_SELF_ACK_WIDTH 1
#define SISLEY_IF_CONTROL_SELF_ACK_DEFAULT 0x00000000
#define SISLEY_IF_CONTROL_SENSOR_CLK_EN_BIT_IDX 1
#define SISLEY_IF_CONTROL_SENSOR_CLK_EN_WIDTH 1
#define SISLEY_IF_CONTROL_SENSOR_CLK_EN_DEFAULT 0x00000001
#define SISLEY_IF_CONTROL_EM_RSTN_TRIGGER_EN_BIT_IDX 4
#define SISLEY_IF_CONTROL_EM_RSTN_TRIGGER_EN_WIDTH 1
#define SISLEY_IF_CONTROL_EM_RSTN_TRIGGER_EN_DEFAULT 0x00000000

#define SISLEY_IF_TRIGGERS_ADDR 0x00000010
#define SISLEY_IF_TRIGGERS_RESET_AFIFO_BIT_IDX 0
#define SISLEY_IF_TRIGGERS_RESET_AFIFO_WIDTH 1
#define SISLEY_IF_TRIGGERS_RESET_AFIFO_DEFAULT 0x00000000

#endif // METAVISION_HAL_GEN3_IF_REGISTER_MAP_H
