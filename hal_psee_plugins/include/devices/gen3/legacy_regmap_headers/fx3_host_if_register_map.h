/**********************************************************************************************
 * WARNING: THIS FILE HAS BEEN GENERATED BY REG_MAP_MANAGER TOOLS PLEASE DO NOT MODIFY IT     *
 **********************************************************************************************
 * File: fx3_host_if_register_map.h                                                           *
 *                                                                                            *
 * Copyright (c) 2015-2018 Prophesee. All rights reserved.                                    *
 *                                                                                            *
 * Date:           08/10/2018 at 12h05m31s                                                    *
 * Name:           fx3_host_if_register_map                                                   *
 * Version:        1.0                                                                        *
 * Hash:           3bfdc4377fe90051cbf001be0c0febb225899c8714f374e9eb0d9898bd4bc05b           *
 **********************************************************************************************
 * WARNING: THIS FILE HAS BEEN GENERATED BY REG_MAP_MANAGER TOOLS PLEASE DO NOT MODIFY IT     *
 *********************************************************************************************/

#ifndef METAVISION_HAL_FX3_HOST_IF_REGISTER_MAP_H
#define METAVISION_HAL_FX3_HOST_IF_REGISTER_MAP_H

//------------------------------------------------------------------------------------------------------------
// FX3_HOST_IF
//------------------------------------------------------------------------------------------------------------

#define FX3_HOST_IF_BASE_ADDR 0x00000000
#define FX3_HOST_IF_LAST_ADDR 0x00000008
#define FX3_HOST_IF_SIZE 0x00000100

#define FX3_HOST_IF_PKT_END_ENABLE_ADDR 0x00000000
#define FX3_HOST_IF_PKT_END_ENABLE_SHORT_PACKET_ENABLE_BIT_IDX 0
#define FX3_HOST_IF_PKT_END_ENABLE_SHORT_PACKET_ENABLE_WIDTH 1
#define FX3_HOST_IF_PKT_END_ENABLE_SHORT_PACKET_ENABLE_DEFAULT 0x00000001
#define FX3_HOST_IF_PKT_END_ENABLE_SHORT_PACKET_ENABLE_SKIP_BIT_IDX 1
#define FX3_HOST_IF_PKT_END_ENABLE_SHORT_PACKET_ENABLE_SKIP_WIDTH 1
#define FX3_HOST_IF_PKT_END_ENABLE_SHORT_PACKET_ENABLE_SKIP_DEFAULT 0x00000000

#define FX3_HOST_IF_PKT_END_INTERVAL_US_ADDR 0x00000004
#define FX3_HOST_IF_PKT_END_INTERVAL_US_BIT_IDX 0
#define FX3_HOST_IF_PKT_END_INTERVAL_US_WIDTH 32
#define FX3_HOST_IF_PKT_END_INTERVAL_US_DEFAULT 0x00000400

#define FX3_HOST_IF_PKT_END_DATA_COUNT_ADDR 0x00000008
#define FX3_HOST_IF_PKT_END_DATA_COUNT_BIT_IDX 0
#define FX3_HOST_IF_PKT_END_DATA_COUNT_WIDTH 32
#define FX3_HOST_IF_PKT_END_DATA_COUNT_DEFAULT 0x00000400

#endif // METAVISION_HAL_FX3_HOST_IF_REGISTER_MAP_H
