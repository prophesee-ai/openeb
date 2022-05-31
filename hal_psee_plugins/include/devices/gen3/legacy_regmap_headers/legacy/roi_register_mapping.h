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

#ifndef METAVISION_HAL_ROI_REGISTER_MAPPING_H
#define METAVISION_HAL_ROI_REGISTER_MAPPING_H

//-----------------------------------------------------------------------------
// Register Bank name is    SISLEY_SENSOR_IF_ROI
//-----------------------------------------------------------------------------

#define SISLEY_ROI_BASE_ADDRESS 0x00000000    // regbank_address
#define SISLEY_ROI_START_ADDR 0x00000000      // regbank_address
#define SISLEY_ROI_LAST_ADDR 0x0000033C       // regbank_address
#define SISLEY_ROI_TD_X_START_ADDR 0x00000000 // regbank_address
#define SISLEY_ROI_TD_X_LAST_ADDR 0x00000050  // regbank_address
#define SISLEY_ROI_TD_X_0_ADDR 0x00000000     // regbank_address
#define SISLEY_ROI_TD_X_1_ADDR 0x00000004     // regbank_address
#define SISLEY_ROI_TD_X_2_ADDR 0x00000008     // regbank_address
#define SISLEY_ROI_TD_X_3_ADDR 0x0000000C     // regbank_address
#define SISLEY_ROI_TD_X_4_ADDR 0x00000010     // regbank_address
#define SISLEY_ROI_TD_X_5_ADDR 0x00000014     // regbank_address
#define SISLEY_ROI_TD_X_6_ADDR 0x00000018     // regbank_address
#define SISLEY_ROI_TD_X_7_ADDR 0x0000001C     // regbank_address
#define SISLEY_ROI_TD_X_8_ADDR 0x00000020     // regbank_address
#define SISLEY_ROI_TD_X_9_ADDR 0x00000024     // regbank_address
#define SISLEY_ROI_TD_X_10_ADDR 0x00000028    // regbank_address
#define SISLEY_ROI_TD_X_11_ADDR 0x0000002C    // regbank_address
#define SISLEY_ROI_TD_X_12_ADDR 0x00000030    // regbank_address
#define SISLEY_ROI_TD_X_13_ADDR 0x00000034    // regbank_address
#define SISLEY_ROI_TD_X_14_ADDR 0x00000038    // regbank_address
#define SISLEY_ROI_TD_X_15_ADDR 0x0000003C    // regbank_address
#define SISLEY_ROI_TD_X_16_ADDR 0x00000040    // regbank_address
#define SISLEY_ROI_TD_X_17_ADDR 0x00000044    // regbank_address
#define SISLEY_ROI_TD_X_18_ADDR 0x00000048    // regbank_address
#define SISLEY_ROI_TD_X_19_ADDR 0x0000004C    // regbank_address
#define SISLEY_ROI_TD_X_20_ADDR 0x00000050    // regbank_address
#define SISLEY_ROI_TD_Y_START_ADDR 0x00000100 // regbank_address
#define SISLEY_ROI_TD_Y_LAST_ADDR 0x0000013C  // regbank_address
#define SISLEY_ROI_TD_Y_0_ADDR 0x00000100     // regbank_address
#define SISLEY_ROI_TD_Y_1_ADDR 0x00000104     // regbank_address
#define SISLEY_ROI_TD_Y_2_ADDR 0x00000108     // regbank_address
#define SISLEY_ROI_TD_Y_3_ADDR 0x0000010C     // regbank_address
#define SISLEY_ROI_TD_Y_4_ADDR 0x00000110     // regbank_address
#define SISLEY_ROI_TD_Y_5_ADDR 0x00000114     // regbank_address
#define SISLEY_ROI_TD_Y_6_ADDR 0x00000118     // regbank_address
#define SISLEY_ROI_TD_Y_7_ADDR 0x0000011C     // regbank_address
#define SISLEY_ROI_TD_Y_8_ADDR 0x00000120     // regbank_address
#define SISLEY_ROI_TD_Y_9_ADDR 0x00000124     // regbank_address
#define SISLEY_ROI_TD_Y_10_ADDR 0x00000128    // regbank_address
#define SISLEY_ROI_TD_Y_11_ADDR 0x0000012C    // regbank_address
#define SISLEY_ROI_TD_Y_12_ADDR 0x00000130    // regbank_address
#define SISLEY_ROI_TD_Y_13_ADDR 0x00000134    // regbank_address
#define SISLEY_ROI_TD_Y_14_ADDR 0x00000138    // regbank_address
#define SISLEY_ROI_TD_Y_15_ADDR 0x0000013C    // regbank_address
#define SISLEY_ROI_EM_X_START_ADDR 0x00000200 // regbank_address
#define SISLEY_ROI_EM_X_LAST_ADDR 0x00000250  // regbank_address
#define SISLEY_ROI_EM_X_0_ADDR 0x00000200     // regbank_address
#define SISLEY_ROI_EM_X_1_ADDR 0x00000204     // regbank_address
#define SISLEY_ROI_EM_X_2_ADDR 0x00000208     // regbank_address
#define SISLEY_ROI_EM_X_3_ADDR 0x0000020C     // regbank_address
#define SISLEY_ROI_EM_X_4_ADDR 0x00000210     // regbank_address
#define SISLEY_ROI_EM_X_5_ADDR 0x00000214     // regbank_address
#define SISLEY_ROI_EM_X_6_ADDR 0x00000218     // regbank_address
#define SISLEY_ROI_EM_X_7_ADDR 0x0000021C     // regbank_address
#define SISLEY_ROI_EM_X_8_ADDR 0x00000220     // regbank_address
#define SISLEY_ROI_EM_X_9_ADDR 0x00000224     // regbank_address
#define SISLEY_ROI_EM_X_10_ADDR 0x00000228    // regbank_address
#define SISLEY_ROI_EM_X_11_ADDR 0x0000022C    // regbank_address
#define SISLEY_ROI_EM_X_12_ADDR 0x00000230    // regbank_address
#define SISLEY_ROI_EM_X_13_ADDR 0x00000234    // regbank_address
#define SISLEY_ROI_EM_X_14_ADDR 0x00000238    // regbank_address
#define SISLEY_ROI_EM_X_15_ADDR 0x0000023C    // regbank_address
#define SISLEY_ROI_EM_X_16_ADDR 0x00000240    // regbank_address
#define SISLEY_ROI_EM_X_17_ADDR 0x00000244    // regbank_address
#define SISLEY_ROI_EM_X_18_ADDR 0x00000248    // regbank_address
#define SISLEY_ROI_EM_X_19_ADDR 0x0000024C    // regbank_address
#define SISLEY_ROI_EM_X_20_ADDR 0x00000250    // regbank_address
#define SISLEY_ROI_EM_Y_START_ADDR 0x00000300 // regbank_address
#define SISLEY_ROI_EM_Y_LAST_ADDR 0x0000033C  // regbank_address
#define SISLEY_ROI_EM_Y_0_ADDR 0x00000300     // regbank_address
#define SISLEY_ROI_EM_Y_1_ADDR 0x00000304     // regbank_address
#define SISLEY_ROI_EM_Y_2_ADDR 0x00000308     // regbank_address
#define SISLEY_ROI_EM_Y_3_ADDR 0x0000030C     // regbank_address
#define SISLEY_ROI_EM_Y_4_ADDR 0x00000310     // regbank_address
#define SISLEY_ROI_EM_Y_5_ADDR 0x00000314     // regbank_address
#define SISLEY_ROI_EM_Y_6_ADDR 0x00000318     // regbank_address
#define SISLEY_ROI_EM_Y_7_ADDR 0x0000031C     // regbank_address
#define SISLEY_ROI_EM_Y_8_ADDR 0x00000320     // regbank_address
#define SISLEY_ROI_EM_Y_9_ADDR 0x00000324     // regbank_address
#define SISLEY_ROI_EM_Y_10_ADDR 0x00000328    // regbank_address
#define SISLEY_ROI_EM_Y_11_ADDR 0x0000032C    // regbank_address
#define SISLEY_ROI_EM_Y_12_ADDR 0x00000330    // regbank_address
#define SISLEY_ROI_EM_Y_13_ADDR 0x00000334    // regbank_address
#define SISLEY_ROI_EM_Y_14_ADDR 0x00000338    // regbank_address
#define SISLEY_ROI_EM_Y_15_ADDR 0x0000033C    // regbank_address
#define SISLEY_ROI_TD_X_0 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_1 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_2 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_3 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_4 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_5 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_6 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_7 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_8 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_9 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_10 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_11 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_12 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_13 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_14 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_15 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_16 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_17 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_18 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_19 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_X_20 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_Y_0 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_Y_1 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_Y_2 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_Y_3 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_Y_4 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_Y_5 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_Y_6 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_Y_7 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_Y_8 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_Y_9 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_Y_10 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_Y_11 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_Y_12 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_Y_13 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_Y_14 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_TD_Y_15 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_0 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_1 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_2 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_3 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_4 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_5 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_6 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_7 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_8 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_9 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_10 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_11 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_12 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_13 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_14 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_15 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_16 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_17 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_18 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_19 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_X_20 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_Y_0 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_Y_1 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_Y_2 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_Y_3 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_Y_4 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_Y_5 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_Y_6 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_Y_7 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_Y_8 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_Y_9 0x00000000          // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_Y_10 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_Y_11 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_Y_12 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_Y_13 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_Y_14 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR
#define SISLEY_ROI_EM_Y_15 0x00000000         // SISLEY_ROI_EM_Y_15_ADDR

#endif // METAVISION_HAL_ROI_REGISTER_MAPPING_H
