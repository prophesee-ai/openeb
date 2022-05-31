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

#ifndef SRC_INC_IMU_REGISTERMAP_H_
#define SRC_INC_IMU_REGISTERMAP_H_

RegmapData IMURegisterMap[] = {
    {R, {{"AX_MSB", 0x3b}}}, {F, {{"VALUE", 0, 8, 0x0}}}, {R, {{"AX_LSB", 0x3c}}}, {F, {{"VALUE", 0, 8, 0x0}}},
    {R, {{"AY_MSB", 0x3d}}}, {F, {{"VALUE", 0, 8, 0x0}}}, {R, {{"AY_LSB", 0x3e}}}, {F, {{"VALUE", 0, 8, 0x0}}},
    {R, {{"AZ_MSB", 0x3f}}}, {F, {{"VALUE", 0, 8, 0x0}}}, {R, {{"AZ_LSB", 0x40}}}, {F, {{"VALUE", 0, 8, 0x0}}},
    {R, {{"GX_MSB", 0x43}}}, {F, {{"VALUE", 0, 8, 0x0}}}, {R, {{"GX_LSB", 0x44}}}, {F, {{"VALUE", 0, 8, 0x0}}},
    {R, {{"GY_MSB", 0x45}}}, {F, {{"VALUE", 0, 8, 0x0}}}, {R, {{"GY_LSB", 0x46}}}, {F, {{"VALUE", 0, 8, 0x0}}},
    {R, {{"GZ_MSB", 0x47}}}, {F, {{"VALUE", 0, 8, 0x0}}}, {R, {{"GZ_LSB", 0x48}}}, {F, {{"VALUE", 0, 8, 0x0}}},
    {R, {{"WHOAMI", 0x75}}}, {F, {{"VALUE", 0, 8, 0x68}}}

};
unsigned int IMURegisterMapSize = sizeof(IMURegisterMap) / sizeof(IMURegisterMap[0]);
#endif
