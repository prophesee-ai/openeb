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

#ifndef METAVISION_HAL_SAMPLE_REGISTER_ACCESS_H
#define METAVISION_HAL_SAMPLE_REGISTER_ACCESS_H


constexpr int kEvk4EndpointControlOut = 0x02;
constexpr int kEvk4EndpointControlIn = 0x82;
constexpr int kEvk4EndpointDataIn = 0x81;
constexpr int kEvk4Interface = 0;

class SampleUSBConnection;

void write_register(const SampleUSBConnection &connection, uint32_t address, uint32_t value);
uint32_t read_register(const SampleUSBConnection &connection, uint32_t address);

#endif // METAVISION_HAL_SAMPLE_REGISTER_ACCESS_H
