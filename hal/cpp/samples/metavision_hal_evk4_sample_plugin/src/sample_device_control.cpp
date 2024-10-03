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

#include "sample_device_control.h"
#include "internal/sample_register_access.h"
#include <iostream>
#include <iomanip>
#include <thread>

SampleDeviceControl::SampleDeviceControl(std::shared_ptr<SampleUSBConnection> usb_connection) :
    usb_connection_(usb_connection) {}

void SampleDeviceControl::reset() {}

void SampleDeviceControl::start() {
    // do the START sequence
    // Digital START
    write_register(*usb_connection_, 0x0000B000, 0x000002F9);
    write_register(*usb_connection_, 0x00009028, 0x00000000);
    write_register(*usb_connection_, 0x00009008, 0x00000645);
    // Analog START
    write_register(*usb_connection_, 0x0000002C, 0x0022C724);
    write_register(*usb_connection_, 0x00000004, 0xF0005442);
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

void SampleDeviceControl::stop() {}