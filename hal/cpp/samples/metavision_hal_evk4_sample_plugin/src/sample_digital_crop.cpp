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

#include <iostream>

#include <metavision/hal/utils/hal_exception.h>
#include "sample_digital_crop.h"
#include "internal/sample_register_access.h"

SampleDigitalCrop::SampleDigitalCrop(std::shared_ptr<SampleUSBConnection> usb_connection) :
    usb_connection_(usb_connection) {}


bool SampleDigitalCrop::enable(bool state) {
    if (state) {
        write_register(*usb_connection_, 0x900C, 0x05);
    } else{
        write_register(*usb_connection_, 0x900C, 0x00);
    }
    return true;
}

bool SampleDigitalCrop::is_enabled() const {
    if (read_register(*usb_connection_, 0x900C) == 0)
        return false;
    else
        return true;
}

uint32_t combineInts(uint32_t lowBytes, uint32_t highBytes) {
    uint32_t result = (lowBytes & 0xFFFF) | ((highBytes & 0xFFFF) << 16);
    return result;
}

bool SampleDigitalCrop::set_window_region(const Region &region, bool reset_origin) {
    uint32_t start_x, start_y, end_x, end_y;
    std::tie(start_x, start_y, end_x, end_y) = region;
    write_register(*usb_connection_, 0x9010, combineInts(start_x,start_y));
    write_register(*usb_connection_, 0x9014, combineInts(end_x,end_y));
    return true;
}

void splitInt(uint32_t input, uint32_t &lowBytes, uint32_t &highBytes) {
    lowBytes = input & 0xFFFF;
    highBytes = (input >> 16) & 0xFFFF;
}

SampleDigitalCrop::Region SampleDigitalCrop::get_window_region() const {
    uint32_t start_x, start_y, end_x, end_y;
    splitInt(read_register(*usb_connection_, 0x9010), start_x, start_y);
    splitInt(read_register(*usb_connection_, 0x9014), end_x, end_y);
    return {start_x, start_y, end_x, end_y};
}

