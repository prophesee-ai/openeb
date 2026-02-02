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

#include "sample_camera_discovery.h"
#include "internal/sample_register_access.h"
#include "internal/sample_usb_connection.h"

void write_register(const SampleUSBConnection &connection, uint32_t address, uint32_t value) {
    int transferred;
    uint32_t payload[5] = {0x40010102, 12, 0, address, value};
    int result = libusb_bulk_transfer(connection.get_device_handle(), kEvk4EndpointControlOut, reinterpret_cast<unsigned char*>(payload), 5 * sizeof(uint32_t), &transferred, 1000);

    if (result != 0) {
        std::cerr << "Failed to write data to the register. Error code: " << result << " - " << libusb_error_name(result) << std::endl;
        return;
    }

    // and read the response
    uint32_t payload_response[10] = {0};
    result = libusb_bulk_transfer(connection.get_device_handle(), kEvk4EndpointControlIn, reinterpret_cast<unsigned char*>(payload_response), sizeof(payload_response), &transferred, 1000);
    if (result == 0) {
        if (payload[0] != 0x40010102)
            std::cerr << "Write Register Command Response is not a SUCCESS" << std::endl;
    } else {
        std::cerr << "Failed to read the response of write register. " << libusb_error_name(result) << std::endl;
    }
}

uint32_t read_register(const SampleUSBConnection &connection, uint32_t address) {
    int transferred;
    uint32_t payload[5] = {0x10102, 12, 0, address, 1};
    int result = libusb_bulk_transfer(connection.get_device_handle(), kEvk4EndpointControlOut, reinterpret_cast<unsigned char*>(payload), 5 * sizeof(uint32_t), &transferred, 1000);

    if (result != 0) {
        std::cerr << "Failed to read data from the register. Error code: " << result << " - " << libusb_error_name(result) << std::endl;
    }

    // and read the response
    uint32_t payload_response[10] = {0};
    result = libusb_bulk_transfer(connection.get_device_handle(), kEvk4EndpointControlIn, reinterpret_cast<unsigned char*>(payload_response), sizeof(payload_response), &transferred, 1000);
    if (result == 0) {
        if (payload[0] != 0x10102)
            std::cerr << "Read Register Command Response is a FAILURE" << std::endl;
    } else {
        std::cerr << "Failed to read the response of write register. " << libusb_error_name(result) << std::endl;
    }
    return payload_response[4];
}

