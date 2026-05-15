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

#include "devices/others/i2c_eeprom.h"
#include "metavision/hal/utils/hal_log.h"
#include <iostream>
#include <iomanip>

#define VCMD_SET LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE
#define VCMD_GET LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE

I2cEeprom::I2cEeprom(uint8_t addr) {
    dev_addr_ = addr;
}

I2cEeprom::~I2cEeprom() {}

int I2cEeprom::read(libusb_device_handle *dev_handle, uint32_t mem_addr, std::vector<uint8_t> &vread,
                    unsigned int bytes) {
    if (mem_addr > mem_addr_max_) {
        MV_LOG_ERROR() << Metavision::Log::no_space << "I2C EEPROM address 0x" << std::hex << mem_addr << std::dec
                       << " is out of range.";
        return -1;
    }

    if (mem_addr + bytes > (mem_addr_max_ + 1)) {
        MV_LOG_ERROR() << "EEPROM data to read exceed memory size (roll-over safety).";
        MV_LOG_INFO() << Metavision::Log::no_space << "Selected base address:           0x" << std::hex << std::setw(5)
                      << std::setfill('0') << mem_addr;
        MV_LOG_INFO() << Metavision::Log::no_space << "Requested bytes count:           " << bytes;
        MV_LOG_INFO() << Metavision::Log::no_space << "Memory max address:              0x" << std::hex << std::setw(5)
                      << std::setfill('0') << mem_addr_max_;
        MV_LOG_INFO() << Metavision::Log::no_space
                      << "Memory size (from base address): " << (mem_addr_max_ - mem_addr + 1) << " byte(s)";
        MV_LOG_INFO() << Metavision::Log::no_space << "Memory total size:               " << mem_size_ << " bytes";
        return -1;
    }

    // Memory address is encoded using 17 bits. Bit 16 is given during I2C device address phase using bit 1.
    // I2C device select code is composed as followed:
    //   - Bits 8-2: I2C device address
    //   - Bit 1:    Memory address bit 16.
    //   - Bit 0:    Mode selection (R/W).
    uint8_t dev_sel_code  = dev_addr_ | ((mem_addr >> 16) & 0x1);
    uint16_t sub_mem_addr = (mem_addr & 0xFFFF);
    vread.resize(bytes);

    // USB vendor specific control transfer will forward all I2C transfer parameters to CX3
    // using wIndex to pass I2C device select code, wValue for Memory address and wLength for bytes count.
    int r = libusb_control_transfer(dev_handle, VCMD_GET, (uint8_t)I2cEepromCmd::READ, dev_sel_code, sub_mem_addr,
                                    &vread[0], bytes, 0);
    if (r <= 0) {
        MV_LOG_ERROR() << "I2C EEPROM read error:" << libusb_error_name(r);

        if (r == LIBUSB_ERROR_PIPE) {
            // Stall packet was send by the device indicating an error during I2C transfer
            get_status(dev_handle);
        }
        return -1;
    }
    return 0;
}

int I2cEeprom::write(libusb_device_handle *dev_handle, uint32_t mem_addr, std::vector<uint8_t> &vdata) {
    unsigned int bytes = vdata.size();

    if (mem_addr > mem_addr_max_) {
        MV_LOG_ERROR() << Metavision::Log::no_space << "I2C EEPROM address 0x" << std::hex << mem_addr << std::dec
                       << " is out of range.";
        return -1;
    }

    if (bytes > mem_page_size_) {
        MV_LOG_ERROR() << Metavision::Log::no_space << "I2C EEPROM page size cannot exceed " << mem_page_size_
                       << " bytes. (Found " << bytes << ")";
        return -1;
    }

    uint8_t page_addr = mem_addr & 0xFF;

    if (page_addr + bytes > 256) {
        MV_LOG_ERROR() << "I2C EEPROM data to write exceed page size (roll-over safety).";
        MV_LOG_INFO() << Metavision::Log::no_space << "Selected page address: 0x" << std::hex << std::setw(5)
                      << std::setfill('0') << mem_addr;
        MV_LOG_INFO() << Metavision::Log::no_space << "Requested bytes count: " << bytes;
        MV_LOG_INFO() << Metavision::Log::no_space << "Page base address:     0x" << std::hex << std::setw(5)
                      << std::setfill('0') << (mem_addr & 0x1FF00);
        MV_LOG_INFO() << Metavision::Log::no_space << "Page max address:      0x" << std::hex << std::setw(5)
                      << std::setfill('0') << ((mem_addr & 0x1FF00) + mem_page_size_ - 1);
        MV_LOG_INFO() << Metavision::Log::no_space << "Page size:             " << mem_page_size_ << " bytes";
        return -1;
    }

    // Memory address is encoded using 17 bits. Bit 16 is given during I2C device address phase using bit 1.
    // I2C device select code is composed as followed:
    //   - Bits 8-2: I2C device address
    //   - Bit 1:    Memory address bit 16.
    //   - Bit 0:    Mode selection (R/W).
    uint8_t dev_sel_code  = dev_addr_ | ((mem_addr >> 16) & 0x01);
    uint16_t sub_mem_addr = (mem_addr & 0xFFFF);

    int r = libusb_control_transfer(dev_handle, VCMD_SET, (uint8_t)I2cEepromCmd::WRITE, dev_sel_code, sub_mem_addr,
                                    &vdata[0], (uint16_t)bytes, 0);

    if (r <= 0) {
        MV_LOG_ERROR() << "I2C EEPROM write error:" << libusb_error_name(r);
        return -1;
    } else {
        if (static_cast<unsigned int>(r) != bytes) {
            MV_LOG_ERROR() << "I2C EEPROM write error. Not all bytes were received by the device.";
            return -1;
        }
        if (get_status(dev_handle) != 0) {
            return -1;
        }
    }
    return 0;
}

int I2cEeprom::get_status(libusb_device_handle *dev_handle) {
    uint8_t status;

    int r = libusb_control_transfer(dev_handle, VCMD_GET, (uint8_t)I2cEepromCmd::STATUS, dev_addr_, 0, &status, 1, 0);

    if (r <= 0) {
        MV_LOG_ERROR() << "I2C EEPROM status error:" << libusb_error_name(r);
        return -1;
    }

    if (status != 0) {
        MV_LOG_ERROR() << Metavision::Log::no_space << "I2C driver error code: 0x" << std::hex << (status & 0xFF);
    }
    r = status;

    return r;
}