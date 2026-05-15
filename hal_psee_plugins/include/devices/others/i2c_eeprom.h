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

#include "cstdint"
#include <vector>

#ifdef _MSC_VER
#define NOMINMAX // libusb.h includes windows.h which defines min max macros that we don't want
#endif
#include <libusb.h>

enum class I2cEepromCmd : uint8_t {
    WRITE  = 0xBA,
    READ   = 0xBB,
    STATUS = 0xBC,
};

class I2cEeprom {
public:
    uint8_t dev_addr_;
    uint32_t mem_addr_max_  = 0x1FFFF;
    uint32_t mem_size_      = mem_addr_max_ * 8;
    uint16_t mem_page_size_ = 256;

    I2cEeprom(uint8_t addr);
    ~I2cEeprom();
    int read(libusb_device_handle *dev_handle, uint32_t mem_addr, std::vector<uint8_t> &vread, unsigned int bytes);
    int write(libusb_device_handle *dev_handle, uint32_t mem_addr, std::vector<uint8_t> &vdata);
    int get_status(libusb_device_handle *dev_handle);
};
