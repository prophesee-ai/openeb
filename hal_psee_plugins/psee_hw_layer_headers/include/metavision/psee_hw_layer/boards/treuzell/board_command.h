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

#ifndef METAVISION_HAL_BOARD_COMMAND_H
#define METAVISION_HAL_BOARD_COMMAND_H

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <ctime>

#include "metavision/hal/utils/data_transfer.h"

namespace Metavision {

class TzCtrlFrame;

class BoardCommand {
public:
    virtual std::vector<uint32_t> read_device_register(uint32_t device, uint32_t address, int nval = 1)           = 0;
    virtual void write_device_register(uint32_t device, uint32_t address, const std::vector<uint32_t> &val)       = 0;
    virtual void transfer_tz_frame(TzCtrlFrame &req)                                                              = 0;
    virtual uint32_t get_device_count()                                                                           = 0;
    virtual std::string get_name()                                                                                = 0;
    virtual std::string get_serial()                                                                              = 0;
    virtual std::unique_ptr<DataTransfer::RawDataProducer> build_raw_data_producer(uint32_t raw_event_size_bytes) = 0;
    virtual long get_board_speed()                                                                                = 0;
    virtual std::string get_manufacturer()                                                                        = 0;
    virtual uint32_t get_version()                                                                                = 0;
    virtual time_t get_build_date()                                                                               = 0;
};
} // namespace Metavision
#endif // METAVISION_HAL_BOARD_COMMAND_H
