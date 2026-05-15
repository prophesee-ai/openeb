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

#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>


#include <metavision/hal/utils/hal_exception.h>
#include "sample_erc.h"
#include "internal/sample_register_access.h"

constexpr uint32_t CD_EVENT_COUNT_DEFAULT = 4000;

SampleErc::SampleErc(std::shared_ptr<SampleUSBConnection> usb_connection) :
    usb_connection_(usb_connection) {}
// we don't initialize the LUT of the ERC module and rely on default values

bool SampleErc::enable(bool en) {
    // we only write t_dropping_en (time dropping), not horizontal/vertical dropping
    if (en) {
        write_register(*usb_connection_, 0x00006050, 0x01);
    } else {
        write_register(*usb_connection_, 0x00006050, 0x00);
    }

    return true;
}

bool SampleErc::is_enabled() const {
    // we only read t_dropping_en (time dropping), not horizontal/vertical dropping
    auto enabled = read_register(*usb_connection_, 0x00006050);
    if (enabled == 0)
        return false;
    else
        return true;
}

uint32_t SampleErc::get_count_period() const {
    return read_register(*usb_connection_, 0x00006008);
}

uint32_t SampleErc::get_cd_event_count() const {
    return read_register(*usb_connection_, 0x0000600C);
}

bool SampleErc::set_cd_event_count(uint32_t count) {
    write_register(*usb_connection_, 0x0000600C, count);
    return true;
}

uint32_t SampleErc::get_min_supported_cd_event_count() const {
    return 0;
}

uint32_t SampleErc::get_max_supported_cd_event_count() const {
    return CD_EVENT_COUNT_MAX;
}

void SampleErc::erc_from_file(const std::string &file_path) {
    // not implemented
}
