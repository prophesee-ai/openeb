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

#include "metavision/psee_hw_layer/devices/treuzell/tz_regmap_device.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {

TzDeviceWithRegmap::TzDeviceWithRegmap(RegmapData regmap_data, std::string root) :
    register_map(std::make_shared<RegisterMap>(regmap_data)), root_(root) {
    register_map->set_read_cb([this](uint32_t address) {
        load_register(address);
        return read_register(address);
    });
    register_map->set_write_cb([this](uint32_t address, uint32_t v) { write_register(address, v); });
}

TzDeviceWithRegmap::TzDeviceWithRegmap(RegmapElement *regarray, uint32_t size, std::string root) :
    TzDeviceWithRegmap(
        {
            std::make_tuple(regarray, size, "", 0),
        },
        root) {}

void TzDeviceWithRegmap::write_register(Register_Addr register_addr, uint32_t value) {
    init_register(register_addr, value);
    send_register(register_addr);
}

uint32_t TzDeviceWithRegmap::read_register(Register_Addr regist) {
    auto it = mregister_state.find(regist);
    if (it == mregister_state.end()) {
        return 0;
    }

    return it->second;
}

void TzDeviceWithRegmap::load_register(Register_Addr regist) {
    init_register(regist, cmd->read_device_register(tzID, regist)[0]);
}

void TzDeviceWithRegmap::set_register_bit(Register_Addr register_addr, int idx, bool state) {
    auto it = mregister_state.find(register_addr);
    if (it == mregister_state.end()) {
        it = mregister_state.insert(std::make_pair(register_addr, static_cast<uint32_t>(0))).first;
    }
    if (state) {
        it->second |= (1 << idx);
    } else {
        it->second &= ~(1 << idx);
    }
}

void TzDeviceWithRegmap::send_register(Register_Addr register_addr) {
    uint32_t val = 0;
    if (has_register(register_addr)) {
        val = read_register(register_addr);
    }
    cmd->write_device_register(tzID, register_addr, std::vector<uint32_t>(1, val));
}

void TzDeviceWithRegmap::send_register_bit(Register_Addr register_addr, int idx, bool state) {
    set_register_bit(register_addr, idx, state);
    send_register(register_addr);
}

uint32_t TzDeviceWithRegmap::read_register_bit(Register_Addr register_addr, int idx) {
    MV_HAL_LOG_DEBUG() << __PRETTY_FUNCTION__ << register_addr;
    auto it = mregister_state.find(register_addr);
    if (it == mregister_state.end()) {
        return 0;
    }

    return (it->second >> idx) & 1;
}

void TzDeviceWithRegmap::init_register(Register_Addr regist, uint32_t value) {
    mregister_state[regist] = value;
}

bool TzDeviceWithRegmap::has_register(Register_Addr regist) {
    auto it = mregister_state.find(regist);
    return it != mregister_state.end();
}

RegisterMap &TzDeviceWithRegmap::regmap() {
    return *register_map;
}

} // namespace Metavision
