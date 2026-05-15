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

#ifndef TZ_REGMAP_DEVICE_H
#define TZ_REGMAP_DEVICE_H

#include <cstdint>
#include <memory>
#include <map>
#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {

class TzLibUSBBoardCommand;
class TzHwRegister;

class TzDeviceWithRegmap : public virtual TzDevice {
    using Register_Addr = uint32_t;

public:
    using RegmapData = RegisterMap::RegmapData;
    /// @brief Writes shadow register (value stored on computer side)
    void write_register(Register_Addr regist, uint32_t value);

    /// @brief Reads shadow register (value stored on computer side)
    /// @return The value of the register
    uint32_t read_register(Register_Addr regist);

    /// @brief Loads the register on the board side with the value stored on computer
    /// @return Nothing to express that the method loads the value from the board and stores it
    void load_register(Register_Addr regist);

    void set_register_bit(Register_Addr regist, int idx, bool state);
    void send_register(Register_Addr regist);
    void send_register_bit(Register_Addr regist, int idx, bool state);
    uint32_t read_register_bit(Register_Addr register_addr, int idx);
    void init_register(Register_Addr regist, uint32_t value);

    RegisterMap &regmap();

protected:
    TzDeviceWithRegmap(RegmapElement *, uint32_t size, std::string root);
    TzDeviceWithRegmap(RegmapData, std::string root);
    std::map<Register_Addr, uint32_t> mregister_state;
    std::shared_ptr<RegisterMap> register_map;
    friend class TzHwRegister;

private:
    bool has_register(Register_Addr regist);
    std::string root_;
};

} // namespace Metavision
#endif /* TZ_REGMAP_DEVICE_H */
