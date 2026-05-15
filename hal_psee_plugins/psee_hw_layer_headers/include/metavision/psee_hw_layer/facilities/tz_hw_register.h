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

#ifndef TZ_HW_REGISTER_
#define TZ_HW_REGISTER_

#include <memory>
#include <vector>

#include "metavision/psee_hw_layer/devices/treuzell/tz_regmap_device.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/facilities/i_hw_register.h"

namespace Metavision {

class RegisterMap;

class TzHwRegister : public I_HW_Register {
public:
    TzHwRegister(std::vector<std::shared_ptr<TzDevice>> &devices);

    virtual void write_register(uint32_t address, uint32_t v) override;
    virtual uint32_t read_register(uint32_t address) override;
    virtual void write_register(const std::string &address, uint32_t v) override;
    virtual uint32_t read_register(const std::string &address) override;
    virtual void write_register(const std::string &address, const std::string &bitfield, uint32_t v) override;
    virtual uint32_t read_register(const std::string &address, const std::string &bitfield) override;

private:
    std::vector<std::shared_ptr<TzDeviceWithRegmap>> regmap_vec_;

    uint32_t get_register_address(uint32_t address);
};

} // namespace Metavision
#endif /* TZ_HW_REGISTER_ */
