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

#ifndef METAVISION_HAL_PSEE_HW_REGISTER_H
#define METAVISION_HAL_PSEE_HW_REGISTER_H

#include <string>
#include <cstdint>

#include "metavision/hal/facilities/i_hw_register.h"

namespace Metavision {

class RegisterMap;

class PseeHWRegister : public I_HW_Register {
public:
    PseeHWRegister(const std::shared_ptr<RegisterMap> &map);

    virtual void write_register(uint32_t address, uint32_t v) override;
    virtual uint32_t read_register(uint32_t address) override;
    virtual void write_register(const std::string &address, uint32_t v) override;
    virtual uint32_t read_register(const std::string &address) override;
    virtual void write_register(const std::string &address, const std::string &bitfield, uint32_t v) override;
    virtual uint32_t read_register(const std::string &address, const std::string &bitfield) override;

private:
    std::shared_ptr<RegisterMap> regmap_;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_HW_REGISTER_H
