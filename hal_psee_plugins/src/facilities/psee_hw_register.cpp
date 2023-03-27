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

#include "metavision/psee_hw_layer/facilities/psee_hw_register.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {

PseeHWRegister::PseeHWRegister(const std::shared_ptr<RegisterMap> &map) : regmap_(map) {}

void PseeHWRegister::write_register(uint32_t address, uint32_t v) {
    regmap_->write(address, v);
}

uint32_t PseeHWRegister::read_register(uint32_t address) {
    return regmap_->read(address);
}

void PseeHWRegister::write_register(const std::string &address, uint32_t v) {
    (*regmap_)[address].write_value(v);
}

uint32_t PseeHWRegister::read_register(const std::string &address) {
    return (*regmap_)[address].read_value();
}

void PseeHWRegister::write_register(const std::string &address, const std::string &bitfield, uint32_t v) {
    (*regmap_)[address][bitfield].write_value(v);
}

uint32_t PseeHWRegister::read_register(const std::string &address, const std::string &bitfield) {
    return (*regmap_)[address][bitfield].read_value();
}

} // namespace Metavision
