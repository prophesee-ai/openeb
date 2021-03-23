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

#ifndef METAVISION_HAL_I_HW_REGISTER_H
#define METAVISION_HAL_I_HW_REGISTER_H

#include <string>
#include <cstdint>

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Interface facility for writing/reading hardware registers
class I_HW_Register : public I_RegistrableFacility<I_HW_Register> {
public:
    /// @brief Writes register
    /// @param address Address of the register to write
    /// @param v Value to write
    virtual void write_register(uint32_t address, uint32_t v) = 0;

    /// @brief Writes register
    /// @param address Address of the register to write
    /// @param v Value to write
    virtual void write_register(const std::string &address, uint32_t v) = 0;

    /// @brief Reads register
    /// @param address Address of the register to read
    /// @return Value read
    virtual uint32_t read_register(uint32_t address) = 0;

    /// @brief Reads register
    /// @param address Address of the register to read
    /// @return Value read
    virtual uint32_t read_register(const std::string &address) = 0;

    /// @brief Writes register
    /// @param address Address of the register to write
    /// @param bitfield Bit field of the register to write
    /// @param v Value to write
    virtual void write_register(const std::string &address, const std::string &bitfield, uint32_t v) = 0;

    /// @brief Reads register
    /// @param address Address of the register to read
    /// @param bitfield Bit field of the register to read
    /// @return Value read
    virtual uint32_t read_register(const std::string &address, const std::string &bitfield) = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_HW_REGISTER_H
