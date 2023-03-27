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

#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdlib.h>

#include "metavision/psee_hw_layer/utils/register_map.h"
#include "metavision/psee_hw_layer/utils/regmap_data.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/sdk/base/utils/log.h"

namespace Metavision {

// Dummy Null Stream to redirect Log Registers
class NullStreamBuf : public std::streambuf {
protected:
    int overflow(int c) override;
};

int NullStreamBuf::overflow(int c) {
    return c;
}

NullStreamBuf gNullStream;
std::ostream NullOStream(&gNullStream);

namespace detail {
namespace hal {
template<Metavision::LogLevel Level>
Metavision::LoggingOperation<Level> log_registers(const std::string &file, int line, const std::string &function) {
#ifndef _WIN32
    if (getenv("LOG_REGISTERS")) {
        return Metavision::LoggingOperation<Level>(Metavision::getLogOptions(), PrefixFmt, file, line, function);
    }
#endif
    // Default to NullStream
    return Metavision::LoggingOperation<Level>(Metavision::LogOptions(Level, NullOStream), PrefixFmt, file, line,
                                               function);
}
} // namespace hal
} // namespace detail

#define MV_HAL_LOG_REGISTERS MV_LOG_WRAP(Metavision::detail::hal::log_registers, Metavision::LogLevel::Debug)

RegisterMap::Field::Field(const std::string &n, uint8_t start, uint8_t len, uint32_t default_value,
                          const std::map<std::string, uint32_t> &aliases) :
    name_(n), start_(start), len_(len) {
    init_mask();
    aliases_       = aliases;
    default_value_ = default_value;
}

void RegisterMap::Field::add_alias(const std::string &name, uint32_t value) {
    aliases_[name] = value;
}

const std::string &RegisterMap::Field::get_name() const {
    return name_;
}

void RegisterMap::Field::set_name(const std::string &n) {
    name_ = n;
}

void RegisterMap::Field::set_bitfield_in_value(uint32_t v, uint32_t &register_value) const {
    register_value = (register_value & ~mask_) | ((v << start_) & mask_);
}
uint32_t RegisterMap::Field::get_bitfield_in_value(uint32_t register_value) const {
    return (register_value & mask_) >> start_;
}
void RegisterMap::Field::set_default_bitfield_in_value(uint32_t &register_value) const {
    set_bitfield_in_value(default_value_, register_value);
}
uint32_t RegisterMap::Field::get_alias_value(const std::string &alias) const {
    auto it = aliases_.find(alias);
    if (it == aliases_.end())
        return -1;
    return it->second;
}

void RegisterMap::Field::init_mask() {
    uint32_t m = 0;
    for (auto i = 0; i < len_; ++i) {
        m = (m << 1) | 1;
    }
    m <<= start_;
    mask_ = m;
}

uint8_t RegisterMap::Field::get_start() const {
    return start_;
}

void RegisterMap::Field::set_start(uint8_t start) {
    start_ = start;
    init_mask();
}

uint8_t RegisterMap::Field::get_len() const {
    return len_;
}

void RegisterMap::Field::set_len(uint8_t len) {
    len_ = len;
    init_mask();
}

const RegisterMap::Field *RegisterMap::FieldAccess::get_field() const {
    return field_;
}

RegisterMap::Field *RegisterMap::FieldAccess::get_field() {
    return field_;
}

void RegisterMap::FieldAccess::write_value(uint32_t v) {
    if (field_ && register_) {
        MV_HAL_LOG_REGISTERS() << "Write Register" << register_->get_name() << "Field" << field_->get_name() << std::hex
                               << v << std::dec;
        uint32_t cur_value = register_->read_value();
        field_->set_bitfield_in_value(v, cur_value);
        register_->write_value(cur_value);
    } else {
        if (register_) {
            MV_HAL_LOG_ERROR() << "Write: Invalid field for register" << register_->get_name();
        } else {
            MV_HAL_LOG_ERROR() << "Write: Invalid register";
        }
    }
}

uint32_t RegisterMap::FieldAccess::read_value() const {
    if (field_ && register_) {
        uint32_t cur_value = register_->read_value();
        return field_->get_bitfield_in_value(cur_value);
    }
    MV_HAL_LOG_ERROR() << "Read: Invalid register or field";
    return 0;
}

RegisterMap::FieldAccess::FieldAccess(Register *reg, Field *field) {
    field_    = field;
    register_ = reg;
}

void RegisterMap::RegisterAccess::write_value(uint32_t v) {
    if (register_) {
        register_->write_value(v);
        MV_HAL_LOG_REGISTERS() << "Write Register" << register_->get_name() << std::hex << v << std::dec;
    }
}

void RegisterMap::RegisterAccess::write_value(const std::map<std::string, uint32_t> &bitfields) {
    if (register_) {
        register_->write_value(bitfields);
        for (auto v : bitfields) {
            MV_HAL_LOG_REGISTERS() << "Write Register" << register_->get_name() << "Field" << v.first << v.second;
        }
    }
}
void RegisterMap::RegisterAccess::write_value(const std::pair<std::string, uint32_t> &bitfield) {
    if (register_) {
        register_->write_value(bitfield);
    }
}
void RegisterMap::RegisterAccess::write_value(const std::string &bitfieldname, const std::string &bitfieldvalue) {
    if (register_) {
        register_->write_value(bitfieldname, bitfieldvalue);
    }
}

void RegisterMap::RegisterAccess::write_value(const std::map<const std::string, const std::string> &l) {
    if (register_) {
        for (auto p : l) {
            register_->write_value(p.first, p.second);
        }
    }
}

RegisterMap::RegisterAccess &RegisterMap::RegisterAccess::operator=(uint32_t v) {
    if (register_) {
        (*register_) = v;
        MV_HAL_LOG_REGISTERS() << "Write Register" << register_->get_name() << std::hex << v << std::dec;
    }
    return *this;
}

RegisterMap::FieldAccess &RegisterMap::FieldAccess::operator=(const std::string &alias) {
    if (field_ && register_) {
        register_->write_value(field_->get_name(), alias);
    }
    return *this;
}

RegisterMap::FieldAccess &RegisterMap::FieldAccess::operator=(uint32_t v) {
    if (field_ && register_) {
        register_->write_value(field_->get_name(), v);
    }
    return *this;
}

RegisterMap::RegisterAccess &RegisterMap::RegisterAccess::operator=(const std::map<std::string, uint32_t> &bitfields) {
    for (auto v : bitfields) {
        MV_HAL_LOG_REGISTERS() << "Write Register" << register_->get_name() << "Field" << v.first;
    }
    if (register_) {
        (*register_) = bitfields;
    }
    return *this;
}

bool RegisterMap::RegisterAccess::operator==(const RegisterMap::RegisterAccess &rhs) const {
    return this->register_ == rhs.register_;
}

const RegisterMap::FieldAccess RegisterMap::RegisterAccess::operator[](const std::string &name) const {
    if (register_) {
        return (*register_)[name];
    }
    return RegisterMap::FieldAccess(nullptr, nullptr);
}

RegisterMap::FieldAccess RegisterMap::RegisterAccess::operator[](const std::string &name) {
    if (register_) {
        return (*register_)[name];
    }
    return RegisterMap::FieldAccess(nullptr, nullptr);
}

uint32_t RegisterMap::RegisterAccess::read_value() const {
    if (register_) {
        return register_->read_value();
    }
    return -1;
}

uint32_t RegisterMap::RegisterAccess::get_address() const {
    if (register_) {
        return register_->get_address();
    }
    return -1;
}

std::string RegisterMap::RegisterAccess::get_name() const {
    if (register_) {
        return register_->get_name();
    }
    return std::string();
}

RegisterMap::RegisterAccess::RegisterAccess(Register *reg) : register_(reg) {}

RegisterMap::Register::Register(const std::string &n, uint32_t address, std::initializer_list<Field> l) :
    name_(n), address_(address) {
    for (auto f : l) {
        add_field(f);
    }
}

uint32_t RegisterMap::Register::get_address() const {
    return address_;
}

void RegisterMap::Register::set_address(uint32_t address) {
    address_ = address;
}

const std::string &RegisterMap::Register::get_name() const {
    return name_;
}

void RegisterMap::Register::set_name(const std::string &n) {
    name_ = n;
}

const RegisterMap::FieldAccess RegisterMap::Register::operator[](const std::string &name) {
    auto it = name_to_field_.find(name);
    if (it == name_to_field_.end()) {
        MV_HAL_LOG_ERROR() << "Unknown field" << name << "for register" << this->get_name();
        return FieldAccess(nullptr, nullptr);
    }
    return FieldAccess(this, &it->second);
}

RegisterMap::Register &RegisterMap::Register::add_field(const Field &f) {
    name_to_field_[f.get_name()] = f;
    return *this;
}

void RegisterMap::Register::set_register_map(RegisterMap *register_map) {
    register_map_ = register_map;
}

RegisterMap::Register &RegisterMap::Register::operator=(uint32_t v) {
    if (register_map_) {
        MV_HAL_LOG_REGISTERS() << "Write" << name_ << std::hex << v << std::dec;
        register_map_->write(address_, v);
    }
    return *this;
}

RegisterMap::Register &RegisterMap::Register::operator=(const std::map<std::string, uint32_t> &bitfields) {
    auto val = this->read_value();
    for (auto v : bitfields) {
        auto it = name_to_field_.find(v.first);
        if (it != name_to_field_.end()) {
            it->second.set_bitfield_in_value(v.second, val);
        } else {
            MV_HAL_LOG_ERROR() << "Unknown field" << v.first << "for register" << this->get_name();
        }
    }
    *this = val;
    return *this;
}

void RegisterMap::Register::write_value(uint32_t v) {
    MV_HAL_LOG_REGISTERS() << "Write Register" << this->get_name() << std::hex << v << std::dec;
    *this = v;
}

void RegisterMap::Register::write_value(const std::string fieldname, uint32_t value) {
    this->write_value({{fieldname, value}});
}

void RegisterMap::Register::write_value(const std::map<std::string, uint32_t> &bitfields) {
    *this = bitfields;
}
void RegisterMap::Register::write_value(const std::pair<std::string, uint32_t> &bitfield) {
    *this = std::map<std::string, uint32_t>{bitfield};
}
void RegisterMap::Register::write_value(const std::string &bitfieldname, const std::string &bitfieldvalue) {
    auto it = name_to_field_.find(bitfieldname);
    if (it != name_to_field_.end()) {
        *this = {std::pair<std::string, uint32_t>{bitfieldname, it->second.get_alias_value(bitfieldvalue)}};
    }
}

uint32_t RegisterMap::Register::read_value() const {
    if (register_map_) {
        MV_HAL_LOG_REGISTERS() << "register_map_->read" << name_;
        return register_map_->read(address_);
    }
    return -1;
}
RegisterMap::Field *RegisterMap::Register::bit_to_field(uint32_t bit) {
    for (auto &it : name_to_field_) {
        auto name   = it.first;
        auto &field = it.second;
        if (field.get_start() <= bit && bit < field.get_start() + field.get_len()) {
            return &field;
        }
    }
    return nullptr;
}

RegisterMap::RegisterMap(RegmapData device_regmap_description) {
    set_write_cb([](uint32_t address, uint32_t v) {});
    set_read_cb([](uint32_t address) { return uint32_t(-1); });

    bool is_curreg_valid   = false;
    bool is_curfield_valid = false;
    RegisterMap::Register curreg;
    RegisterMap::Field curfield;
    for (auto sub_desc : device_regmap_description) {
        RegmapElement *curdata = std::get<0>(sub_desc);
        size_t size            = std::get<1>(sub_desc);
        std::string prefix(std::get<2>(sub_desc));
        RegmapElement *data_end = curdata + size;

        for (; curdata != data_end; ++curdata) {
            if (curdata->type == R) {
                if (is_curfield_valid) {
                    curreg.add_field(curfield);
                }
                if (is_curreg_valid) {
                    this->add_register(curreg);
                }
                curreg            = RegisterMap::Register(prefix + curdata->register_data.name,
                                               curdata->register_data.addr + std::get<3>(sub_desc));
                is_curreg_valid   = true;
                is_curfield_valid = false;
            } else if (curdata->type == F) {
                if (is_curfield_valid) {
                    curreg.add_field(curfield);
                }
                is_curfield_valid = true;
                curfield          = RegisterMap::Field(curdata->field_data.name, curdata->field_data.start,
                                              curdata->field_data.len, curdata->field_data.default_value);
            } else if (curdata->type == A) {
                if (is_curfield_valid) {
                    curfield.add_alias(curdata->alias_data.name, curdata->alias_data.value);
                }
            }
        }
        if (is_curfield_valid) {
            curreg.add_field(curfield);
        }
        if (is_curreg_valid) {
            this->add_register(curreg);
        }
    }

    dump();
}

RegisterMap::RegisterAccess RegisterMap::operator[](uint32_t addr) {
    return access(addr_to_register_, addr);
}

const RegisterMap::RegisterAccess RegisterMap::operator[](uint32_t addr) const {
    return access(addr_to_register_, addr);
}

RegisterMap::RegisterAccess RegisterMap::operator[](const std::string &name) {
    return access(name_to_register_, name);
}

const RegisterMap::RegisterAccess RegisterMap::operator[](const std::string &name) const {
    return access(name_to_register_, name);
}

void RegisterMap::add_register(const Register &r) {
    std::shared_ptr<Register> ptr         = std::make_shared<Register>(r);
    addr_to_register_[ptr->get_address()] = ptr;
    name_to_register_[ptr->get_name()]    = ptr;
    ptr->set_register_map(this);
}

void RegisterMap::write(uint32_t address, uint32_t v) {
    if (getenv("LOG_REGISTERS")) {
        std::ostringstream s(std::ostringstream::ate);
        s << "write, 0x" << std::setw(8) << std::setfill('0') << std::hex << address;
        s << ", 0x" << std::setw(8) << std::setfill('0') << std::hex << v;
        MV_HAL_LOG_INFO() << s.str();
    }
    write_cb_(address, v);
}
uint32_t RegisterMap::read(uint32_t address) {
    uint32_t v = read_cb_(address);
    if (getenv("LOG_REGISTERS")) {
        std::ostringstream s(std::ostringstream::ate);
        s << "read, 0x" << std::setw(8) << std::setfill('0') << std::hex << address;
        s << ", 0x" << std::setw(8) << std::setfill('0') << std::hex << v;
        MV_HAL_LOG_INFO() << s.str();
    }
    return v;
}
void RegisterMap::set_write_cb(write_cb_t cb) {
    write_cb_ = cb;
}
void RegisterMap::set_read_cb(read_cb_t cb) {
    read_cb_ = cb;
}

void RegisterMap::dump() {
    for (auto &a : name_to_register_)
        MV_HAL_LOG_REGISTERS() << a.first;
}

} // namespace Metavision
