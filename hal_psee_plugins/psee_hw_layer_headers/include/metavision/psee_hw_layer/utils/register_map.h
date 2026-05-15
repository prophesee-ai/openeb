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

#ifndef METAVISION_HAL_REGISTER_MAP_H
#define METAVISION_HAL_REGISTER_MAP_H

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <iostream>

#include "metavision/hal/utils/hal_log.h"

struct RegmapElement;

namespace Metavision {

class RegisterMap {
public:
    class Field {
    public:
        Field()              = default;
        Field(const Field &) = default;
        Field(const std::string &n, uint8_t start, uint8_t len, uint32_t default_value = 0,
              const std::map<std::string, uint32_t> &aliases = {});

        void add_alias(const std::string &name, uint32_t value);
        uint8_t get_start() const;
        uint8_t get_len() const;
        void set_start(uint8_t b);
        void set_len(uint8_t b);
        const std::string &get_name() const;
        void set_name(const std::string &n);

        void set_bitfield_in_value(uint32_t v, uint32_t &register_value) const;
        uint32_t get_bitfield_in_value(uint32_t register_value) const;
        void set_default_bitfield_in_value(uint32_t &register_value) const;
        uint32_t get_alias_value(const std::string &alias) const;

    private:
        void init_mask();
        std::string name_;
        uint8_t start_;
        uint8_t len_;
        uint32_t mask_          = 0;
        uint32_t default_value_ = 0;
        std::map<std::string, uint32_t> aliases_;
    };

    class Register;
    class RegisterAccess;
    class FieldAccess {
    public:
        friend class Register;
        friend class RegisterAccess;
        void write_value(uint32_t v);
        Field *get_field();
        const Field *get_field() const;
        uint32_t read_value() const;
        FieldAccess &operator=(const std::string &alias);
        FieldAccess &operator=(uint32_t v);

    private:
        FieldAccess(Register *reg, Field *field);
        Field *field_;
        Register *register_;
    };

    class RegisterAccess {
    public:
        friend class RegisterMap;

        void write_value(uint32_t v);
        void write_value(const std::map<std::string, uint32_t> &bitfields);
        void write_value(const std::pair<std::string, uint32_t> &bitfield);
        void write_value(const std::string &bitfieldname, const std::string &bitfieldvalue);
        void write_value(const std::map<const std::string, const std::string> &bitfield);

        RegisterAccess &operator=(uint32_t v);
        RegisterAccess &operator=(const std::map<std::string, uint32_t> &bitfields);
        bool operator==(const RegisterAccess &rhs) const;

        FieldAccess operator[](const std::string &name);
        const FieldAccess operator[](const std::string &name) const;

        uint32_t read_value() const;
        uint32_t get_address() const;
        std::string get_name() const;

    private:
        RegisterAccess(Register *field);
        Register *register_;
    };

    class Register {
    public:
        Register()                 = default;
        Register(const Register &) = default;
        Register(const std::string &n, uint32_t address, std::initializer_list<Field> l = {});

        uint32_t get_address() const;
        void set_address(uint32_t addr);

        const std::string &get_name() const;
        void set_name(const std::string &n);

        void set_register_map(RegisterMap *register_map);
        Register &add_field(const Field &f);

        void write_value(uint32_t v);
        void write_value(const std::string fieldname, uint32_t value);
        void write_value(const std::map<std::string, uint32_t> &bitfields);
        void write_value(const std::pair<std::string, uint32_t> &bitfield);
        void write_value(const std::string &bitfieldname, const std::string &bitfieldvalue);

        Register &operator=(uint32_t v);
        Register &operator=(const std::map<std::string, uint32_t> &bitfields);

        const FieldAccess operator[](const std::string &name);

        uint32_t read_value() const;
        Field *bit_to_field(uint32_t bit);

    private:
        std::string name_;
        uint32_t address_          = 0;
        RegisterMap *register_map_ = nullptr;
        std::map<std::string, Field> name_to_field_;
    };
    using RegmapData = std::vector<std::tuple<RegmapElement *, uint32_t, std::string, uint32_t>>;
    RegisterMap(RegmapData);

    RegisterAccess operator[](uint32_t addr);
    const RegisterAccess operator[](uint32_t addr) const;

    RegisterAccess operator[](const std::string &name);
    const RegisterAccess operator[](const std::string &name) const;

    void add_register(const Register &r);

    void write(uint32_t address, uint32_t v);
    uint32_t read(uint32_t address);

    typedef std::function<void(uint32_t address, uint32_t v)> write_cb_t;
    typedef std::function<uint32_t(uint32_t address)> read_cb_t;
    void set_write_cb(write_cb_t cb);
    void set_read_cb(read_cb_t cb);

    void dump();

private:
    template<class U>
    static RegisterAccess access(U &container, typename U::key_type addr) {
        auto it = container.find(addr);
        if (it == container.end()) {
            MV_HAL_LOG_ERROR() << "Unknown register address" << addr;
            return RegisterMap::RegisterAccess(nullptr);
        }
        return RegisterAccess(it->second.get());
    }

    write_cb_t write_cb_;
    read_cb_t read_cb_;
    std::map<uint32_t, std::shared_ptr<Register>> addr_to_register_;
    std::map<std::string, std::shared_ptr<Register>> name_to_register_;
};

} // namespace Metavision

#endif // METAVISION_HAL_REGISTER_MAP_H
