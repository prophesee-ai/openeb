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

#include <algorithm>
#include "metavision/hal/utils/device_config.h"

namespace Metavision {

DeviceConfigOption::DeviceConfigOption() : type_(Type::Invalid) {}

DeviceConfigOption::DeviceConfigOption(bool default_value) : type_(Type::Boolean), def_val_b_(default_value) {}

DeviceConfigOption::DeviceConfigOption(int min, int max, int default_value) :
    type_(Type::Int), range_i_(min, max), def_val_i_(default_value) {
    if (default_value < min || default_value > max) {
        throw std::runtime_error("default value must be within range");
    }
}

DeviceConfigOption::DeviceConfigOption(double min, double max, double default_value) :
    type_(Type::Double), range_d_(min, max), def_val_d_(default_value) {
    if (default_value < min || default_value > max) {
        throw std::runtime_error("default value must be within range");
    }
}

DeviceConfigOption::DeviceConfigOption(const std::vector<std::string> &values, const std::string &default_value) :
    type_(Type::String), values_(values), def_val_s_(default_value) {
    if (std::find(values.begin(), values.end(), default_value) == values.end()) {
        throw std::runtime_error("default value must be within allowed values");
    }
}

DeviceConfigOption::~DeviceConfigOption() {
    destroy();
}

DeviceConfigOption::DeviceConfigOption(const DeviceConfigOption &opt) {
    copy(opt);
}

DeviceConfigOption &DeviceConfigOption::operator=(const DeviceConfigOption &opt) {
    if (this != &opt) {
        destroy();
        copy(opt);
    }
    return *this;
}

template<typename T>
std::pair<T, T> DeviceConfigOption::get_range() const {
    throw std::runtime_error("get_range called with incompatible type");
}

template<>
std::pair<int, int> DeviceConfigOption::get_range<int>() const {
    if (type_ != Type::Int) {
        throw std::runtime_error("get_range called with incompatible type");
    }
    return range_i_;
}

template<>
std::pair<double, double> DeviceConfigOption::get_range<double>() const {
    if (type_ != Type::Double) {
        throw std::runtime_error("get_range called with incompatible type");
    }
    return range_d_;
}

template<typename T>
T DeviceConfigOption::get_default_value() const {
    throw std::runtime_error("get_default_value called with incompatible type");
}

template<>
bool DeviceConfigOption::get_default_value<bool>() const {
    if (type_ != Type::Boolean) {
        throw std::runtime_error("get_default_value called with incompatible type");
    }
    return def_val_b_;
}

template<>
int DeviceConfigOption::get_default_value<int>() const {
    if (type_ != Type::Int) {
        throw std::runtime_error("get_default_value called with incompatible type");
    }
    return def_val_i_;
}

template<>
double DeviceConfigOption::get_default_value<double>() const {
    if (type_ != Type::Double) {
        throw std::runtime_error("get_default_value called with incompatible type");
    }
    return def_val_d_;
}

template<>
std::string DeviceConfigOption::get_default_value<std::string>() const {
    if (type_ != Type::String) {
        throw std::runtime_error("get_default_value called with incompatible type");
    }
    return def_val_s_;
}

std::vector<std::string> DeviceConfigOption::get_values() const {
    if (type_ != Type::String) {
        throw std::runtime_error("get_values called with incompatible type");
    }
    return values_;
}

DeviceConfigOption::Type DeviceConfigOption::type() const {
    return type_;
}

void DeviceConfigOption::destroy() {
    switch (type_) {
    case Type::Invalid:
    case Type::Boolean:
        break;
    case Type::String:
        using string = std::string;
        using vec    = std::vector<std::string>;
        values_.~vec();
        def_val_s_.~string();
        break;
    case Type::Int:
        using pi = std::pair<int, int>;
        range_i_.~pi();
        break;
    case Type::Double:
        using pd = std::pair<double, double>;
        range_d_.~pd();
        break;
    }
}

void DeviceConfigOption::copy(const DeviceConfigOption &opt) {
    type_ = opt.type_;
    switch (type_) {
    case Type::Invalid:
    case Type::Boolean:
        def_val_b_ = opt.def_val_b_;
        break;
    case Type::String:
        new (&values_) std::vector<std::string>(opt.values_);
        new (&def_val_s_) std::string(opt.def_val_s_);
        break;
    case Type::Int:
        new (&range_i_) std::pair<int, int>(opt.range_i_);
        def_val_i_ = opt.def_val_i_;
        break;
    case Type::Double:
        new (&range_d_) std::pair<double, double>(opt.range_d_);
        def_val_d_ = opt.def_val_d_;
        break;
    }
}

std::ostream &operator<<(std::ostream &os, const DeviceConfigOption &opt) {
    switch (opt.type_) {
    case DeviceConfigOption::Type::Invalid:
        os << "default: N/A";
        break;
    case DeviceConfigOption::Type::Boolean:
        os << "default: " << opt.def_val_b_ << " values: true | false";
        break;
    case DeviceConfigOption::Type::String:
        os << "default: " << opt.def_val_s_ << " values: ";
        if (!opt.values_.empty()) {
            os << opt.values_[0];
            size_t s = opt.values_.size();
            for (size_t i = 1; i < s; ++i) {
                os << " | " << opt.values_[i];
            }
        }
        break;
    case DeviceConfigOption::Type::Int:
        os << "default: " << opt.def_val_i_ << " range: [" << opt.range_i_.first << "," << opt.range_i_.second << "]";
        break;
    case DeviceConfigOption::Type::Double:
        os << "default: " << opt.def_val_d_ << " range: [" << opt.range_d_.first << "," << opt.range_d_.second << "]";
        break;
    }
    return os;
}

std::string DeviceConfig::get_format_key() {
    return "format";
}

std::string DeviceConfig::format() const {
    return get<std::string>(get_format_key());
}

void DeviceConfig::set_format(const std::string &format) {
    set(get_format_key(), format);
}

std::string DeviceConfig::get_biases_range_check_bypass_key() {
    return "ll_biases_range_check_bypass";
}

bool DeviceConfig::biases_range_check_bypass() const {
    return get<bool>(get_biases_range_check_bypass_key(), false);
}

void DeviceConfig::enable_biases_range_check_bypass(bool enabled) {
    return set(get_biases_range_check_bypass_key(), enabled);
}

void DeviceConfig::set(const std::string &key, bool value) {
    map[key] = value ? "true" : "false";
}

void DeviceConfig::set(const std::string &key, const char *const value) {
    map[key] = std::string(value);
}

void DeviceConfig::set(const std::string &key, const std::string &value) {
    map[key] = value;
}

std::string DeviceConfig::get(const std::string &key, const std::string &def) const {
    auto it = map.find(key);
    if (it != map.end()) {
        return it->second;
    }
    return def;
}

std::ostream &operator<<(std::ostream &os, const DeviceConfig &conf) {
    for (const auto &p : conf.map) {
        os << p.first << ": " << p.second << std::endl;
    }
    return os;
}

} // namespace Metavision