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

#ifndef METAVISION_HAL_DEVICE_CONFIG_H
#define METAVISION_HAL_DEVICE_CONFIG_H

#include <sstream>
#include <vector>
#include <string>
#include <map>

namespace Metavision {

/// @brief Class representing options allowed for a device configuration entry
///
/// For each key available in a @ref DeviceConfig as reported by @ref DeviceDiscovery::list_device_config_options"",
/// there is a corresponding option listing the type and accepted values when opening the Device via @ref
/// DeviceDiscovery::open
/// It is composed of a type (boolean, numeric or string) and a set of accepted values
///
/// @note The @ref DeviceConfig class is string based to keep its usage simple and generic, whereas this class
///       represent a type and optional range/set of values. The added information can be useful to guide the user
///       when presenting the options that are available, but ultimately the value set for a specific key will be a
///       string, so care must be taken when converting the value before calling one of the functions to set a
///       (key,value) pair in the @ref DeviceConfig
/// @sa @ref DeviceConfig
class DeviceConfigOption {
public:
    /// @brief Enumeration class representing the type of an option
    enum class Type { Invalid, Boolean, Int, Double, String };

    DeviceConfigOption();
    DeviceConfigOption(bool default_value);
    DeviceConfigOption(int min, int max, int default_value);
    DeviceConfigOption(double min, double max, double default_value);
    DeviceConfigOption(const std::vector<std::string> &values, const std::string &default_value);

    DeviceConfigOption(const DeviceConfigOption &opt);
    DeviceConfigOption &operator=(const DeviceConfigOption &opt);

    ~DeviceConfigOption();

    /// @brief Gets the range of accepted values
    /// @return The range of accepted values for options representing a numeric type (Int or Double)
    template<typename T>
    std::pair<T, T> get_range() const;

    /// @brief Gets the set of accepted values
    /// @return The set of accepted values for options representing a string type
    std::vector<std::string> get_values() const;

    /// @brief Gets the set of accepted values
    /// @return The default value
    template<typename T>
    T get_default_value() const;

    /// @brief Gets the type of this class
    /// @return The type that is represented by this class
    Type type() const;

private:
    Type type_;
    // an alternative is to use a std::variant, but only in C++17
    union {
        std::pair<int, int> range_i_;
        std::pair<double, double> range_d_;
        std::vector<std::string> values_;
    };
    // an alternative is to use a std::variant, but only in C++17
    union {
        bool def_val_b_;
        int def_val_i_;
        double def_val_d_;
        std::string def_val_s_;
    };

    void destroy();
    void copy(const DeviceConfigOption &opt);
    friend std::ostream &operator<<(std::ostream &os, const DeviceConfigOption &opt);
};

using DeviceConfigOptionMap = std::map<std::string, DeviceConfigOption>;

/// @brief Class storing a map of (key,values) that can be used to customize how a device should be opened
///
/// @warning The class stores values as string for ease of use, so proper care should be taken to make sure
///          that the value can be properly parsed when calling one of the functions to get the (typed) value
/// @sa @ref DeviceDiscovery::list_device_config_options
/// @sa @ref DeviceDiscovery::open
class DeviceConfig {
public:
    static std::string get_format_key();

    /// @brief Gets the event format
    /// @return string representing current event format setting
    std::string format() const;

    void set_format(const std::string &format);

    static std::string get_biases_range_check_bypass_key();

    /// @brief Gets the status of the "bypass biases range check" option
    /// @return true if biases range check should be bypassed, false otherwise
    bool biases_range_check_bypass() const;

    void enable_biases_range_check_bypass(bool enabled);

    /// @brief Sets a value for a named key in the config dictionary
    /// @param key Key of the config
    /// @param value Value of the config
    template<typename T>
    void set(const std::string &key, const T &value) {
        map[key] = std::to_string(value);
    }

    /// @brief Sets a value for a named key in the config dictionary
    /// @overload
    void set(const std::string &key, bool value);

    /// @brief Sets a value for a named key in the config dictionary
    /// @overload
    void set(const std::string &key, const char *const value);

    /// @brief Sets a value for a named key in the config dictionary
    /// @overload
    void set(const std::string &key, const std::string &value);

    /// @brief Gets the (typed) value for a named key in the config dictionary if it exists and can be extracted from
    ///        the corresponding value safely or the provided default value
    /// @tparam T type of the value to return as
    /// @param key Name of the config key
    /// @param def Default value if the config key is not found
    /// @return Value of the config
    template<typename T>
    T get(const std::string &key, const T &def = T()) const {
        return get(key, def, 0);
    }

    /// @brief Gets the (typed) value for a named key in the config dictionary if it exists and can be extracted from
    ///        the corresponding value safely or the provided default value
    /// @param key Name of the config key
    /// @param def Default value if the config key is not found
    /// @return Value of the config
    /// @overload
    std::string get(const std::string &key, const std::string &def = std::string()) const;

    [[deprecated(
        "This function is deprecated since version 4.1.0. Please use get_format_key() instead.")]] static std::string
        get_evt_format_key();
    [[deprecated("This function is deprecated since version 4.1.0. Please use format() instead.")]] std::string
        evt_format() const;
    [[deprecated("This function is deprecated since version 4.1.0. Please use set_format() instead.")]] void
        set_evt_format(const std::string &);

private:
    // private get<T> helper, to avoid template specialization errors on GCC
    template<typename T, typename U = typename std::enable_if<!std::is_same<std::string, T>::value>::type>
    T get(const std::string &key, const T &def, int) const {
        T t(def);
        auto it = map.find(key);
        if (it != map.end()) {
            std::istringstream iss(it->second);
            iss >> std::boolalpha >> t;
        }
        return t;
    }

    // private get<T> helper, to avoid template specialization errors on GCC
    std::string get(const std::string &key, const std::string &def, int) const {
        return get(key, def);
    }

    std::map<std::string, std::string> map;
    friend std::ostream &operator<<(std::ostream &os, const DeviceConfig &opt);
};
} // namespace Metavision

#endif // METAVISION_HAL_DEVICE_CONFIG_H
