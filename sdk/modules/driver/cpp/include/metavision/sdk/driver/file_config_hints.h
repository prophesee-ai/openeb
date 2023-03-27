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

#ifndef METAVISION_SDK_DRIVER_FILE_CONFIG_HINTS_H
#define METAVISION_SDK_DRIVER_FILE_CONFIG_HINTS_H

#include <string>
#include <sstream>
#include <unordered_map>

namespace Metavision {

/// @brief Class represented by a map of string key/value pair, used to control how a file is read by the @ref Camera
///        class
///
/// @note Some keys can be ignored when not applicable to the file format
class FileConfigHints {
public:
    static std::string get_real_time_playback_key() {
        return "real_time_playback";
    }

    static std::string get_time_shift_key() {
        return "time_shift";
    }

    static std::string get_max_memory_key() {
        return "max_memory";
    }

    static std::string get_max_read_per_op_key() {
        return "max_read_per_op";
    }

    /// @brief Constructor
    ///
    /// By default, if applicable, the file will be read using a maximum memory footprint of 12Mo,
    /// with real time playback speed and time shifting enabled
    explicit FileConfigHints() {
        map[get_time_shift_key()]         = std::to_string(true);
        map[get_real_time_playback_key()] = std::to_string(true);
        map[get_max_read_per_op_key()]    = std::to_string(1024 * 1024 * 4);
        map[get_max_memory_key()]         = std::to_string(3 * 1024 * 1024 * 4);
    }

    /// @brief Gets the real-time playback status
    /// @return true if enabled, false otherwise
    bool real_time_playback() const {
        return get<bool>(get_real_time_playback_key(), false);
    }

    /// @brief Named constructor for the real time playback status
    /// @param enabled true if the setting should be enabled, false otherwise
    /// @return FileConfigHints& Reference to the modified config
    FileConfigHints &real_time_playback(bool enabled) {
        map[get_real_time_playback_key()] = std::to_string(enabled);
        return *this;
    }

    /// @brief Gets the timeshift status
    /// @return true if enabled, false otherwise
    bool time_shift() const {
        return get<bool>(get_time_shift_key(), true);
    }

    /// @brief Named constructor for the time shift status
    /// @param enabled true if the setting should be enabled, false otherwise
    /// @return FileConfigHints& Reference to the modified config
    FileConfigHints &time_shift(bool enabled) {
        map[get_time_shift_key()] = std::to_string(enabled);
        return *this;
    }

    /// @brief Gets the maximum memory used (if applicable) setting
    /// @return Maximum memory used
    size_t max_memory() const {
        return get<std::size_t>(get_max_memory_key(), 0);
    }

    /// @brief Named constructor for the max memory setting
    /// @param max_memory Maximum memory that should be used when reading the file (if applicable)
    /// @return FileConfigHints& Reference to the modified config
    FileConfigHints &max_memory(std::size_t max_memory) {
        map[get_max_memory_key()] = std::to_string(max_memory);
        return *this;
    }

    /// @brief Gets the maximum read size (if applicable) setting
    /// @return Maximum read size in bytes
    size_t max_read_per_op() const {
        return get<std::size_t>(get_max_read_per_op_key(), 0);
    }

    /// @brief Named constructor for the max read size per read operation
    /// @param max_read_per_op Maximum size of data read per read operation
    /// @return FileConfigHints& Reference to the modified config
    FileConfigHints &max_read_per_op(std::size_t max_read_per_op) {
        map[get_max_read_per_op_key()] = std::to_string(max_read_per_op);
        return *this;
    }

    /// @brief Sets a value for a named key in the config dictionary
    /// @param key Key of the config
    /// @param value Value of the config
    template<typename T>
    void set(const std::string &key, const T &value) {
        map[key] = std::to_string(value);
    }

    /// @brief Sets a value for a named key in the config dictionary
    /// @overload
    void set(const std::string &key, const std::string &value) {
        map[key] = value;
    }

    /// @brief Gets the (typed) value for a named key in the config dictionary if it exists and can be extracted from
    ///        the corresponding value safely or the provided default value
    /// @tparam T type of the value to return as
    /// @param key Name of the config key
    /// @param def Default value if the config key is not found
    /// @return Value of the config
    template<typename T>
    T get(const std::string &key, const T &def = T()) const {
        T t(def);
        auto it = map.find(key);
        if (it != map.end()) {
            std::istringstream iss(it->second);
            iss >> t;
        }
        return t;
    }

private:
    std::unordered_map<std::string, std::string> map;
};
} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_FILE_CONFIG_HINTS_H
