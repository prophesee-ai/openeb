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

#ifndef METAVISION_HAL_I_LL_BIASES_H
#define METAVISION_HAL_I_LL_BIASES_H

#include <limits>
#include <string>
#include <map>
#include <utility>

#include "metavision/hal/facilities/i_registrable_facility.h"
#include "metavision/hal/utils/device_config.h"

namespace Metavision {

class LL_Bias_Info;

/// @brief Interface facility for Low Level Biases
class I_LL_Biases : public I_RegistrableFacility<I_LL_Biases> {
public:
    /// @brief Constructor
    /// @param device_config Device configuration
    I_LL_Biases(const DeviceConfig &device_config);

    /// @brief Sets bias value
    /// @param bias_name Bias to set
    /// @param bias_value Value to set the bias to
    /// @return true on success
    bool set(const std::string &bias_name, int bias_value);

    /// @brief Gets bias value
    /// @param bias_name Name of the bias whose value to get
    /// @return The bias value
    int get(const std::string &bias_name);

    /// @brief Gets bias metadata
    /// @param bias_name Name of the bias whose metadata to get
    /// @param bias_info Metadata of the bias to get
    /// @return true on success
    bool get_bias_info(const std::string &bias_name, LL_Bias_Info &bias_info) const;

    /// @brief Gets all biases values
    /// @return A map containing the biases values
    virtual std::map<std::string, int> get_all_biases() = 0;

protected:
    DeviceConfig device_config_;

private:
    /// @brief Sets bias value
    ///
    /// The implementation can assume the @p bias_name is valid
    /// When called, get_bias_info_impl has already been called, so no additional validation
    /// is required
    ///
    /// @param bias_name Name of the bias to set
    /// @param bias_value Value to set the bias to
    /// @return true on success
    virtual bool set_impl(const std::string &bias_name, int bias_value) = 0;

    /// @brief Gets bias value
    ///
    /// The implementation can assume the @p bias_name is valid
    /// When called, get_bias_info_impl has already been called, so no additional validation
    /// is required
    ///
    /// @param bias_name Name of the bias whose value to get
    /// @return The bias value
    virtual int get_impl(const std::string &bias_name) = 0;

    /// @brief Gets bias metadata
    ///
    /// The implementation must make sure the @p bias_name is valid
    ///
    /// @param bias_name Name of the bias whose metadata to get
    /// @param bias_info Metadata of the bias to get
    /// @return true on success
    virtual bool get_bias_info_impl(const std::string &bias_name, LL_Bias_Info &bias_info) const = 0;
};

/// @brief Base class used to represent Low Level Biases metadata
class LL_Bias_Info {
public:
    /// @brief Constructor
    /// @param min_value Minimum allowed value for corresponding bias
    /// @param max_value Maximum allowed value for corresponding bias
    /// @param description String describing the bias
    /// @param modifiable Whether the values of the bias can be programmatically changed
    /// @param category String representing the bias' category
    LL_Bias_Info(int min_value = std::numeric_limits<int>::min(), int max_value = std::numeric_limits<int>::max(),
                 const std::string &description = "", bool modifiable = false, const std::string &category = "");

    /// @brief Constructor
    /// @param min_allowed_value Minimum allowed value for corresponding bias
    /// @param max_allowed_value Maximum allowed value for corresponding bias
    /// @param min_recommended_value Minimum recommended value for corresponding bias
    /// @param max_recommended_value Maximum recommended value for corresponding bias
    /// @param description String describing the bias
    /// @param modifiable Whether the values of the bias can be programmatically changed
    /// @param category String representing the bias' category
    LL_Bias_Info(int min_allowed_value, int max_allowed_value, int min_recommended_value, int max_recommended_value,
                 const std::string &description = "", bool modifiable = false, const std::string &category = "");

    /// @brief Gets the bias' description
    /// @returns bias description string
    const std::string &get_description() const;

    /// @brief Gets the bias' category
    /// @returns bias category string
    const std::string &get_category() const;

    /// @brief Gets the bias' range of valid values
    ///
    /// The range of values returned is by default the recommended one
    /// If the @ref DeviceConfig::biases_range_check_bypass option has been enabled via
    /// the @ref DeviceConfig when opening the device, the range returned is the allowed one
    ///
    /// @returns A pair representing the [min, max] range of values supported by the bias
    std::pair<int, int> get_bias_range() const;

    /// @brief Gets the bias' range of recommended values
    /// @returns A pair representing the [min, max] range of values recommended
    std::pair<int, int> get_bias_recommended_range() const;

    /// @brief Gets the bias' range of allowed values
    /// @returns A pair representing the [min, max] range of values allowed
    std::pair<int, int> get_bias_allowed_range() const;

    /// @brief Gets the bias' modifiable parameter
    /// @returns True if bias' value can be modified
    bool is_modifiable() const;

private:
    std::string description_;
    std::string category_;
    bool modifiable_;
    bool use_recommended_range_;
    std::pair<int, int> bias_allowed_range_, bias_recommended_range_;

    void disable_recommended_range();
    friend bool I_LL_Biases::get_bias_info(const std::string &, LL_Bias_Info &) const;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_LL_BIASES_H
