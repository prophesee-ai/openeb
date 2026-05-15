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

#include <fstream>
#include <iomanip>
#include <memory>

#include "metavision/sdk/base/utils/generic_header.h"
#include "metavision/hal/facilities/i_ll_biases.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

I_LL_Biases::I_LL_Biases(const DeviceConfig &device_config) : device_config_(device_config) {}

bool I_LL_Biases::set(const std::string &bias_name, int bias_value) {
    LL_Bias_Info bias_info;
    if (!get_bias_info(bias_name, bias_info)) {
        throw HalException(HalErrorCode::NonExistingValue, "Unavailable bias: \"" + bias_name + "\".");
        return false;
    }
    if (!bias_info.is_modifiable()) {
        throw HalException(HalErrorCode::OperationNotPermitted, "Bias \"" + bias_name + "\" cannot be modified.");
        return false;
    }
    if (!device_config_.biases_range_check_bypass()) {
        auto range = bias_info.get_bias_range();
        if (bias_value > range.second || bias_value < range.first) {
            std::stringstream ss;
            ss << "Invalid value " << bias_value << " for bias \"" << bias_name << "\". Value should be within range ["
               << range.first << ", " << range.second << "].";
            throw HalException(HalErrorCode::ValueOutOfRange, ss.str());
            return false;
        }
    }
    return set_impl(bias_name, bias_value);
}

int I_LL_Biases::get(const std::string &bias_name) const {
    LL_Bias_Info bias_info;
    if (!get_bias_info(bias_name, bias_info)) {
        throw HalException(HalErrorCode::NonExistingValue, "Unavailable bias: \"" + bias_name + "\".");
        return false;
    }
    return get_impl(bias_name);
}

bool I_LL_Biases::get_bias_info(const std::string &bias_name, LL_Bias_Info &bias_info) const {
    if (!get_bias_info_impl(bias_name, bias_info)) {
        throw HalException(HalErrorCode::NonExistingValue, "Unavailable bias: \"" + bias_name + "\".");
        return false;
    }
    if (device_config_.biases_range_check_bypass()) {
        bias_info.disable_recommended_range();
    }
    return true;
}

void I_LL_Biases::load_from_file(const std::filesystem::path &src_file) {
    // Check extension
    const auto extension = src_file.extension().string();
    if (extension != ".bias") {
        throw HalException(HalErrorCode::InvalidArgument,
                           "For bias file '" + src_file.string() +
                           "' : expected '.bias' extension to set the bias from this file but got '." +
                           extension + "'");
    }

    // open file
    std::ifstream bias_file(src_file);
    if (!bias_file.is_open()) {
        throw HalException(HalErrorCode::InvalidArgument,
                           "Could not open file '" + src_file.string() + "' for reading. Failed to set biases.");
    }

    // Skip header if any
    GenericHeader header(bias_file);

    // Get available biases :
    std::map<std::string, int> available_biases = get_all_biases();

    // Parse the file to get the list of the biases that the user wants to set
    std::map<std::string, int> biases_to_set;
    for (std::string line; std::getline(bias_file, line) && !line.empty();) {
        std::stringstream ss(line);

        // Get value and name
        std::string value_str, bias_name, separator;
        ss >> value_str >> separator >> bias_name;
        std::transform(value_str.begin(), value_str.end(), value_str.begin(), ::tolower);

        if (value_str.empty() || bias_name.empty()) {
            throw HalException(HalErrorCode::InvalidArgument,
                               "Cannot read bias file '" + src_file.string() + "' : wrong line format '" + line + "'");
        }
        int value;
        if (value_str.find("0x") != std::string::npos) {
            value = std::stoi(value_str, 0, 16);

        } else {
            value = std::stol(value_str);
        }

        // Check if the bias that we want to set is compatible and not read only
        LL_Bias_Info bias_info;
        get_bias_info(bias_name, bias_info);
        if (!bias_info.is_modifiable()) {
            continue;
        }

        auto it = biases_to_set.find(bias_name);
        if (it != biases_to_set.end()) {
            if (value != it->second) {
                throw HalException(HalErrorCode::InvalidArgument, "Given two different values for bias '" +
                                   bias_name + "' in file '" + src_file.string() + "'");
            }
        }
        biases_to_set.emplace(bias_name, value);
    }

    // If we get here, no error was found, and we can proceed in setting the biases
    for (auto it = biases_to_set.begin(), it_end = biases_to_set.end(); it != it_end; ++it) {
        set(it->first, it->second);
    }
}

void I_LL_Biases::save_to_file(const std::filesystem::path &dest_file) const {
    const auto extension = dest_file.extension().string();
    if (extension != ".bias") {
        throw HalException(HalErrorCode::InvalidArgument,
                           "For bias file '" + dest_file.string() +
                           "' : expected '.bias' extension to set the bias from this file but got '." +
                           extension + "'");
    }

    std::ofstream output_file(dest_file);
    if (!output_file.is_open()) {
        throw HalException(HalErrorCode::InvalidArgument,
                           "Could not open file '" + dest_file.string() + "' for writing. Failed to save biases.");
    }

    // Get available biases :
    std::map<std::string, int> available_biases = get_all_biases();

    for (auto it = available_biases.begin(), it_end = available_biases.end(); it != it_end; ++it) {
        output_file << std::left << std::setw(5) << it->second << "% " << it->first << std::endl;
    }
    output_file.close();
}

LL_Bias_Info::LL_Bias_Info(int min_value, int max_value, const std::string &description, bool modifiable,
                           const std::string &category) :
    description_(description),
    category_(category),
    modifiable_(modifiable),
    use_recommended_range_(true),
    bias_allowed_range_(std::make_pair(min_value, max_value)),
    bias_recommended_range_(std::make_pair(min_value, max_value)) {}

LL_Bias_Info::LL_Bias_Info(int min_allowed_value, int max_allowed_value, int min_recommended_value,
                           int max_recommended_value, const std::string &description, bool modifiable,
                           const std::string &category) :
    description_(description),
    category_(category),
    modifiable_(modifiable),
    use_recommended_range_(true),
    bias_allowed_range_(std::make_pair(min_allowed_value, max_allowed_value)),
    bias_recommended_range_(std::make_pair(min_recommended_value, max_recommended_value)) {}

const std::string &LL_Bias_Info::get_description() const {
    return description_;
}

const std::string &LL_Bias_Info::get_category() const {
    return category_;
}

std::pair<int, int> LL_Bias_Info::get_bias_range() const {
    return use_recommended_range_ ? bias_recommended_range_ : bias_allowed_range_;
}

std::pair<int, int> LL_Bias_Info::get_bias_allowed_range() const {
    return bias_allowed_range_;
}

std::pair<int, int> LL_Bias_Info::get_bias_recommended_range() const {
    return bias_recommended_range_;
}

bool LL_Bias_Info::is_modifiable() const {
    return modifiable_;
}

void LL_Bias_Info::disable_recommended_range() {
    use_recommended_range_ = false;
}

} // namespace Metavision
