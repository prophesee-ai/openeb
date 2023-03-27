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
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <boost/filesystem.hpp>

#include "metavision/hal/utils/hal_exception.h"
#include "metavision/sdk/driver/biases.h"
#include "metavision/sdk/driver/camera_exception.h"
#include "metavision/sdk/driver/camera_error_code.h"
#include "metavision/sdk/driver/internal/camera_error_code_internal.h"
#include "metavision/sdk/base/utils/generic_header.h"
#include "metavision/sdk/driver/camera_exception.h"

namespace Metavision {

Biases::Biases(I_LL_Biases *i_ll_biases) : pimpl_(i_ll_biases) {}

Biases::~Biases() {}

void Biases::set_from_file(const std::string &biases_filename) {
    // Check extension
    const auto extension = boost::filesystem::extension(biases_filename);
    if (extension != ".bias") {
        throw CameraException(CameraErrorCode::WrongExtension,
                              "For bias file '" + biases_filename +
                                  "' : expected '.bias' extension to set the bias from this file but got '." +
                                  extension + "'");
    }

    // open file
    std::ifstream bias_file(biases_filename);
    if (!bias_file.is_open()) {
        throw CameraException(CameraErrorCode::CouldNotOpenFile,
                              "Could not open file '" + biases_filename + "' for reading. Failed to set biases.");
    }

    // Skip header if any
    GenericHeader header(bias_file);

    // Get available biases :
    std::map<std::string, int> available_biases = pimpl_->get_all_biases();

    // Parse the file to get the list of the biases that the user wants to set
    std::map<std::string, int> biases_to_set;
    for (std::string line; std::getline(bias_file, line) && !line.empty();) {
        std::stringstream ss(line);

        // Get value and name
        std::string value_str, bias_name, separator;
        ss >> value_str >> separator >> bias_name;
        std::transform(value_str.begin(), value_str.end(), value_str.begin(), ::tolower);

        if (value_str.empty() || bias_name.empty()) {
            throw CameraException(BiasesErrors::UnsupportedBiasFile,
                                  "Cannot read bias file '" + biases_filename + "' : wrong line format '" + line + "'");
        }
        int value;
        if (value_str.find("0x") != std::string::npos) {
            value = std::stoi(value_str, 0, 16);

        } else {
            value = std::stol(value_str);
        }

        // Check if the bias that we want to set is compatible and not read only
        LL_Bias_Info bias_info;
        bool ret = true;
        try {
            ret = pimpl_->get_bias_info(bias_name, bias_info);
        } catch (Metavision::HalException &) { ret = false; }
        if (!ret) {
            throw CameraException(BiasesErrors::UnsupportedBias,
                                  "Bias '" + bias_name + "' is not compatible with the device.");
        }
        if (!bias_info.is_modifiable()) {
            continue;
        }

        auto it = biases_to_set.find(bias_name);
        if (it != biases_to_set.end()) {
            if (value != it->second) {
                throw CameraException(CameraErrorCode::BiasesError, "Given two different values for bias '" +
                                                                        bias_name + "' in file '" + biases_filename +
                                                                        "'");
            }
        }
        biases_to_set.emplace(bias_name, value);
    }

    // If we get here, no error was found, and we can proceed in setting the biases
    for (auto it = biases_to_set.begin(), it_end = biases_to_set.end(); it != it_end; ++it) {
        pimpl_->set(it->first, it->second);
    }
}

void Biases::save_to_file(const std::string &dest_file) const {
    const auto extension = boost::filesystem::extension(dest_file);
    if (extension != ".bias") {
        throw CameraException(CameraErrorCode::WrongExtension,
                              "For bias file '" + dest_file +
                                  "' : expected '.bias' extension to set the bias from this file but got '." +
                                  extension + "'");
    }

    std::ofstream output_file(dest_file);
    if (!output_file.is_open()) {
        throw CameraException(CameraErrorCode::CouldNotOpenFile,
                              "Could not open file '" + dest_file + "' for writing. Failed to save biases.");
    }

    // Get available biases :
    std::map<std::string, int> available_biases = pimpl_->get_all_biases();

    for (auto it = available_biases.begin(), it_end = available_biases.end(); it != it_end; ++it) {
        output_file << std::left << std::setw(5) << it->second << "% " << it->first << std::endl;
    }
    output_file.close();
}

I_LL_Biases *Biases::get_facility() const {
    return pimpl_;
}

} // namespace Metavision
