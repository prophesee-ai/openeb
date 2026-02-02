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

#ifndef METAVISION_SDK_CORE_JSON_PARSER_H
#define METAVISION_SDK_CORE_JSON_PARSER_H

#include <boost/property_tree/ptree.hpp>
#include <filesystem>
#include <variant>
#include <unordered_map>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/core/preprocessors/event_preprocessor_type.h"

namespace Metavision {

/// @brief Opens a JSON file safely and provides a ptree to explore it
/// @param file_path JSON file to read
/// @return Tree describing the JSON structure
/// @throw runtime_error if the file does not exist or if the parsing is not successful
boost::property_tree::ptree get_tree_from_file(const std::filesystem::path &file_path);

/// @brief Gets the value stored in a ptree referenced by its name
/// @tparam T Type of the data to retrieve
/// @param pt Root containing the information tree
/// @param key Name of the value to retrieve
/// @returns The value referenced by the input key
/// @throws std::runtime_error if the value was not found in the ptree
template<typename T>
T get_element_from_ptree(const boost::property_tree::ptree &pt, const std::string &key);

using PreprocessingParameters = std::variant<bool, uint8_t, int8_t, int, float, timestamp, std::string,
                                             std::vector<std::string>, EventPreprocessorType>;

/// @brief Reads the preprocessor parameters from the provided node
/// @param pt The node storing the different preprocessors configurations. It can either be a list or a single processor
/// parameters definition.
/// @param preprocess_maps List of the read preprocessor parameters
void parse_preprocessors_params(const boost::property_tree::ptree &pt,
                                std::vector<std::unordered_map<std::string, PreprocessingParameters>> &preprocess_maps);

} // namespace Metavision

#include "metavision/sdk/core/preprocessors/detail/json_parser_impl.h"

#endif // METAVISION_SDK_CORE_JSON_PARSER_H
