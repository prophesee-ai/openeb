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

#include <boost/property_tree/json_parser.hpp>

#include "metavision/sdk/core/preprocessors/json_parser.h"
#include "metavision/sdk/core/preprocessors/event_preprocessor_type.h"

namespace Metavision {

namespace detail {

void read_diff(const boost::property_tree::ptree &node,
               std::unordered_map<std::string, PreprocessingParameters> &params_map) {
    params_map["max_incr_per_pixel"] = get_element_from_ptree<float>(node, "max_incr_per_pixel");
    params_map["clip_value_after_normalization"] =
        get_element_from_ptree<float>(node, "clip_value_after_normalization");
}

void read_histo(const boost::property_tree::ptree &node,
                std::unordered_map<std::string, PreprocessingParameters> &params_map) {
    params_map["max_incr_per_pixel"] = get_element_from_ptree<float>(node, "max_incr_per_pixel");
    params_map["clip_value_after_normalization"] =
        get_element_from_ptree<float>(node, "clip_value_after_normalization");
    params_map["use_CHW"] = get_element_from_ptree<bool>(node, "use_CHW");
}

void read_event_cube(const boost::property_tree::ptree &node,
                     std::unordered_map<std::string, PreprocessingParameters> &params_map) {
    params_map["delta_t"]            = get_element_from_ptree<timestamp>(node, "delta_t");
    params_map["max_incr_per_pixel"] = get_element_from_ptree<float>(node, "max_incr_per_pixel");
    params_map["clip_value_after_normalization"] =
        get_element_from_ptree<float>(node, "clip_value_after_normalization");
    params_map["num_utbins"]     = get_element_from_ptree<int>(node, "num_utbins");
    params_map["split_polarity"] = get_element_from_ptree<bool>(node, "split_polarity");
}

void read_time_surface(const boost::property_tree::ptree &node,
                       std::unordered_map<std::string, PreprocessingParameters> &params_map) {
    params_map["nb_channels"] = get_element_from_ptree<uint8_t>(node, "nb_channels");
}

void read_hardware_diff(const boost::property_tree::ptree &node,
                        std::unordered_map<std::string, PreprocessingParameters> &params_map) {
    params_map["min_val"]        = get_element_from_ptree<int8_t>(node, "min_val");
    params_map["max_val"]        = get_element_from_ptree<int8_t>(node, "max_val");
    params_map["allow_rollover"] = get_element_from_ptree<bool>(node, "allow_rollover");
}

void read_hardware_histo(const boost::property_tree::ptree &node,
                         std::unordered_map<std::string, PreprocessingParameters> &params_map) {
    params_map["neg_saturation"] = get_element_from_ptree<uint8_t>(node, "neg_saturation");
    params_map["pos_saturation"] = get_element_from_ptree<uint8_t>(node, "pos_saturation");
}

} // namespace detail

boost::property_tree::ptree get_tree_from_file(const std::filesystem::path &file_path) {
    std::stringstream file_buffer;
    std::ifstream file;

    try {
        file.open(file_path);
        file_buffer << file.rdbuf();
        file.close();
    } catch (...) { throw std::runtime_error(std::string(" No such file: '") + file_path.string() + "'"); }

    boost::property_tree::ptree pt;
    try {
        boost::property_tree::read_json(file_buffer, pt);
    } catch (const std::exception &e) {
        throw std::runtime_error(e.what() + std::string(" reading '") + file_path.string() + "'");
    } catch (...) { throw std::runtime_error("Unknown exception thrown reading '" + file_path.string() + "'"); }
    return pt;
}

void parse_preprocessors_params(
    const boost::property_tree::ptree &pt,
    std::vector<std::unordered_map<std::string, PreprocessingParameters>> &preprocess_maps) {
    const auto read_parameters = [](const boost::property_tree::ptree &node,
                                    std::unordered_map<std::string, PreprocessingParameters> &params_map) {
        const EventPreprocessorType process_type = node.get_child("type").get_value<EventPreprocessorType>();
        params_map["type"]                       = process_type;
        switch (process_type) {
        case EventPreprocessorType::DIFF:
            detail::read_diff(node, params_map);
            break;
        case EventPreprocessorType::HISTO:
            detail::read_histo(node, params_map);
            break;
        case EventPreprocessorType::EVENT_CUBE:
            detail::read_event_cube(node, params_map);
            break;
        case EventPreprocessorType::TIME_SURFACE:
            detail::read_time_surface(node, params_map);
            break;
        case EventPreprocessorType::HARDWARE_DIFF:
            detail::read_hardware_diff(node, params_map);
            break;
        case EventPreprocessorType::HARDWARE_HISTO:
            detail::read_hardware_histo(node, params_map);
            break;
        }
        std::vector<std::string> input_names;
        const auto inputs_node = node.get_child("input_names");
        if (inputs_node.empty())
            throw std::runtime_error(
                "No inputs provided for the current preprocessor. Provide names with the 'input_names' attribute.");

        if (inputs_node.count("") == 1) {
            // If the value node has an empty key, it's a list
            for (auto &name : inputs_node)
                input_names.push_back(name.second.get_value<std::string>());
        } else {
            input_names.push_back(inputs_node.get_value<std::string>());
        }
        params_map["input_names"] = input_names;
    };

    if (pt.empty())
        return;

    if (pt.count("") == 1) {
        // If the value node has an empty key, it's a list
        for (const auto &process_node : pt) {
            std::unordered_map<std::string, PreprocessingParameters> proc;
            read_parameters(process_node.second, proc);
            preprocess_maps.emplace_back(proc);
        }
    } else {
        std::unordered_map<std::string, PreprocessingParameters> proc;
        read_parameters(pt, proc);
        preprocess_maps.emplace_back(proc);
    }
}

} // namespace Metavision
