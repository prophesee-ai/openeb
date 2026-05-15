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

#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "utils/psee_geometry.h"
#include <string>
#include <sstream>

namespace Metavision {

StreamFormat::StreamFormat(std::string format) {
    std::istringstream sf(format);

    // Format strings are expected to look like
    // "EVT3;height=720;width=1280"

    // first element is the format name
    std::getline(sf, format_name, ';');

    // then what remains is options
    // we don't try to detect malformed strings
    while (!sf.eof()) {
        std::string option;
        getline(sf, option, ';');
        {
            std::string name, value;
            std::istringstream so(option);
            std::getline(so, name, '=');
            std::getline(so, value, '=');
            this->options[name] = value;
        }
    }
}

bool StreamFormat::contains(const std::string &option) const {
    return options.find(option) != options.end();
}

std::string &StreamFormat::operator[](const std::string &name) {
    return options[name];
}

const std::string &StreamFormat::operator[](const std::string &name) const {
    static const std::string empty_string = "";
    auto it                               = options.find(name);
    if (it != options.end()) {
        return it->second;
    }
    return empty_string;
}

std::string StreamFormat::name() const {
    return format_name;
}

std::string StreamFormat::to_string() const {
    std::string format = format_name;
    for (auto option : options) {
        format += ";" + option.first + "=" + option.second;
    }
    return format;
}

std::unique_ptr<I_Geometry> StreamFormat::geometry() const {
    long width, height;
    try {
        width  = strtol(options.at("width").c_str(), NULL, 0);
        height = strtol(options.at("height").c_str(), NULL, 0);
        if (!width || !height) {
            throw std::invalid_argument("Format is missing a valid geometry");
        }
    } catch (const std::out_of_range &) {
        // Here we catch throws on map::at
        throw std::invalid_argument("Format has no geometry");
    }
    return std::make_unique<PseeGeometry>(width, height);
}

} // namespace Metavision
