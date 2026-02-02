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

#ifndef METAVISION_SDK_CORE_PREPROCESSORS_JSON_PARSER_IMPL_H
#define METAVISION_SDK_CORE_PREPROCESSORS_JSON_PARSER_IMPL_H

#include "metavision/sdk/core/preprocessors/json_parser.h"

namespace Metavision {

template<typename T>
T get_element_from_ptree(const boost::property_tree::ptree &pt, const std::string &key) {
    try {
        T value = pt.get<T>(key);
        return value;
    } catch (const boost::property_tree::ptree_error &e) {
        std::ostringstream oss;
        oss << "Error: could not find the key '" << key << "' in provided tree." << std::endl;
        oss << e.what() << std::endl;
        throw std::runtime_error(oss.str());
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_PREPROCESSORS_JSON_PARSER_IMPL_H
