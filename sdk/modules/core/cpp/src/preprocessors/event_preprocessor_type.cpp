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

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/preprocessors/event_preprocessor_type.h"

namespace std {

std::istream &operator>>(std::istream &in, Metavision::EventPreprocessorType &type) {
    std::string type_str;
    in >> type_str;

    if (Metavision::stringToEventPreprocessorTypeMap.count(type_str) > 0)
        type = Metavision::stringToEventPreprocessorTypeMap.at(type_str);
    else
        throw std::runtime_error(type_str + " is not a compatible event preprocessor type");

    return in;
}

std::ostream &operator<<(std::ostream &os, const Metavision::EventPreprocessorType &type) {
    os << Metavision::eventPreprocessorTypeToStringMap.at(type);
    return os;
}

} // namespace std
