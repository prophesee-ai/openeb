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

#ifndef METAVISION_SDK_BASE_STRING_H
#define METAVISION_SDK_BASE_STRING_H

#include <string>

namespace Metavision {

/// @brief Checks if a suffix is at the end of an iterable
///
/// @tparam T Usually a string, or any iterable.
/// @param value String to check.
/// @param suffix String suffix.
/// @return True if value ends with the suffix, false otherwise.
template<typename T>
bool ends_with(T const &value, T const &suffix) {
    if (suffix.size() > value.size())
        return false;
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

/// @brief Converts a string into a positive long if possible.
///
/// @param value String to convert.
/// @param result Result.
/// @return True if the value could be concerted into a positive long, false otherwise
bool unsigned_long_from_str(const std::string &value, unsigned long &result) {
    try {
        result = std::stol(value);
    } catch (const std::exception &) { return false; }
    return true;
}

} // namespace Metavision

#endif // METAVISION_SDK_BASE_STRING_H
