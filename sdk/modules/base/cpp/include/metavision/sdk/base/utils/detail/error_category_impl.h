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

#ifndef METAVISION_SDK_BASE_DETAIL_ERROR_CATEGORY_IMPL_H
#define METAVISION_SDK_BASE_DETAIL_ERROR_CATEGORY_IMPL_H

#include <sstream>

namespace Metavision {

inline ErrorCategory::ErrorCategory(int error_code, const std::string &name, const std::string &message) : name_(name) {
    error_message_ = "\n------------------------------------------------\n" + name + "\n\n";
    std::ostringstream error_as_hex;
    error_as_hex << std::hex << error_code;
    error_message_ += "Error " + error_as_hex.str() + ": " + message;
    error_message_ += "\n------------------------------------------------\n";
}

inline ErrorCategory::~ErrorCategory() {}

inline const char *ErrorCategory::name() const noexcept {
    return name_.c_str();
}

inline std::string ErrorCategory::message(int ev) const {
    return error_message_;
}

} // namespace Metavision

#endif // METAVISION_SDK_BASE_DETAIL_ERROR_CATEGORY_IMPL_H
