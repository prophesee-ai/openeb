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

#ifndef METAVISION_UTILS_PYBIND_DEPRECATION_WARNING_EXCEPTION_H
#define METAVISION_UTILS_PYBIND_DEPRECATION_WARNING_EXCEPTION_H

#include <exception>
#include <string>

namespace Metavision {

class DeprecationWarningException : std::exception {
public:
    DeprecationWarningException(const std::string &old_name, const std::string &new_name) {
        msg_ = old_name + " is deprecated. Use " + new_name + " instead.";
    }

    DeprecationWarningException(const std::string &old_name) {
        msg_ = old_name + " is deprecated.";
    }

    char const *what() const noexcept {
        return msg_.c_str();
    }

private:
    std::string msg_;
};

} // namespace Metavision

#endif // METAVISION_UTILS_PYBIND_DEPRECATION_WARNING_EXCEPTION_H
