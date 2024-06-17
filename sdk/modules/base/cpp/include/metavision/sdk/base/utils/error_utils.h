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

#ifndef METAVISION_SDK_BASE_ERROR_UTILS_H
#define METAVISION_SDK_BASE_ERROR_UTILS_H

#include <sstream>
#include <string>
#include <system_error>

namespace Metavision {

/// @brief Class serving as base for all exceptions thrown by Metavision SDK and HAL
/// @sa http://www.cplusplus.com/reference/system_error/system_error/
/// @sa http://en.cppreference.com/w/cpp/error/error_code
class BaseException : public std::system_error {
public:
    /// @brief Creates an exception with default error message
    /// @param error_code Code identifying the error
    /// @param public_error_code overrides value that will be exposed as the std::error_code of the exception
    /// @param ecat Error category used for the exception
    /// @param what_arg Additional information for the error
    BaseException(int error_code, int public_error_code, const std::error_category &ecat, const std::string &what_arg) :
        std::system_error(public_error_code, ecat, what_arg) {
        msg_ = "\n------------------------------------------------\n";
        msg_ += std::string(ecat.name()) + "\n\n";
        std::ostringstream error_as_hex;
        error_as_hex << std::hex << error_code;
        msg_ += "Error " + error_as_hex.str() + ": ";
        if (what_arg != "") {
            msg_ += what_arg + "\n";
        }
        msg_ += ecat.message(error_code);
        msg_ += "\n------------------------------------------------\n";
    }

    /// @brief Creates an exception with default error message
    /// @param error_code Code identifying the error
    /// @param ecat Error category used for the exception
    /// @param what_arg Additional information for the error
    BaseException(int error_code, const std::error_category &ecat, const std::string &what_arg) :
        BaseException(error_code, error_code, ecat, what_arg) {}

    /// @brief Copy constructor
    BaseException(const BaseException &other) :
        std::system_error(other) {
        msg_ = other.msg_;
    }

    /// @brief Destructor
    virtual ~BaseException() {}

    /// @brief Returns the explanatory string
    /// @return Pointer to a null-terminated string with explanatory information
    virtual const char *what() const noexcept override {
        return msg_.c_str();
    }

private:
    std::string msg_;
};

} // namespace Metavision

#endif // METAVISION_SDK_BASE_ERROR_UTILS_H
