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

#ifndef METAVISION_HAL_HAL_CONNECTION_EXCEPTION_H
#define METAVISION_HAL_HAL_CONNECTION_EXCEPTION_H

#include <string>
#include <system_error>

#include "metavision/sdk/base/utils/error_utils.h"

namespace Metavision {

/// @brief Class for all connection exceptions thrown by Metavision HAL
/// @sa http://www.cplusplus.com/reference/system_error/system_error/
/// @sa http://en.cppreference.com/w/cpp/error/error_code
class HalConnectionException : public BaseException {
public:
    HalConnectionException(int ev, const std::error_category &ecat) : HalConnectionException(ev, ecat, "") {}

    HalConnectionException(int ev, const std::error_category &ecat, const std::string &additional_info) :
        BaseException(ev, ecat, additional_info) {}
};

} // namespace Metavision

#endif // METAVISION_HAL_HAL_CONNECTION_EXCEPTION_H
