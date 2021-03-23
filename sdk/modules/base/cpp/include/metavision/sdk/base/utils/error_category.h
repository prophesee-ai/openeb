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

#ifndef METAVISION_SDK_BASE_ERROR_CATEGORY_H
#define METAVISION_SDK_BASE_ERROR_CATEGORY_H

#include <system_error>
#include <string>

namespace Metavision {

struct ErrorCategory : public std::error_category {
    ErrorCategory()                      = delete;
    ErrorCategory(const ErrorCategory &) = delete;
    ErrorCategory(int error_code, const std::string &name = "", const std::string &message = "");
    virtual ~ErrorCategory();

    const char *name() const noexcept override;
    std::string message(int ev) const override;

private:
    std::string name_, error_message_;
};

} // namespace Metavision

#include "detail/error_category_impl.h"

#endif // METAVISION_SDK_BASE_ERROR_CATEGORY_H
