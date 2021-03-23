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

#ifndef METAVISION_SDK_BASE_DETAIL_ERROR_CODE_IMPL_H
#define METAVISION_SDK_BASE_DETAIL_ERROR_CODE_IMPL_H

namespace Metavision {

inline int get_public_error_code(int error_code) {
    int has_internal_error_code = (error_code & INTERNAL_ERROR_TYPE_CODE_MASK);

    // if error is not internal, then return it as is
    if (!has_internal_error_code) {
        return error_code;
    }

    // error as an internal type: we return only the public part
    return (error_code & 0xFFFFFF00);
}

} // namespace Metavision

#endif // METAVISION_SDK_BASE_DETAIL_ERROR_CODE_IMPL_H
