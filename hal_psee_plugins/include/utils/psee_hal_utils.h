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

#ifndef METAVISION_HAL_PSEE_HAL_UTILS_H
#define METAVISION_HAL_PSEE_HAL_UTILS_H

#include <string>

namespace Metavision {

/// @brief Returns a description for a given bias name
/// @return The bias description or an empty string
const std::string &get_bias_description(const std::string &bias);

/// @brief Returns a category for a given bias name
/// @return The bias category or an empty string
const std::string &get_bias_category(const std::string &bias);

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_HAL_UTILS_H
