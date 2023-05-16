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

#ifndef METAVISION_SDK_CORE_FAST_MATH_FUNCTIONS_H
#define METAVISION_SDK_CORE_FAST_MATH_FUNCTIONS_H

#include <vector>

namespace Metavision {
namespace Math {

/// @brief Initializes the LUT to use with function @ref fast_exp_decay.
/// @param N Size of the LUT to initialize, higher sizes leading to higher accuracy.
/// @return LUT to use with function @ref fast_exp_decay.
std::vector<float> init_exp_decay_lut(std::size_t N);

/// @brief Uses a LUT pre-initialized using @ref init_exp_decay_lut to efficiently approximate the function f(v) =
/// exp(-v), for v in [0; +inf[.
/// @param lut LUT pre-initialized using @ref init_exp_decay_lut function.
/// @param v positive value (in [0; +inf[) at which to evaluate the function.
/// @return approximation of the evaluated function f(v) = exp(-v)
float fast_exp_decay(const std::vector<float> &lut, float v);

} // namespace Math
} // namespace Metavision

#endif // METAVISION_SDK_CORE_FAST_MATH_FUNCTIONS_H
