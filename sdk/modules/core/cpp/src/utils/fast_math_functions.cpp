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

#include <algorithm>
#include <assert.h>
#include <cmath>
#include "metavision/sdk/core/utils/fast_math_functions.h"

namespace Metavision {
namespace Math {

std::vector<float> init_exp_decay_lut(std::size_t N) {
    std::vector<float> lut(N);
    for (std::size_t i = 0; i < N; ++i) {
        lut[i] = -std::log((N - i) / static_cast<float>(N));
    }
    return lut;
}

float fast_exp_decay(const std::vector<float> &lut, float v) {
    assert(v >= 0);
    if (v <= 0.f)
        return 1.f;
    auto it_lut = std::upper_bound(lut.cbegin(), lut.cend(), v);
    if (it_lut == lut.cend())
        return 0.f;
    const std::size_t N   = lut.size();
    const float d0v       = *std::prev(it_lut) - v;
    const float d01       = *std::prev(it_lut) - *it_lut;
    const float intercept = (N + 1 - std::distance(lut.cbegin(), it_lut)) / static_cast<float>(N);
    const float slope     = -1 / (d01 * static_cast<float>(N));
    return intercept + slope * d0v;
}

} // namespace Math
} // namespace Metavision
