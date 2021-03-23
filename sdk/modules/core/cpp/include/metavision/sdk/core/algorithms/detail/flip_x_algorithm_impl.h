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

#ifndef METAVISION_SDK_CORE_DETAIL_FLIP_X_ALGORITHM_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_FLIP_X_ALGORITHM_IMPL_H

namespace Metavision {

inline FlipXAlgorithm::FlipXAlgorithm(std::int16_t width_minus_one) : width_minus_one_(width_minus_one) {}

inline std::int16_t FlipXAlgorithm::width_minus_one() const {
    return width_minus_one_;
}

inline void FlipXAlgorithm::set_width_minus_one(std::int16_t width_minus_one) {
    width_minus_one_ = width_minus_one;
}

inline void FlipXAlgorithm::operator()(Event2d &ev) const {
    ev.x = static_cast<std::uint16_t>(width_minus_one_ - ev.x);
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_FLIP_X_ALGORITHM_IMPL_H
