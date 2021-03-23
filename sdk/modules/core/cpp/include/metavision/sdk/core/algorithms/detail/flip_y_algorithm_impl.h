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

#ifndef METAVISION_SDK_CORE_DETAIL_FLIP_Y_ALGORITHM_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_FLIP_Y_ALGORITHM_IMPL_H

namespace Metavision {

inline void Metavision::FlipYAlgorithm::operator()(Metavision::Event2d &ev) const {
    ev.y = static_cast<std::uint16_t>(height_minus_one_ - ev.y);
}

inline FlipYAlgorithm::FlipYAlgorithm(std::int16_t height_minus_one) : height_minus_one_(height_minus_one) {}

inline std::int16_t FlipYAlgorithm::height_minus_one() const {
    return height_minus_one_;
}

inline void FlipYAlgorithm::set_height_minus_one(std::int16_t height_minus_one) {
    height_minus_one_ = height_minus_one;
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_FLIP_Y_ALGORITHM_IMPL_H
