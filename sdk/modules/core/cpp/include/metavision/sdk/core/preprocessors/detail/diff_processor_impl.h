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

#ifndef METAVISION_SDK_CORE_DETAIL_DIFF_PROCESSOR_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_DIFF_PROCESSOR_IMPL_H

#include "metavision/sdk/core/preprocessors/diff_processor.h"

namespace Metavision {

template<typename InputIt>
DiffProcessor<InputIt>::DiffProcessor(int event_input_width, int event_input_height, float max_incr_per_pixel,
                                      float clip_value_after_normalization, float width_scale, float height_scale) :
    EventPreprocessor<InputIt>(TensorShape({{"C", 1}, {"H", event_input_height}, {"W", event_input_width}}),
                               BaseType::FLOAT32),
    clip_value_after_normalization_(clip_value_after_normalization),
    width_(event_input_width) {
    if (max_incr_per_pixel == 0.f)
        throw std::invalid_argument("max_incr_per_pixel can't be 0");
    if (clip_value_after_normalization <= 0.f)
        throw std::invalid_argument("clip_value_after_normalization must be > 0");

    increment_ = 1.f / max_incr_per_pixel;
    if (width_scale <= 0.f || height_scale <= 0.f)
        throw std::runtime_error("Scaling factors for width and height should be > 0. Got " +
                                 std::to_string(width_scale) + " and " + std::to_string(height_scale));

    // Further normalize the increment_ to make up for adding more events per frame cell (when there is a previous
    // rescaling of events)
    increment_ *= width_scale * height_scale;
}

template<typename InputIt>
void DiffProcessor<InputIt>::compute(const timestamp cur_frame_start_ts, InputIt begin, InputIt end,
                                     Tensor &tensor) const {
    assert(tensor.type() == BaseType::FLOAT32);
    auto buff            = tensor.data<float>();
    const auto buff_size = tensor.shape().get_nb_values();
    for (auto it = begin; it != end; ++it) {
        const auto &ev = *it;
        assert((ev.p == 0) || (ev.p == 1));
        assert(ev.t >= cur_frame_start_ts);
        assert(ev.x >= 0);
        assert(ev.x < width_);
        assert(ev.y >= 0);
        assert(ev.y < get_dim(this->output_tensor_shape_, "H"));
        const int idx = ev.x + width_ * ev.y;
        assert(idx >= 0);
        assert(idx < static_cast<int>(buff_size));
        const int p = 2 * ev.p - 1;
        buff[idx]   = std::max(-clip_value_after_normalization_,
                             std::min(clip_value_after_normalization_, buff[idx] + increment_ * p));
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_DIFF_PROCESSOR_IMPL_H
