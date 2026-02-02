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

#ifndef METAVISION_SDK_CORE_DETAIL_HISTO_PROCESSOR_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_HISTO_PROCESSOR_IMPL_H

#include "metavision/sdk/core/preprocessors/histo_processor.h"

namespace Metavision {

template<typename InputIt>
HistoProcessor<InputIt>::HistoProcessor(int event_input_width, int event_input_height, float max_incr_per_pixel,
                                        float clip_value_after_normalization, bool use_CHW, float width_scale,
                                        float height_scale) :
    EventPreprocessor<InputIt>((use_CHW) ?
                                   TensorShape({{"C", 2}, {"H", event_input_height}, {"W", event_input_width}}) :
                                   TensorShape({{"H", event_input_height}, {"W", event_input_width}, {"C", 2}}),
                               BaseType::FLOAT32),
    clip_value_after_normalization_(clip_value_after_normalization),
    width_(event_input_width),
    height_(event_input_height),
    channels_(2) {
    if (max_incr_per_pixel == 0.f)
        throw std::invalid_argument("max_incr_per_pixel can't be 0");
    if (clip_value_after_normalization <= 0.f)
        throw std::invalid_argument("clip_value_after_normalization must be > 0");
    if (width_scale <= 0.f || height_scale <= 0.f)
        throw std::runtime_error("Scaling factors for width and height should be > 0. Got " +
                                 std::to_string(width_scale) + " and " + std::to_string(height_scale));

    increment_ = 1.f / max_incr_per_pixel;
    // Further normalize the increment_ to make up for adding more events per histogram cell (when there is a previous
    // rescaling of events)
    increment_ *= width_scale * height_scale;
}

template<typename InputIt>
bool HistoProcessor<InputIt>::is_CHW(const Tensor &t) const {
    const auto &dimensions = t.shape().dimensions;
    return dimensions[0].name == "C" && dimensions[1].name == "H" && dimensions[2].name == "W";
}

template<typename InputIt>
void HistoProcessor<InputIt>::compute(const timestamp cur_frame_start_ts, InputIt begin, InputIt end,
                                      Tensor &tensor) const {
    assert(tensor.type() == BaseType::FLOAT32);
    auto buff            = tensor.data<float>();
    const auto buff_size = tensor.shape().get_nb_values();
    assert(buff_size == this->output_tensor_shape_.get_nb_values());
    if (is_CHW(tensor)) {
        for (auto it = begin; it != end; ++it) {
            const auto &ev = *it;
            assert((ev.p == 0) || (ev.p == 1));
            assert(ev.t >= cur_frame_start_ts);
            assert(ev.x >= 0);
            assert(ev.x < width_);
            assert(ev.y >= 0);
            assert(ev.y < height_);
            const int idx = width_ * (height_ * ev.p + ev.y) + ev.x;
            assert(idx < static_cast<int>(buff_size));
            buff[idx] = std::min(clip_value_after_normalization_, buff[idx] + increment_);
        }
    } else {
        for (auto it = begin; it != end; ++it) {
            const auto &ev = *it;
            assert((ev.p == 0) || (ev.p == 1));
            assert(ev.t >= cur_frame_start_ts);
            assert(ev.x >= 0);
            assert(ev.x < width_);
            assert(ev.y >= 0);
            assert(ev.y < height_);
            const int idx = channels_ * (width_ * ev.y + ev.x) + ev.p;
            assert(idx < static_cast<int>(buff_size));
            buff[idx] = std::min(clip_value_after_normalization_, buff[idx] + increment_);
        }
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_HISTO_PROCESSOR_IMPL_H
