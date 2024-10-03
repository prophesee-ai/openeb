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

#ifndef METAVISION_SDK_CORE_DETAIL_EVENT_CUBE_PROCESSOR_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_EVENT_CUBE_PROCESSOR_IMPL_H

#include "metavision/sdk/core/preprocessors/event_cube_processor.h"

#include <cmath>

namespace Metavision {

template<typename InputIt>
EventCubeProcessor<InputIt>::EventCubeProcessor(timestamp delta_t, int event_input_width, int event_input_height,
                                                int num_utbins, bool split_polarity, float max_incr_per_pixel,
                                                float clip_value_after_normalization, float width_scale,
                                                float height_scale) :
    EventPreprocessor<InputIt>(TensorShape({{"C", 1}, {"H", event_input_height}, {"W", event_input_width}}),
                               BaseType::FLOAT32),
    width_(event_input_width),
    // Further normalize to make up for adding more events per cell (when there is a previous rescaling of events)
    normalization_factor_(1.f / max_incr_per_pixel * width_scale * height_scale),
    split_polarity_(split_polarity),
    num_polarities_(split_polarity_ ? 2 : 1),
    num_utbins_(num_utbins),
    clip_value_after_normalization_(clip_value_after_normalization),
    num_utbins_over_delta_t_(static_cast<float>(num_utbins) / delta_t),
    w_h_(event_input_width * event_input_height),
    w_h_p_(w_h_ * num_polarities_) {
    if (num_utbins <= 0)
        throw std::runtime_error("num_utbins should be >0. Got " + std::to_string(num_utbins));
    if (max_incr_per_pixel <= 0)
        throw std::runtime_error("max_incr_per_pixel should be >0. Got " + std::to_string(max_incr_per_pixel));
    if (width_scale <= 0.f || height_scale <= 0.f)
        throw std::runtime_error("Scaling factors for width and height should be > 0. Got " +
                                 std::to_string(width_scale) + " and " + std::to_string(height_scale));
    if (clip_value_after_normalization_ <= 0.f)
        throw std::runtime_error("Clip value after nrmalization should be >0. Got " +
                                 std::to_string(clip_value_after_normalization_));

    const auto network_num_channels_ = num_polarities_ * num_utbins_;
    set_dim(this->output_tensor_shape_, "C", network_num_channels_);
}

template<typename InputIt>
void EventCubeProcessor<InputIt>::set_value(float *buff, const std::size_t buff_size, const int bin, const int p,
                                            const int x, const int y, const float val) const {
    const int idx = x + (y * width_) + p * (w_h_) + bin * (w_h_p_);
    assert(idx >= 0);
    assert(idx < static_cast<int>(buff_size));
    if (clip_value_after_normalization_ != 0.f) {
        buff[idx] =
            std::max(-clip_value_after_normalization_, std::min(clip_value_after_normalization_, buff[idx] + val));
    } else {
        buff[idx] += val;
    }
}

template<typename InputIt>
void EventCubeProcessor<InputIt>::compute(const timestamp cur_frame_start_ts, InputIt begin, InputIt end,
                                          Tensor &tensor) const {
    assert(tensor.type() == BaseType::FLOAT32);
    auto buff            = tensor.data<float>();
    const auto buff_size = tensor.shape().get_nb_values();
    assert(buff_size == this->output_tensor_shape_.get_nb_values());
    for (auto it = begin; it != end; ++it) {
        auto &ev = *it;
        assert((ev.p == 0) || (ev.p == 1));
        assert(ev.t >= cur_frame_start_ts);
        assert(ev.x >= 0);
        assert(ev.x < get_dim(this->output_tensor_shape_, "W"));
        assert(ev.y >= 0);
        assert(ev.y < get_dim(this->output_tensor_shape_, "H"));

        const float ti_star = ((ev.t - cur_frame_start_ts) * num_utbins_over_delta_t_) - 0.5f;
        const int lbin      = floor(ti_star);
        const int rbin      = lbin + 1;

        float left_value  = std::max(0.f, 1.f - std::abs(lbin - ti_star));
        float right_value = 1.f - left_value;

        const int p = split_polarity_ ? ev.p : 0;
        if (!split_polarity_) {
            const int pol = ev.p ? ev.p : -1;
            left_value *= pol;
            right_value *= pol;
        }

        if ((lbin >= 0) && (lbin < num_utbins_)) {
            set_value(buff, buff_size, lbin, p, ev.x, ev.y, left_value * normalization_factor_);
        }
        if (rbin < num_utbins_) {
            set_value(buff, buff_size, rbin, p, ev.x, ev.y, right_value * normalization_factor_);
        }
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_EVENT_CUBE_PROCESSOR_IMPL_H
