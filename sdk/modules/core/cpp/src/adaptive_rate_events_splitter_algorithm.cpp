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

#include "metavision/sdk/core/algorithms/adaptive_rate_events_splitter_algorithm.h"

namespace Metavision {

AdaptiveRateEventsSplitterAlgorithm::AdaptiveRateEventsSplitterAlgorithm(int height, int width, float thr_var_per_event,
                                                                         int downsampling_factor) {
    if (downsampling_factor < 0) {
        throw std::invalid_argument("Error: downsampling_factor must be >= 0");
    }
    int shift          = downsampling_factor;
    height_            = height >> shift;
    width_             = width >> shift;
    shift_             = shift;
    thr_var_per_event_ = thr_var_per_event;
    assert(width_ > 0);
    assert(height_ > 0);
    reset_local_variables();
    one_over_height_times_width_         = 1.f / (height_ * width_);
    one_over_height_times_width_squared_ = one_over_height_times_width_ * one_over_height_times_width_;
}

void AdaptiveRateEventsSplitterAlgorithm::reset_local_variables() {
    img_pos_.resize(height_ * width_);
    img_neg_.resize(height_ * width_);
    std::fill(img_pos_.begin(), img_pos_.end(), 0);
    std::fill(img_neg_.begin(), img_neg_.end(), 0);

    nb_pos_                 = 0;
    mean_pos_               = 0.f;
    var_pos_                = 0.f;
    prev_var_pos_           = 0.f;
    prev_var_per_event_pos_ = 0.f;
    nb_pos_pix_             = 0;

    nb_neg_                 = 0;
    mean_neg_               = 0.f;
    var_neg_                = 0.f;
    prev_var_neg_           = 0.f;
    prev_var_per_event_neg_ = 0.f;
    nb_neg_pix_             = 0;

    nb_both_pos_and_neg_pix_ = 0;

    events_.clear();
}

} // namespace Metavision