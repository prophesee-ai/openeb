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

#ifndef METAVISION_SDK_CORE_ADAPTIVE_RATE_EVENTS_SPLITTER_ALGORITHM_H
#define METAVISION_SDK_CORE_ADAPTIVE_RATE_EVENTS_SPLITTER_ALGORITHM_H

#include "metavision/sdk/base/events/event_cd.h"
#include <cmath>
#include <assert.h>

namespace Metavision {

/// @brief Class used to split a stream of events into slices of variable duration and variable number of events
///
/// This algorithm produces reasonably sharp slices of events, based on the content of the stream itself
/// Internally, it computes the variance per event as a criterion for the sharpness of the current slice of
/// events. An additional criterion is the maximum proportion of active pixels containing both positive and negative
/// events.
///
class AdaptiveRateEventsSplitterAlgorithm {
public:
    /// @brief Constructs a new AdaptiveRateEventsSplitterAlgorithm
    ///
    /// @param height height of the input frame of events
    /// @param width width of the input frame of events
    /// @param thr_var_per_event minimum variance per pixel value to reach before considering splitting the slice
    /// @param downsampling_factor performs a downsampling of the input before computing the statistics. Original
    ///                            coordinates will be multiplied by 2**(-downsampling_factor)
    AdaptiveRateEventsSplitterAlgorithm(int height, int width, float thr_var_per_event = 5e-4f,
                                        int downsampling_factor = 2);

    /// @brief Destructor
    ~AdaptiveRateEventsSplitterAlgorithm(){};

    /// @brief Process a slice of events, and determines if slicing should be performed or not at the end
    ///
    /// @param begin Iterator pointing to the beginning of the events buffer
    /// @param end Iterator pointing to the end of the events buffer
    /// @return true if the slice is ready, false is more events should be gathered before splitting
    template<typename InputIt>
    bool process_events(InputIt begin, InputIt end);

    /// @brief Retrieves the slice of events and resets internal state
    ///
    /// @param out_vec output vector of events
    void retrieve_events(std::vector<EventCD> &out_vec) {
        out_vec.clear();
        out_vec.swap(events_);
        reset_local_variables();
    }

private:
    void reset_local_variables();

    int height_, width_, shift_;
    float thr_var_per_event_;

    int nb_pos_;
    float mean_pos_;
    float var_pos_;
    float prev_var_pos_;
    float prev_var_per_event_pos_;
    int nb_pos_pix_;
    std::vector<std::uint16_t> img_pos_;

    int nb_neg_;
    float mean_neg_;
    float var_neg_;
    float prev_var_neg_;
    float prev_var_per_event_neg_;
    int nb_neg_pix_;
    std::vector<std::uint16_t> img_neg_;

    int nb_both_pos_and_neg_pix_;

    float one_over_height_times_width_;
    float one_over_height_times_width_squared_;

    std::vector<EventCD> events_;

    static constexpr float kMaxRatioBothPix = 0.1f;
};

template<typename InputIt>
bool AdaptiveRateEventsSplitterAlgorithm::process_events(InputIt begin, InputIt end) {
    std::copy(begin, end, std::back_inserter(events_));
    const int two_pow_shift = 1 << shift_;
    for (auto it = begin; it != end; ++it) {
        if ((it->x % two_pow_shift) || (it->y % two_pow_shift)) {
            continue;
        }
        const int x       = it->x >> shift_;
        const int y       = it->y >> shift_;
        const int idx_pix = y * width_ + x;
        if (it->p == 1) {
            if (img_pos_[idx_pix] == 0) {
                nb_pos_pix_++;
                if (img_neg_[idx_pix]) {
                    nb_both_pos_and_neg_pix_++;
                }
            }
            mean_pos_ += one_over_height_times_width_;
            var_pos_ += one_over_height_times_width_squared_ +
                        (2.f * (img_pos_[idx_pix] - mean_pos_) + 1.f) * one_over_height_times_width_;
            img_pos_[idx_pix]++;
            nb_pos_++;
        } else {
            if (img_neg_[idx_pix] == 0) {
                nb_neg_pix_++;
                if (img_pos_[idx_pix]) {
                    nb_both_pos_and_neg_pix_++;
                }
            }
            mean_neg_ += one_over_height_times_width_;
            var_neg_ += one_over_height_times_width_squared_ +
                        (2.f * (img_neg_[idx_pix] - mean_neg_) + 1.f) * one_over_height_times_width_;
            img_neg_[idx_pix]++;
            nb_neg_++;
        }
    }
    float var_per_event_pos = 0.f;
    if (nb_pos_ == 0) {
        assert(var_pos_ == 0.f);
    } else {
        var_per_event_pos = var_pos_ / nb_pos_;
    }
    float var_per_event_neg = 0.f;
    if (nb_neg_ == 0) {
        assert(var_neg_ == 0.f);
    } else {
        var_per_event_neg = var_neg_ / nb_neg_;
    }
    const float ratio_pix_both = nb_both_pos_and_neg_pix_ / (nb_pos_pix_ + nb_neg_pix_ + 1e-5f);

    if ((ratio_pix_both >= kMaxRatioBothPix) ||
        ((var_per_event_neg < prev_var_per_event_neg_) && (var_per_event_neg > thr_var_per_event_)) ||
        ((var_per_event_pos < prev_var_per_event_pos_) && (var_per_event_pos > thr_var_per_event_))) {
        assert(events_.size());
        auto first_ts = events_[0].t;
        auto last_ts  = events_[events_.size() - 1].t;
        return true;
    }
    prev_var_per_event_neg_ = var_per_event_neg;
    prev_var_per_event_pos_ = var_per_event_pos;
    return false;
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_ADAPTIVE_RATE_EVENTS_SPLITTER_ALGORITHM_H
