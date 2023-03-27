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

#include <cassert>
#include "metavision/sdk/core/algorithms/event_buffer_reslicer_algorithm.h"

namespace Metavision {
namespace Detail {

ReslicingCondition ReslicingCondition::make_identity() {
    ReslicingCondition c;
    return c;
}

ReslicingCondition ReslicingCondition::make_n_events(std::size_t delta_n_events) {
    ReslicingCondition c;
    c.type           = ReslicingConditionType::N_EVENTS;
    c.delta_n_events = delta_n_events;
    return c;
}

ReslicingCondition ReslicingCondition::make_n_us(timestamp delta_ts) {
    ReslicingCondition c;
    c.type     = ReslicingConditionType::N_US;
    c.delta_ts = delta_ts;
    return c;
}

ReslicingCondition ReslicingCondition::make_mixed(timestamp delta_ts, std::size_t delta_n_events) {
    ReslicingCondition c;
    c.type           = ReslicingConditionType::MIXED;
    c.delta_ts       = delta_ts;
    c.delta_n_events = delta_n_events;
    return c;
}

} // namespace Detail

template<bool enable_interruptions>
EventBufferReslicerAlgorithmT<enable_interruptions>::EventBufferReslicerAlgorithmT(OnNewSliceCb on_new_slice_cb,
                                                                                   const Condition &condition) {
    set_on_new_slice_callback(on_new_slice_cb);
    set_slicing_condition(condition);
}

template<bool enable_interruptions>
void EventBufferReslicerAlgorithmT<enable_interruptions>::set_on_new_slice_callback(OnNewSliceCb on_new_slice_cb) {
    on_new_slice_cb_ = on_new_slice_cb;
}

template<bool enable_interruptions>
void EventBufferReslicerAlgorithmT<enable_interruptions>::set_slicing_condition(const Condition &condition) {
    condition_ = condition;
    if (has_processing_started_) {
        if ((condition_.is_tracking_events_count() && n_events_in_current_slice_ >= condition_.delta_n_events) ||
            (condition_.is_tracking_duration() &&
             curr_slice_last_observed_ts_ >= curr_slice_ref_ts_ + condition_.delta_ts) ||
            condition_.type == ConditionType::IDENTITY) {
            close_and_restart_new_slice(ConditionStatus::MET_AUTOMATIC);
        }
    }
}

template<bool enable_interruptions>
void EventBufferReslicerAlgorithmT<enable_interruptions>::reset() {
    has_processing_started_ = false;
    if (enable_interruptions)
        BaseEventBufferReslicerAlgorithmT<enable_interruptions>::reset_interruption();
}

template<bool enable_interruptions>
void EventBufferReslicerAlgorithmT<enable_interruptions>::flush() {
    if (has_processing_started_) {
        close_and_restart_new_slice(ConditionStatus::MET_AUTOMATIC);
    }
}

template<bool enable_interruptions>
void EventBufferReslicerAlgorithmT<enable_interruptions>::notify_elapsed_time(timestamp ts) {
    if (!has_processing_started_) {
        initialize_processing(0);
    }
    if (condition_.is_tracking_duration()) {
        timestamp next_slice_ref_ts = curr_slice_ref_ts_ + condition_.delta_ts;
        while (next_slice_ref_ts < ts) {
            // If interruptions are handled, check and process
            if (enable_interruptions && BaseEventBufferReslicerAlgorithmT<enable_interruptions>::is_interrupted()) {
                return;
            }
            close_and_restart_new_slice(ConditionStatus::MET_N_US);
            next_slice_ref_ts += condition_.delta_ts;
        }
    }
}

template<bool enable_interruptions>
void EventBufferReslicerAlgorithmT<enable_interruptions>::initialize_processing(timestamp ts) {
    if (condition_.is_tracking_duration()) {
        curr_slice_ref_ts_ = condition_.delta_ts * (ts / condition_.delta_ts);
    } else {
        curr_slice_ref_ts_ = ts;
    }
    curr_slice_last_observed_ts_ = ts;
    n_events_in_current_slice_   = 0;
    has_processing_started_      = true;
}

template<bool enable_interruptions>
void EventBufferReslicerAlgorithmT<enable_interruptions>::close_and_restart_new_slice(ConditionStatus status) {
    assert(has_processing_started_);
    const timestamp curr_slice_ts_upper_bound =
        (status == ConditionStatus::MET_N_US ? curr_slice_ref_ts_ + condition_.delta_ts : curr_slice_last_observed_ts_);
    on_new_slice_cb_(status, curr_slice_ts_upper_bound, n_events_in_current_slice_);
    curr_slice_ref_ts_           = curr_slice_ts_upper_bound;
    curr_slice_last_observed_ts_ = curr_slice_ts_upper_bound;
    n_events_in_current_slice_   = 0;
}

// Template instanciation
template class EventBufferReslicerAlgorithmT<false>;
template class EventBufferReslicerAlgorithmT<true>;

} // namespace Metavision
