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

#ifndef METAVISION_SDK_CORE_EVENT_BUFFER_RESLICER_ALGORITHM_IMPL_H
#define METAVISION_SDK_CORE_EVENT_BUFFER_RESLICER_ALGORITHM_IMPL_H

#include <algorithm>

namespace Metavision {

template<bool enable_interruptions>
template<typename InputIt, typename OnEventsCbT>
void EventBufferReslicerAlgorithmT<enable_interruptions>::process_events(InputIt it_input_begin, InputIt it_input_end,
                                                                         OnEventsCbT on_events_cb) {
    if (it_input_begin == it_input_end)
        return;

    if (!has_processing_started_) {
        initialize_processing(it_input_begin->t);
    }

    auto it_input_curr = it_input_begin;
    while (it_input_curr != it_input_end) {
        // If interruptions are handled, check and process
        if (enable_interruptions && BaseEventBufferReslicerAlgorithmT<enable_interruptions>::is_interrupted()) {
            return;
        }
        // Find the next position where the slicing condition is met
        InputIt it_buffer_end;
        const ConditionStatus status = find_next_slicing_condition_met(it_input_curr, it_input_end, it_buffer_end);
        // Process the events until the returned position, if any
        if (it_input_curr != it_buffer_end) {
            n_events_in_current_slice_ += std::distance(it_input_curr, it_buffer_end);
            curr_slice_last_observed_ts_ = std::prev(it_buffer_end)->t;
            on_events_cb(it_input_curr, it_buffer_end);
        }
        // If the slicing condition was met, close the current slice and restart a new one before processing the
        // rest of the input buffer
        if (status != ConditionStatus::NOT_MET) {
            close_and_restart_new_slice(status);
        }
        it_input_curr = it_buffer_end;
    }
}

template<bool enable_interruptions>
template<typename InputIt>
typename EventBufferReslicerAlgorithmT<enable_interruptions>::ConditionStatus
    EventBufferReslicerAlgorithmT<enable_interruptions>::find_next_slicing_condition_met(InputIt it_begin,
                                                                                         InputIt it_end,
                                                                                         InputIt &it_buffer_end) {
    assert(it_begin != it_end);
    // The goal of this function is to look for the end of the current slice, as defined by the slicing condition, and
    // return a status indicating whether and how the end of current slice was met, along with an iterator defining the
    // events to include in the current slice
    switch (condition_.type) {
    case ConditionType::N_EVENTS:
        return find_next_slicing_condition_met_n_events(it_begin, it_end, it_buffer_end);
    case ConditionType::N_US:
        return find_next_slicing_condition_met_n_us(it_begin, it_end, it_buffer_end);
    case ConditionType::MIXED: {
        // In the MIXED case, first look for the position where the N_US slicing condition would be met. If not found,
        // revert to N_EVENTS case. If found, look whether the N_EVENTS slicing condition occurs before the found
        // position. If yes, return that N_EVENTS slicing condition was met with corresponding position. Otherwise,
        // return that N_US slicing condition was met with initially found position.
        if (find_next_slicing_condition_met_n_us(it_begin, it_end, it_buffer_end) != ConditionStatus::NOT_MET) {
            if (find_next_slicing_condition_met_n_events(it_begin, it_buffer_end, it_buffer_end) !=
                ConditionStatus::NOT_MET)
                return ConditionStatus::MET_N_EVENTS;
            return ConditionStatus::MET_N_US;
        }
        return find_next_slicing_condition_met_n_events(it_begin, it_end, it_buffer_end);
    }
    case ConditionType::IDENTITY:
    default: {
        it_buffer_end = it_end;
        return ConditionStatus::MET_AUTOMATIC;
    }
    }
}

template<bool enable_interruptions>
template<typename InputIt>
typename EventBufferReslicerAlgorithmT<enable_interruptions>::ConditionStatus
    EventBufferReslicerAlgorithmT<enable_interruptions>::find_next_slicing_condition_met_n_events(
        InputIt it_begin, InputIt it_end, InputIt &it_buffer_end) {
    if (std::distance(it_begin, it_end) + n_events_in_current_slice_ < condition_.delta_n_events) {
        it_buffer_end = it_end;
        return ConditionStatus::NOT_MET;
    }
    it_buffer_end = it_begin + (condition_.delta_n_events - n_events_in_current_slice_);
    return ConditionStatus::MET_N_EVENTS;
}

template<bool enable_interruptions>
template<typename InputIt>
typename EventBufferReslicerAlgorithmT<enable_interruptions>::ConditionStatus
    EventBufferReslicerAlgorithmT<enable_interruptions>::find_next_slicing_condition_met_n_us(InputIt it_begin,
                                                                                              InputIt it_end,
                                                                                              InputIt &it_buffer_end) {
    const timestamp next_slice_ref_ts = curr_slice_ref_ts_ + condition_.delta_ts;
    if (std::prev(it_end)->t < next_slice_ref_ts) {
        it_buffer_end = it_end;
        return ConditionStatus::NOT_MET;
    }
    it_buffer_end =
        std::lower_bound(it_begin, it_end, next_slice_ref_ts, [](const auto &ev, timestamp ts) { return ev.t < ts; });
    return ConditionStatus::MET_N_US;
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_EVENT_BUFFER_RESLICER_ALGORITHM_IMPL_H
