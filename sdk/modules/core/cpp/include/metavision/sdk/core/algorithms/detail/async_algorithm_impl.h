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

#ifndef METAVISION_SDK_CORE_DETAIL_ASYNC_ALGORITHM_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_ASYNC_ALGORITHM_IMPL_H

namespace Metavision {
template<typename Impl>
inline void AsyncAlgorithm<Impl>::set_processing_n_events(const int delta_n_events) {
    assert(delta_n_events > 0);

    processing_     = Processing::N_EVENTS;
    delta_n_events_ = delta_n_events;

    if (is_initialized_) {
        if (n_processed_events_ >= delta_n_events_) {
            processing_ts_ = last_event_ts_ + 1;

            child_cast().process_async(processing_ts_, n_processed_events_);

            next_processing_n_events_ = delta_n_events_;
            n_processed_events_       = 0;
        } else {
            next_processing_n_events_ = delta_n_events_ - n_processed_events_;
        }
    }
}

template<typename Impl>
inline void AsyncAlgorithm<Impl>::set_processing_n_us(const timestamp delta_ts) {
    assert(delta_ts > 0);

    processing_ = Processing::N_US;
    delta_ts_   = delta_ts;

    if (is_initialized_) {
        if (processing_ts_ + delta_ts_ <= last_event_ts_) {
            processing_ts_ = last_event_ts_ + 1;

            child_cast().process_async(processing_ts_, n_processed_events_);
        }

        next_processing_ts_ = processing_ts_ + delta_ts_;
    }
}

template<typename Impl>
inline void AsyncAlgorithm<Impl>::set_processing_mixed(const int delta_n_events, const timestamp delta_ts) {
    assert(delta_n_events > 0 && delta_ts > 0);

    processing_     = Processing::MIXED;
    delta_n_events_ = delta_n_events;
    delta_ts_       = delta_ts;

    if (is_initialized_) {
        if ((processing_ts_ + delta_ts_ <= last_event_ts_) || (delta_n_events_ <= n_processed_events_)) {
            processing_ts_ = last_event_ts_ + 1;

            child_cast().process_async(processing_ts_, n_processed_events_);

            next_processing_n_events_ = delta_n_events_;
            n_processed_events_       = 0;

        } else {
            next_processing_n_events_ = delta_n_events_ - n_processed_events_;
        }

        next_processing_ts_ = processing_ts_ + delta_ts_;
    }
}

template<typename Impl>
inline void AsyncAlgorithm<Impl>::set_processing_sync() {
    processing_ = Processing::SYNC;

    if (is_initialized_) {
        processing_ts_ = last_event_ts_ + 1;

        child_cast().process_async(processing_ts_, n_processed_events_);
        n_processed_events_ = 0;
    }
}

template<typename Impl>
inline void AsyncAlgorithm<Impl>::set_processing_external() {
    processing_ = Processing::EXTERNAL;

    if (is_initialized_) {
        processing_ts_ = last_event_ts_ + 1;

        child_cast().process_async(processing_ts_, n_processed_events_);
        n_processed_events_ = 0;
    }
}

template<typename Impl>
void AsyncAlgorithm<Impl>::reset() {
    is_initialized_ = false;
}

template<typename Impl>
inline void AsyncAlgorithm<Impl>::flush() {
    if (is_initialized_) {
        processing_ts_ = last_event_ts_ + 1;

        child_cast().process_async(processing_ts_, n_processed_events_);

        switch (processing_) {
        case Processing::N_EVENTS:
            next_processing_n_events_ = delta_n_events_;
            break;
        case Processing::N_US:
            next_processing_ts_ = processing_ts_ + delta_ts_;
            break;
        case Processing::MIXED:
            next_processing_n_events_ = delta_n_events_;
            next_processing_ts_       = processing_ts_ + delta_ts_;
            break;
        default:
            break;
        }
    }

    n_processed_events_ = 0;
}

template<typename Impl>
template<typename InputIt>
inline void AsyncAlgorithm<Impl>::process_events(InputIt it_begin, InputIt it_end) {
    if (it_begin == it_end)
        return;

    Impl &impl = child_cast();
    if (!is_initialized_) {
        initialize(it_begin->t);
        impl.on_init(processing_ts_);
    }

    // Get first iterator
    InputIt it_current = it_begin;

    while (it_current != it_end) {
        // Get last iterator
        InputIt to_it;
        const bool process_async = find_last_it(it_current, it_end, to_it);
        const auto num_events    = static_cast<int>(std::distance(it_current, to_it));

        // find_last_it can return to_it = it_current when process_events is called with a buffer of events that is so
        // recent that it triggers past asynchronous conditions.
        if (num_events > 0) {
            n_processed_events_ += num_events;

            // Call child function to process current buffer
            impl.process_online(it_current, to_it);
        }

        if (process_async) {
            // Call child function to process the state asynchronously
            impl.process_async(processing_ts_, n_processed_events_);
            n_processed_events_ = 0;
        }

        it_current = to_it;
    }

    last_event_ts_ = std::prev(it_end)->t;
}

template<typename Impl>
template<typename InputIt>
inline void AsyncAlgorithm<Impl>::process_events(const timestamp ts, InputIt it_begin, InputIt it_end) {
    // Process the buffer
    process_events(it_begin, it_end);

    // Initialize with ts if it was not done in process_events
    if (!is_initialized_) {
        initialize(ts);
        child_cast().on_init(processing_ts_);
    }

    // All events were processed. We use the timestamp instead of the last event to determine the end of the time slice
    if (processing_ == Processing::N_US || processing_ == Processing::MIXED) {
        while (next_processing_ts_ <= ts) {
            processing_ts_ = next_processing_ts_;
            next_processing_ts_ += delta_ts_;
            child_cast().process_async(processing_ts_, n_processed_events_);
            n_processed_events_ = 0;
        }
    }
}

template<typename Impl>
template<typename InputIt>
inline bool AsyncAlgorithm<Impl>::find_last_it(const InputIt it_begin, const InputIt it_end, InputIt &to_it) {
    switch (processing_) {
    case Processing::N_EVENTS: {
        const int buffer_size = static_cast<int>(std::distance(it_begin, it_end));
        if (buffer_size < next_processing_n_events_) {
            // Not enough events for the next processing
            to_it          = it_end;
            processing_ts_ = std::prev(to_it)->t;
            next_processing_n_events_ -= buffer_size;

            return false;
        } else {
            to_it          = it_begin + next_processing_n_events_;
            processing_ts_ = std::prev(to_it)->t;

            // Reset next_processing_n_events_
            next_processing_n_events_ = delta_n_events_;
            return true;
        }
    }

    case Processing::N_US: {
        if (std::prev(it_end)->t >= next_processing_ts_) {
            // Find first event with a timestamp higher than (next_processing_ts_ - 1)
            to_it = std::lower_bound(it_begin, it_end, next_processing_ts_,
                                     [](const auto &ev, const timestamp &ts) { return ev.t < ts; });

            processing_ts_ = next_processing_ts_;
            next_processing_ts_ += delta_ts_;
            return true;
        } else {
            // No need to run upper bound in this case -> this is faster than running it to have it_end as result (quick
            // bench was made).

            to_it = it_end;
            return false;
        }
    }

    case Processing::MIXED: {
        const int buffer_size = static_cast<int>(std::distance(it_begin, it_end));

        // ------------------------------
        // If excess of events, look if timeslice has been passed
        if (buffer_size >= next_processing_n_events_) {
            to_it = it_begin + next_processing_n_events_;
            if (std::prev(to_it)->t >= next_processing_ts_) {
                to_it = std::lower_bound(it_begin, to_it, next_processing_ts_,
                                         [](const auto &ev, const timestamp &ts) { return ev.t < ts; });

                processing_ts_ = next_processing_ts_;
                next_processing_ts_ += delta_ts_;
            } else {
                processing_ts_ = std::prev(to_it)->t;
                // events in next time slice goes from processing_ts to processing_ts_ + delta_ts_ excluded. This is
                // because nothing guarantees that the event after this one won't have the same timestamp. It must be
                // included in the time slice
                next_processing_ts_ = processing_ts_ + delta_ts_;
            }

            // Reset count conditions
            next_processing_n_events_ = delta_n_events_;
            return true;
        }

        // ------------------------------
        // Look for time slice
        if (std::prev(it_end)->t >= next_processing_ts_) {
            // Find first event with a timestamp higher than next_processing_ts_
            to_it = std::lower_bound(it_begin, it_end, next_processing_ts_,
                                     [](const auto &ev, const timestamp &ts) { return ev.t < ts; });

            processing_ts_ = next_processing_ts_;
            next_processing_ts_ += delta_ts_;
            next_processing_n_events_ = delta_n_events_;
            return true;
        }
        // ------------------------------
        // No condition fulfilled
        to_it          = it_end;
        processing_ts_ = std::prev(to_it)->t;
        next_processing_n_events_ -= buffer_size;
        return false;
    }

    case Processing::EXTERNAL:
        processing_ts_ = std::prev(it_end)->t + 1;
        to_it          = it_end;
        return false; // None police: do not call process async

    case Processing::SYNC:
    default:
        // Default behavior: treat the slice synchronously + 1
        processing_ts_ = std::prev(it_end)->t + 1;
        to_it          = it_end;
        return true;
    }
}

template<typename Impl>
void AsyncAlgorithm<Impl>::initialize(timestamp ts) {
    switch (processing_) {
    case Processing::N_EVENTS: {
        processing_ts_            = ts;
        next_processing_ts_       = processing_ts_;
        next_processing_n_events_ = delta_n_events_;
        break;
    }
    case Processing::MIXED:
        next_processing_n_events_ = delta_n_events_;
        [[fallthrough]];
    case Processing::N_US: {
        processing_ts_      = delta_ts_ * (ts / delta_ts_);
        next_processing_ts_ = processing_ts_ + delta_ts_;
        break;
    }
    default:
        processing_ts_ = ts;
        break;
    }
    is_initialized_     = true;
    n_processed_events_ = 0;
}
} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_ASYNC_ALGORITHM_IMPL_H
