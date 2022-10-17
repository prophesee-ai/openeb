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

#ifndef METAVISION_SDK_CORE_GENERIC_PRODUCER_ALGORITHM_H
#define METAVISION_SDK_CORE_GENERIC_PRODUCER_ALGORITHM_H

#include <limits>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <sstream>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/utils/detail/ring.h"
#include "metavision/sdk/core/utils/timing_profiler.h"

class GenericProducerAlgorithm_GTest;
namespace Metavision {

/// @brief Allows insertion and storage of events from any source via @ref register_new_event_buffer.
///
/// The events will then be produced in a chronological manner with @ref process_events.
///
/// A maximum duration defined by the difference of timestamps between first
/// and last stored event can be set to limit the number of events stored. To
/// enforce this constraint, the producer will either wait for events to be consumed
/// or drop enough events to be able to insert new events, @ref set_max_duration_stored
/// and @ref set_allow_drop_when_overfilled.
///
/// Conversely, a timeout can be set, to avoid the producer waiting indefinitely for
/// events to be inserted. According to the timeout value, the producer will either not wait,
/// wait indefinitely or wait for a predefined amount of time before returning the events,
///
/// @sa @ref set_timeout.
template<class EventType>
class GenericProducerAlgorithm {
public:
    /// @brief Constructor
    /// @param timeout Maximum time to wait for events (in us). If equal 0, no timeout is set (equivalent to
    /// infinite timeout); if negative, there will be no wait at all.
    /// @param max_events_per_second Maximum event rate when "processing" events up to some time,
    /// the latest k events will be output, where k is max_events_per_second * (req_ts - last_ts), the
    /// preceding events will simply be dropped
    /// @param max_duration_stored Maximum time difference (in us) between first and last event stored by the producer
    /// if the producer already stores enough events, when new events must be added,
    /// the oldest events will be dropped if \@ref allow_drop_when_overfilled is true,
    /// otherwise the producer will wait for enough events to be consumed before continuing
    /// @param allow_drop_when_overfilled If true, the oldest events (i.e. event for which the timestamp if older than
    /// latest.t - max_duration_stored) are dropped, otherwise, the producer blocks until the oldest events are consumed
    GenericProducerAlgorithm(timestamp timeout = 0, uint32_t max_events_per_second = 0,
                             timestamp max_duration_stored   = std::numeric_limits<timestamp>::max(),
                             bool allow_drop_when_overfilled = false) :
        ring_event_(256, false),
        timeout_(timeout),
        max_events_per_microseconds_(static_cast<float>(max_events_per_second) / 1000000),
        max_duration_stored_(max_duration_stored),
        allow_drop_when_overfilled_(allow_drop_when_overfilled) {}
    virtual ~GenericProducerAlgorithm() {}

    /// @brief method to add a new buffer of events
    ///
    /// When inserting events, a maximum duration stored by the producer can be set.
    /// It is disabled by default, so that the producer storage may grow indefinitely if
    /// no events are produced, or the production is too slow.
    /// If a maximum duration is set, the producer will never store events such that the
    /// difference between the first and last event stored is greater than the maximum duration.
    /// To achieve this effect, it will either wait for events to be produced if
    /// drop is disabled when the producer has overfilled @ref set_allow_drop_when_overfilled,
    /// or it will drop enough events so that this condition is met if drop is enabled.
    template<typename IteratorEv>
    void register_new_event_buffer(IteratorEv start, IteratorEv end) {
        if (start != end) {
            if (max_duration_stored_ == std::numeric_limits<timestamp>::max()) {
                //  no max_duration set, just insert the events
                MV_SDK_LOG_DEBUG() << "GenericProducerAlgorithm: no max_duration set, just insert the events";
                enqueue(start, end);
            } else {
                MV_SDK_LOG_DEBUG() << "GenericProducerAlgorithm: max_duration_stored_ is not infinite:"
                                   << max_duration_stored_;
                MV_SDK_LOG_DEBUG() << "GenericProducerAlgorithm: we want to insert a chunk of"
                                   << std::distance(start, end) << "events, between" << start->t << "and"
                                   << (end - 1)->t << "diff: " << (end - 1)->t - start->t;
                if ((end - 1)->t - start->t > max_duration_stored_) {
                    MV_SDK_LOG_DEBUG() << "GenericProducerAlgorithm: range is larger than max_duration_stored_:"
                                       << max_duration_stored_;
                    // the range of events is bigger than max_duration
                    if (allow_drop_when_overfilled_) {
                        // consider increasing the `max_duration_stored` if this message shows up too often
                        MV_SDK_LOG_WARNING() << "Too many events already queued, dropping the oldest ones...";
                        clear_and_enqueue(start, end, (end - 1)->t - max_duration_stored_);
                    } else {
                        throw std::runtime_error(
                            "Range of events to insert is too big : max_duration < diff(start, end). max_duration = " +
                            std::to_string(max_duration_stored_) +
                            " and diff(start, end) = " + std::to_string((end - 1)->t - start->t));
                    }
                } else {
                    // the range of events is fine, it should fit in the ring
                    MV_SDK_LOG_DEBUG() << "GenericProducerAlgorithm: the range of events is fine, it should "
                                          "fit in the ring";
                    MV_SDK_LOG_DEBUG() << "GenericProducerAlgorithm: ring.data_available():"
                                       << ring_event_.data_available();
                    MV_SDK_LOG_DEBUG() << "GenericProducerAlgorithm: ring first time:" << ring_event_.get_first_time()
                                       << "ring last time:" << ring_event_.get_last_time();
                    MV_SDK_LOG_DEBUG() << "GenericProducerAlgorithm: diff from ring first to last event to add:"
                                       << (end - 1)->t - ring_event_.get_first_time();
                    if (ring_event_.data_available() &&
                        (end - 1)->t - ring_event_.get_first_time() > max_duration_stored_) {
                        // ... but there are already too many events
                        if (allow_drop_when_overfilled_) {
                            // consider increasing the `max_duration_stored` if this message shows up too often
                            MV_SDK_LOG_TRACE() << "Too many events already queued, dropping the oldest ones...";
                            clear_and_enqueue(start, end);
                        } else {
                            // we can't drop, the range is OK but there are too many elements already inserted ...
                            // wait until we can insert the whole range
                            if (!wait_to_enqueue(start, end)) {
                                // set_source_done was set while we were waiting!
                                return;
                            }
                            enqueue(start, end);
                        }
                    } else {
                        enqueue(start, end);
                    }
                }
            }
        }
    }

    /// @brief Produces events up to some timestamp
    ///
    /// If the timeout has a positive value, it will wait at most timeout us before
    /// returning the events "generated".
    /// If the timeout has a negative value, it will immediately return the events
    /// generated.
    /// If the timeout is zero, it will wait until at least one event with a timestamp
    /// greater than ts is registered, before returning the events with a timestamp
    /// less or equal to ts.
    ///
    /// @param ts Timestamp before which to include events.
    /// @param inserter Output iterator or back inserter
    /// @param timing_profiler Profiler to debug
    template<class OutputIt, typename TimingProfilerType = TimingProfiler<false>>
    void process_events(timestamp ts, OutputIt inserter,
                        TimingProfilerType *timing_profiler = TimingProfilerType::instance()) {
        std::ostringstream oss;
        MV_SDK_LOG_DEBUG() << "--> GenericProducerAlgorithm::process() with ts:" << ts;
        wanted_ts_ = ts;
        if (ring_event_.get_last_time() < ts) {
            typename TimingProfilerType::TimedOperation t("CD Producer Idle", timing_profiler);
            Metavision::timestamp timeout = timeout_;
            if (timeout > 0) {
                // Wait until events up to time ts have been received or timeout reached
                wait_to_dequeue_with_timeout(ts, timeout);
            } else if (timeout == 0) {
                // Wait until events up to time ts have been received
                wait_to_dequeue(ts);
            }
        }
        float max_events_per_deltat = max_events_per_microseconds_;
        if (max_events_per_deltat > 0) {
            max_events_per_deltat *= ts - last_processed_ts_;
            dequeue_with_drop(inserter, ts, max_events_per_deltat);
        } else {
            dequeue(inserter, ts);
        }
        last_processed_ts_ = ts;
        MV_SDK_LOG_DEBUG() << "GenericProducerAlgorithm: before overfilled_wait_cond_.notify_all()";
        overfilled_wait_cond_.notify_all();
        MV_SDK_LOG_DEBUG() << "--> GenericProducerAlgorithm::process() with ts:" << ts;
    }

    /// @brief Sets timeout
    /// @param timeout Maximum time to wait for events (in us). If equal 0, no timeout is set (equivalent to
    /// infinite timeout); if negative, there will be no wait at all.
    void set_timeout(timestamp timeout) {
        std::unique_lock<std::mutex> lock(underfilled_wait_mut_);
        timeout_ = timeout;
    }

    /// @brief Gets timeout
    /// @return Timeout
    timestamp get_timeout() const {
        return timeout_;
    }

    /// @brief Sets max events per second
    /// @param max_events_per_second Maximum event rate when "processing" events up to some time,
    /// the latest k events will be output, where k is max_events_per_second * (req_ts - last_ts), the
    /// preceding events will simply be dropped
    void set_max_events_per_second(uint32_t max_events_per_second) {
        std::unique_lock<std::mutex> lock(underfilled_wait_mut_);
        max_events_per_microseconds_ = static_cast<float>(max_events_per_second) / 1000000;
    }

    /// @brief Gets max events per second
    /// @return Max events per second
    uint32_t get_max_events_per_second() const {
        return max_events_per_microseconds_ * 1000000;
    }

    /// @brief Sets max duration stored
    /// @param max_duration_stored Maximum time difference (in us) between first and last event stored by the producer
    /// if the producer already stores enough events, when new events must be added,
    /// the oldest events will be dropped if \@ref allow_drop_when_overfilled is true,
    /// otherwise the producer will wait for enough events to be consumed before continuing
    void set_max_duration_stored(timestamp max_duration_stored) {
        std::unique_lock<std::mutex> lock(overfilled_wait_mut_);
        max_duration_stored_ = max_duration_stored;
    }

    /// @brief Gets max duration stored
    /// @return Max duration stored
    uint32_t get_max_duration_stored() const {
        return max_duration_stored_;
    }

    /// @brief Enables/disables drop
    /// @param allow_drop_when_overfilled If true, the oldest events (i.e. event for which the timestamp if older than
    /// latest.t - max_duration_stored) are dropped, otherwise, the producer blocks until the oldest events are consumed
    void set_allow_drop_when_overfilled(bool allow_drop_when_overfilled) {
        std::unique_lock<std::mutex> lock(overfilled_wait_mut_);
        allow_drop_when_overfilled_ = allow_drop_when_overfilled;
    }

    /// @brief Gets value of drop when overfilled
    /// @return true if the oldest events (i.e. event for which the timestamp if older than
    /// latest.t - max_duration_stored) are dropped, false otherwise
    bool get_allow_drop_when_overfilled() const {
        return allow_drop_when_overfilled_;
    }

    /// @brief Sets source as done to let the producer know that no new events will be received.
    void set_source_as_done() {
        {
            std::unique_lock<std::mutex> lock(underfilled_wait_mut_);
            std::unique_lock<std::mutex> lock2(overfilled_wait_mut_);
            source_is_done_ = true;
        }
        underfilled_wait_cond_.notify_all();
        overfilled_wait_cond_.notify_all();
    }

    /// @brief Checks if source is done providing events
    /// @return true if source is done, false otherwise
    bool is_source_done() const {
        return source_is_done_;
    }

    /// @brief Checks if producer has finished processing all input events
    /// @return true if all events have been processed, false otherwise
    bool is_done() const {
        return source_is_done_ && !ring_event_.data_available();
    }

    /// @brief Get time of latest event received
    /// @return Time of latest event
    timestamp latest_event_timestamp_available() const {
        return ring_event_.get_last_time();
    }

private:
    template<typename IteratorEv>
    bool wait_to_enqueue(IteratorEv, IteratorEv end) {
        std::unique_lock<std::mutex> lock(overfilled_wait_mut_);
        overfilled_wait_cond_.wait(lock, [this, &end] {
            if (Metavision::LogLevel::Debug >= getLogLevel()) {
                bool ring_event_data_not_available = !ring_event_.data_available();
                bool last_minus_first_smaller_than_max_duration_stored =
                    ((end - 1)->t - ring_event_.get_first_time() <= max_duration_stored_);
                bool source_is_done = this->source_is_done_;
                MV_SDK_LOG_DEBUG() << Log::function << "ring_event_data_not_available:" << ring_event_data_not_available
                                   << "last_minus_first_smaller_than_max_duration_stored:"
                                   << last_minus_first_smaller_than_max_duration_stored
                                   << "ring_event_.get_first_time():" << ring_event_.get_first_time()
                                   << "(end - 1)->t:" << (end - 1)->t
                                   << "diff:" << (end - 1)->t - ring_event_.get_first_time()
                                   << "max_duration_stored_:" << max_duration_stored_
                                   << "source_is_done:" << source_is_done;
            }
            return !ring_event_.data_available() ||
                   (end - 1)->t - ring_event_.get_first_time() <= max_duration_stored_ || this->source_is_done_;
        });

        return !this->source_is_done_;
    }

    // empty the ring, then insert from the range [start, end) all events e
    // such that e.t >= min_start_time
    template<typename IteratorEv>
    void clear_and_enqueue(IteratorEv start, IteratorEv end, timestamp min_start_time = -1) {
        if (min_start_time >= 0) {
            EventType ev;
            ev.t  = min_start_time;
            start = std::lower_bound(start, end, ev, [](const EventType &ev1, const EventType &ev2) {
                return Metavision::detail::get_time(ev1) < Metavision::detail::get_time(ev2);
            });
        }

        {
            std::unique_lock<std::mutex> lock(underfilled_wait_mut_);
            std::unique_lock<std::mutex> lock2(overfilled_wait_mut_);
            ring_event_.drop();
            ring_event_.add(start, end);
        }
        notify_if_needed();
    }

    template<typename IteratorEv>
    void enqueue(IteratorEv start, IteratorEv end) {
        {
            std::unique_lock<std::mutex> lock(underfilled_wait_mut_);
            std::unique_lock<std::mutex> lock2(overfilled_wait_mut_);
            ring_event_.add(start, end);
        }
        notify_if_needed();
    }

    void notify_if_needed() {
        Metavision::timestamp last_ts   = ring_event_.get_last_time();
        Metavision::timestamp wanted_ts = wanted_ts_;
        bool already_notified           = last_notified_ts_ >= wanted_ts;
        if (!already_notified && wanted_ts <= last_ts) {
            underfilled_wait_cond_.notify_all();
            last_notified_ts_ = last_ts;
        }
    }

    bool wait_to_dequeue_with_timeout(timestamp ts, timestamp timeout) {
        std::unique_lock<std::mutex> lock(underfilled_wait_mut_);
        underfilled_wait_cond_.wait_for(lock, std::chrono::duration<timestamp, std::micro>(timeout_), [this, &ts] {
            return ring_event_.get_last_time() >= ts || this->source_is_done_;
        });

        MV_SDK_LOG_DEBUG() << "wait_to_enqueue(): source_is_done:" << source_is_done_;
        MV_SDK_LOG_DEBUG() << "wait_to_dequeue_with_timeout(): ts:" << ts;
        MV_SDK_LOG_DEBUG() << "wait_to_dequeue_with_timeout(): source_is_done_:" << this->source_is_done_;
        MV_SDK_LOG_DEBUG() << "wait_to_dequeue_with_timeout(): ring_event_.get_last_time():"
                           << ring_event_.get_last_time() << "\n";
        return !this->source_is_done_;
    }

    bool wait_to_dequeue(timestamp ts) {
        std::unique_lock<std::mutex> lock(underfilled_wait_mut_);
        underfilled_wait_cond_.wait(lock,
                                    [this, &ts] { return ring_event_.get_last_time() >= ts || this->source_is_done_; });
        return !this->source_is_done_;
    }
    template<typename OutputIt>
    void dequeue_with_drop(OutputIt inserter, timestamp ts, float max_events_per_deltat) {
        std::unique_lock<std::mutex> lock(underfilled_wait_mut_);
        std::unique_lock<std::mutex> lock2(overfilled_wait_mut_);
        ring_event_.fill_buffer_to_drop_max_events(inserter, ts, max_events_per_deltat);
    }

    template<typename OutputIt>
    void dequeue(OutputIt inserter, timestamp ts) {
        std::unique_lock<std::mutex> lock(underfilled_wait_mut_);
        std::unique_lock<std::mutex> lock2(overfilled_wait_mut_);
        ring_event_.fill_buffer_to(inserter, ts);
    }

    Metavision::detail::Ring<EventType> ring_event_;
    std::condition_variable underfilled_wait_cond_;
    mutable std::mutex underfilled_wait_mut_;
    std::condition_variable overfilled_wait_cond_;
    mutable std::mutex overfilled_wait_mut_;

    timestamp last_notified_ts_  = 0;
    timestamp last_processed_ts_ = 0;
    std::atomic<timestamp> wanted_ts_{0};
    std::atomic<timestamp> timeout_{0};
    std::atomic<float> max_events_per_microseconds_{-1};
    std::atomic<timestamp> max_duration_stored_{0};
    std::atomic<bool> allow_drop_when_overfilled_{false};

    bool source_is_done_ = false;
    friend class ::GenericProducerAlgorithm_GTest;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_GENERIC_PRODUCER_ALGORITHM_H
