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

#ifndef METAVISION_SDK_CORE_ASYNC_ALGORITHM_H
#define METAVISION_SDK_CORE_ASYNC_ALGORITHM_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief An asynchronous events processor class.
///
/// Here, asynchronous means that the output of the processing is at variable frequency.
/// This is useful when one wants to apply a process on events on the fly, and another specific one when a
/// condition is fulfilled.
///
/// As opposed to frames that arrive at a fixed frequency, events are asynchronous. The event rate depends on the
/// activity, and so the rate varies with time. In the context of event-based processing, it is often that one wants to
/// apply a processing to events on the fly (online processing such as filling an histogram for instance) and another
/// process when it is considered that enough events have been received (for example in term of events processed count,
/// or time slice of events).
///
/// The only entry point of this algorithm is the public @ref process_events method. This method will then call two
/// internal processing methods (process_online and process_async) that should be defined by the user and be private.
/// While the first one processes all events that are passed to @ref process_events, the latter one is called whenever
/// the asynchronous condition is met.
///
/// The asynchronous condition is user defined, and set via the various 'set' methods. See @ref Processing for more
/// details.
///
/// @warning This class uses a Curiously Recursive Template Pattern design. The input template parameter is expected to
/// inherit from this class and this class should be declared as friend of the derived class.
///
/// @note Also see EventBufferReslicerAlgorithm, an implementation that offers a similar functionality but is designed
/// to be used via aggregation rather than via inheritance.
///
/// @tparam Impl The Asynchronous process implementation.
template<typename Impl>
class AsyncAlgorithm {
public:
    /// @brief Processing types
    ///
    /// Processing policies that define the state to rely on to call the asynchronous process (process_async).
    ///
    /// N_EVENTS: event count processing policy. Relies on the number of events processed.
    /// N_US: time slice processing policy. Relies on the timestamp of the input events. A time slice T holds events
    /// between [(n-1)*T; n*T[.
    /// MIXED: a mix between N_US and N_EVENTS processing policy. In this policy, the time slice has priority over the
    /// events count.
    /// SYNC: synchronous condition. process_async is called at the end of the process_events method.
    /// EXTERNAL: Relies on an external condition. process_async is called at each flush call.
    enum class Processing { N_EVENTS, N_US, MIXED, SYNC, EXTERNAL };

    AsyncAlgorithm()  = default;
    ~AsyncAlgorithm() = default;

    /// @brief Function to call process_async every n events
    /// @note This call can trigger a flush in case where the new condition is already satisfied by the events processed
    /// so far. However, be careful in that case because the produced events buffer is likely to be bigger than what
    /// expected with the new condition. Indeed, if you decrease the @p delta_n_events condition, the condition gets
    /// more strict and you might end up with a buffer having an intermediate size:
    /// [       delta_n1       ] -> [     tmp     ] -> [ delta_n2 ], with [ N ] being a buffer of size N events and
    /// delta_n1 > tmp >= delta_n2.
    /// In case where you increase the @p delta_n_events condition, the condition gets less strict and the transition is
    /// perfect, no flush occurs:
    /// [ delta_n1 ] -> [    delta_n2     ] with delta_n1 < delta_n2
    void set_processing_n_events(const int delta_n_events);

    /// @brief Getter to retrieve the number of events between two consecutive process_async calls (in N_EVENTS mode).
    int get_processing_n_events() {
        return delta_n_events_;
    }

    /// @brief Function to call process_async every n microseconds
    /// @note This call can trigger a flush in case where the new condition is already satisfied by the events processed
    /// so far. However, be careful in that case because the produced time slice is likely to be longer than what
    /// expected with the new condition. Indeed, if you decrease the @p delta_ts condition, the condition gets more
    /// strict and you might end up with a buffer having an intermediate duration:
    /// [       dt_1       ] -> [     tmp     ] -> [ dt_2 ], with [ dt ] being a time slice of duration dt us and
    /// dt_1 > tmp >= dt_2.
    /// In case where you increase the @p delta_ts condition, the condition gets less strict and the transition is
    /// perfect, no flush occurs:
    /// [ dt_1 ] -> [    dt_2     ] with dt_1 < dt_2
    void set_processing_n_us(const timestamp delta_ts);

    /// @brief Getter to retrieve the period at which process_async gets called (in N_US mode).
    timestamp get_processing_n_us() {
        return delta_ts_;
    }

    /// @brief Function to call process_async every n events and n microseconds. The processing is done
    /// if at least one of the conditions is fulfilled
    /// @note This call can trigger a flush in case where at least one of the new conditions is already satisfied by the
    /// events processed so far. However, be careful in that case because the produced time slice / events buffer is
    /// likely to be bigger (i.e. in terms of number of events) and/or longer (i.e. in terms of duration) than what
    /// expected with the new condition (see @ref set_processing_n_events and @ref set_processing_n_us for further
    /// details).
    void set_processing_mixed(const int delta_n_events, const timestamp delta_ts);

    /// @brief Function to call process_async after each process online
    /// @note This call can trigger a flush if some events have already been processed
    void set_processing_sync();

    /// @brief Function to only call process_events without calling process_async
    ///
    /// This is especially useful if the condition to trigger the creation of a buffer is independent from the content
    /// of the processed events (for instance, an external trigger events, an other algorithm condition, etc.). The
    /// user must then call @ref flush when the condition is fulfilled.
    ///
    /// @note This call can trigger a flush if some events have already been processed
    void set_processing_external();

    /// @brief Resets the internal state of the algorithm
    ///
    /// This is to be called when one wants to process events older than those already processed (e.g. in the case one
    /// wants to switch the source producing the events).
    ///
    /// @note This method doesn't change the algorithm's processing mode (see @ref Processing) nor flushes the ongoing
    /// time slice. It is the user's responsibility to call @ref flush before this method to retrieve the incomplete
    /// time slice if needed.
    void reset();

    /// @brief Forces a call to process_async
    ///
    /// The resulting processed time slice corresponds to all the events processed since the last call to process_async
    /// (i.e. the time slice's timestamp is the last processed event's timestamp + 1)
    ///
    /// @note The internal state is updated so that the next time slice will start just after this one (i.e.
    /// [last event's timestamp + 1, next_processing_ts_[).
    inline void flush();

    /// @brief Processes a buffer of events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    template<typename InputIt>
    inline void process_events(InputIt it_begin, InputIt it_end);

    /// @brief Processes a buffer of events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param ts End timestamp of the buffer. Used if higher than the timestamp of the last event
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    template<typename InputIt>
    inline void process_events(const timestamp ts, InputIt it_begin, InputIt it_end);

private:
    // Member functions to define in the implementation. Do nothing by default

    /// @brief Function to process directly the events
    template<typename InputIt>
    inline void process_online(InputIt it_begin, InputIt it_end) {}

    /// @brief Function called when the input processing mode condition is met (see @ref Processing).
    /// @param processing_ts The event's timestamp for which the processing condition was met.
    /// @param n_processed_events The number of events processed (since the last call to this method) when the
    /// processing condition was met.
    /// @warning This method can be called while no events have been processed since the last call. This happens when
    /// @ref process_events is called with a buffer of events so recent that it triggers past asynchronous conditions.
    /// It is left to the inherited class to decide whether to process the asynchronous operation or not.
    void process_async(const timestamp processing_ts, const size_t n_processed_events) {}

    /// @brief Called when the internal states are initialized
    /// @param processing_ts The timestamp of the last async call if the algorithm was
    /// already initialized.
    void on_init(const timestamp processing_ts) {}

    /// @brief Function to get the end iterator for the next processing
    /// @param it_begin Iterator to the next event to process
    /// @param it_end Iterator to the end of the buffer
    /// @param to_it Iterator to the end of the next buffer to process
    /// @return If true, the state must be processed in after the next process_online
    template<typename InputIt>
    inline bool find_last_it(const InputIt it_begin, const InputIt it_end, InputIt &to_it);

    /// @brief Function that initializes the algorithm synchronization
    /// @param ts The timestamp to use to initialize the internal states
    void initialize(timestamp ts);

    /// @brief Cast as child
    Impl &child_cast() {
        return *static_cast<Impl *>(this);
    }

    timestamp processing_ts_ = timestamp(-1); ///< Timestamp of the last call to process async

    Processing processing_        = Processing::SYNC;
    bool is_initialized_          = false;         ///< Check if the algorithm is initialized
    timestamp delta_ts_           = 0;             ///< Delta between 2 processings when using N_US or MIXED
    timestamp next_processing_ts_ = timestamp(-1); ///< Next processing timestamp
    timestamp last_event_ts_      = 0;             ///< Last processed event's timestamp

    int n_processed_events_ = 0; ///< Number of events processed since the last call to process async
    int next_processing_n_events_ =
        0;                   ///< Number of events needed before the next async processing in N_EVENTS or MIXED
    int delta_n_events_ = 0; ///< Number of events between 2 processings in N_EVENTS or MIXED
};
} // namespace Metavision

#include "metavision/sdk/core/algorithms/detail/async_algorithm_impl.h"

#endif // METAVISION_SDK_CORE_ASYNC_ALGORITHM_H
