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

#ifndef METAVISION_SDK_CORE_EVENT_BUFFER_RESLICER_ALGORITHM_H
#define METAVISION_SDK_CORE_EVENT_BUFFER_RESLICER_ALGORITHM_H

#include <atomic>
#include <functional>
#include <limits>
#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

namespace Detail {

/// @brief Slicing condition type
///
/// IDENTITY: output buffers are sliced exactly as input buffers.
/// N_EVENTS: output buffers are sliced to contain a fixed number of events.
/// N_US: output buffers are sliced to contain a fixed duration.
/// MIXED: output buffers are sliced to contain a fixed duration with maximum number of events.
enum class ReslicingConditionType { IDENTITY, N_EVENTS, N_US, MIXED };

/// @brief Structure defining the slicing condition
struct ReslicingCondition {
    ReslicingConditionType type = ReslicingConditionType::IDENTITY;
    timestamp delta_ts          = -1;
    std::size_t delta_n_events  = 0;

    inline ReslicingCondition() {}

    inline bool is_tracking_events_count() const {
        return type == ReslicingConditionType::N_EVENTS || type == ReslicingConditionType::MIXED;
    }

    inline bool is_tracking_duration() const {
        return type == ReslicingConditionType::N_US || type == ReslicingConditionType::MIXED;
    }

    static ReslicingCondition make_identity();
    static ReslicingCondition make_n_events(std::size_t delta_n_events);
    static ReslicingCondition make_n_us(timestamp delta_ts);
    static ReslicingCondition make_mixed(timestamp delta_ts, std::size_t delta_n_events);
};

/// @brief Condition status
enum class ReslicingConditionStatus { NOT_MET, MET_AUTOMATIC, MET_N_EVENTS, MET_N_US };

} // namespace Detail

/// @brief Base class for EventBufferReslicerAlgorithmT, providing support for interruptibility when the template
/// parameter @p enable_interruptions is set to true.
template<bool enable_interruptions>
class BaseEventBufferReslicerAlgorithmT {
protected:
    inline bool is_interrupted() const {
        return false;
    }
    inline void reset_interruption() {}
};

template<>
class BaseEventBufferReslicerAlgorithmT<true> {
public:
    /// @brief Asynchronously requests the processing loop of EventBufferReslicerAlgorithmT to be interrupted.
    void interrupt() {
        interrupted_ = true;
    }

protected:
    inline bool is_interrupted() const {
        return interrupted_;
    }

    inline void reset_interruption() {
        interrupted_ = false;
    }

private:
    std::atomic<bool> interrupted_{false};
};

/// @brief class reslicing input event buffers in a user-specified way.
///
/// This class plays the same role than AsyncAlgorithm, but is designed to be used via aggregation rather than
/// inheritance. This class is not protected against concurrent accesses and assumes input event buffers are sorted in
/// chronological order.
///
/// The role of this class is to reslice input event buffers according to a specified condition. Input event buffers are
/// provided to this class using the process_events function. Output event buffers are defined by successive calls
/// to two callbacks: 1/ the event callback on_events_cb, provided to the process_event function, and 2/ the
/// slicing callback on_new_slice_cb, configured independently in this class.
///
/// As an example, the reslicing operation can occur as follow:
///    * reslicer.process_events(input1_beg, input1_end, on_events_cb)
///       * on_events_cb(input1_beg1, input1_end1)
///       * on_new_slice_cb(output1_ts, output1_nevents)
///       * on_events_cb(input1_beg2, input1_end2)
///    * reslicer.process_events(input2_beg, input2_end, on_events_cb)
///       * on_events_cb(input2_beg, input2_end)
///    * reslicer.process_events(input3_beg, input3_end, on_events_cb)
///       * on_events_cb(input3_beg1, input3_end1)
///       * on_new_slice_cb(output2_ts, output2_nevents)
///       * on_events_cb(input3_beg2, input3_end2)
///       * on_new_slice_cb(output3_ts, output3_nevents)
///       * on_events_cb(input3_beg3, input3_end3)
///    * etc
template<bool enable_interruptions>
class EventBufferReslicerAlgorithmT : public BaseEventBufferReslicerAlgorithmT<enable_interruptions> {
public:
    using ConditionType   = Detail::ReslicingConditionType;
    using Condition       = Detail::ReslicingCondition;
    using ConditionStatus = Detail::ReslicingConditionStatus;

    /// @brief Type of the callback called when the specified slicing condition is met. This callback marks the end of a
    /// new output slice and, when called, is provided with the condition met status, the slicing timestamp and the
    /// number of events in the slice. The events included in the slice are provided separately, through the callback
    /// passed to the process_events function.
    /// @warning The slicing timestamp is here defined as the upper bound of the temporal range of the current slice. In
    /// case of a slicing status of MET_N_US, this means the first timestamp of the temporal range of the next slice. In
    /// all other cases, this means the timestamp of the last event in the slice.
    using OnNewSliceCb = std::function<void(ConditionStatus, timestamp, std::size_t)>;

    /// @brief Constructor
    /// @param on_new_slice_cb callback to be called to mark the end of a new output slice.
    /// @param condition definition of the slicing condition monitored by the slicer, set to identity by default.
    EventBufferReslicerAlgorithmT(OnNewSliceCb on_new_slice_cb = nullptr, const Condition &condition = Condition());
    ~EventBufferReslicerAlgorithmT() = default;

    /// @brief Updates the callback to be called to mark the end of a new output slice.
    /// @param on_new_slice_cb new callback.
    void set_on_new_slice_callback(OnNewSliceCb on_new_slice_cb);

    /// @brief Updates the slicing condition.
    /// @param condition definition of the slicing condition monitored by the slicer.
    /// @note In case the new condition is already satisfied by the events processed so far, which can happen if the new
    /// slicing condition is more strict than the previous one, this call will trigger a call to the @ref flush
    /// function. If the new condition is less strict, then the transition is perfect and no call to @ref flush occurs.
    void set_slicing_condition(const Condition &condition);

    /// @brief Getter to retrieve the policy used to slice event buffers.
    inline const Condition &get_slicing_condition() const {
        return condition_;
    }

    /// @brief Resets the internal state of the slicing algorithm.
    ///
    /// This may for instance be called when one wants to process events older than those already processed (e.g. in the
    /// case one wants to switch the source producing the events).
    /// @note This method doesn't change the algorithm's slicing condition nor flushes the ongoing time slice. It is the
    /// user's responsibility to call flush before this method to retrieve the incomplete time slice if needed.
    void reset();

    /// @brief Forces the generation of a new output slice.
    ///
    /// The resulting output slice corresponds to all the events processed since the last output slice, and the internal
    /// state is updated so that the next time slice will start with the next event.
    void flush();

    /// @brief Processes a buffer of events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of EventCD
    /// or equivalent. Underlying event instances must have a `t` field representing the timestamp.
    /// @tparam OnEventsCbT Type of the callback used to forward re-sliced event buffers for down-stream processing,
    /// which should match the following signature: void(InputIt, InputIt). The events forwarded via this callback are
    /// part of the current output slice, whose end will be notified by a call to the slicing callback.
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param on_events_cb event callback to be called when a new output buffer has been sliced, providing the
    /// begin and end iterators, the event timestamp and number of processed events when the slicing condition was met.
    ///
    /// @note If @p enable_interruptions is true, this function can be interrupted asynchronously, with no guarantees
    /// on when the interruption will happen or what events will have been processed. In case of interruption, the user
    /// is left in charge of making a synchronous call to @ref reset to recover a valid state.
    template<typename InputIt, typename OnEventsCbT>
    void process_events(InputIt it_begin, InputIt it_end, OnEventsCbT on_events_cb);

    /// @brief Notify the reslicing algorithm that time has elapsed without new events, which may trigger several calls
    /// to the slicing callback depending on the configured slicing condition.
    /// @param ts current timestamp
    ///
    /// @note If @p enable_interruptions is true, this function can be interrupted asynchronously, with no guarantees
    /// on when the interruption will happen or what events will have been processed. In case of interruption, the user
    /// is left in charge of making a synchronous call to @ref reset to recover a valid state.
    void notify_elapsed_time(timestamp ts);

private:
    void initialize_processing(timestamp ts);

    template<typename InputIt>
    ConditionStatus find_next_slicing_condition_met(InputIt it_begin, InputIt it_end, InputIt &it_buffer_end);
    template<typename InputIt>
    ConditionStatus find_next_slicing_condition_met_n_events(InputIt it_begin, InputIt it_end, InputIt &it_buffer_end);
    template<typename InputIt>
    ConditionStatus find_next_slicing_condition_met_n_us(InputIt it_begin, InputIt it_end, InputIt &it_buffer_end);

    void close_and_restart_new_slice(ConditionStatus status);

    OnNewSliceCb on_new_slice_cb_;
    Condition condition_;

    bool has_processing_started_ = false;
    std::size_t n_events_in_current_slice_;
    timestamp curr_slice_ref_ts_, curr_slice_last_observed_ts_;
};

using EventBufferReslicerAlgorithm              = EventBufferReslicerAlgorithmT<false>;
using InterruptibleEventBufferReslicerAlgorithm = EventBufferReslicerAlgorithmT<true>;

} // namespace Metavision

#include "metavision/sdk/core/algorithms/detail/event_buffer_reslicer_algorithm_impl.h"

#endif // METAVISION_SDK_CORE_EVENT_BUFFER_RESLICER_ALGORITHM_H
