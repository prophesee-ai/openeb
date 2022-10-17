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

#ifndef METAVISION_SDK_CORE_ALGORITHM_STAGE_H
#define METAVISION_SDK_CORE_ALGORITHM_STAGE_H

#include "metavision/sdk/core/pipeline/base_stage.h"

namespace Metavision {

/// @brief Stage that wraps an algorithm in the consuming callback
///
/// The easiest way to use this stage is to use @ref Pipeline::add_algorithm_stage, rather
/// than trying to instantiate it yourself.
///
/// Be mindful of the order of the template arguments, the output event type is given first, because most of the time
/// the InputEventType is EventCD, while the OutputEventType varies.
///
/// @tparam Algorithm the type of wrapped algorithm
/// @tparam OutputEventType optionally, the type of event produced by this stage in @ref produce
/// @tparam InputEventType optionally, the type of event consumed by this stage in the consuming callback
template<typename Algorithm, typename OutputEventType = EventCD, typename InputEventType = EventCD>
class AlgorithmStage : public BaseStage {
    using InputEventBuffer      = std::vector<InputEventType>;
    using InputEventBufferPool  = SharedObjectPool<InputEventBuffer>;
    using InputEventBufferPtr   = typename InputEventBufferPool::ptr_type;
    using OutputEventBuffer     = std::vector<OutputEventType>;
    using OutputEventBufferPool = SharedObjectPool<OutputEventBuffer>;
    using OutputEventBufferPtr  = typename OutputEventBufferPool::ptr_type;

public:
    /// @brief Constructor
    /// @param algo The wrapped algorithm for which process will be called
    AlgorithmStage(std::unique_ptr<Algorithm> &&algo) : algo_(std::move(algo)), enabled_(true) {
        init();
    }

    /// @brief Constructor
    ///
    /// Overload constructor available when the type of event consumed is the same as the type
    /// of events produced.
    ///
    /// @sa @ref enable, @ref disable, @ref set_enabled.
    template<typename TOutputEventType = OutputEventType, typename TInputEventType = InputEventType,
             typename = std::enable_if_t<std::is_same<TOutputEventType, TInputEventType>::value>>
    AlgorithmStage(std::unique_ptr<Algorithm> &&algo, bool enabled = true) : algo_(std::move(algo)), enabled_(enabled) {
        init();
    }

    /// @brief Constructor
    ///
    /// Overload constructor that simplifies setting the previous stage.
    ///
    /// @param algo The wrapped algorithm for which process will be called
    /// @param prev_stage The previous stage of this stage
    AlgorithmStage(std::unique_ptr<Algorithm> &&algo, BaseStage &prev_stage) :
        BaseStage(prev_stage),
        algo_(std::move(algo)),
        event_buffer_pool_(OutputEventBufferPool::make_bounded()),
        enabled_(true) {
        init();
    }

    /// @brief Constructor
    ///
    /// Overload constructor that simplifies setting the previous stage and is only available when the type of event
    /// consumed is the same as the type of events produced.
    ///
    /// @param algo The wrapped algorithm for which process will be called
    /// @param prev_stage The previous stage of this stage
    /// @param enabled true if the algorithm is enabled by default, false otherwise
    /// @sa @ref enable, @ref disable, @ref set_enabled.
    template<typename TOutputEventType = OutputEventType, typename TInputEventType = InputEventType,
             typename = std::enable_if_t<std::is_same<TOutputEventType, TInputEventType>::value>>
    AlgorithmStage(std::unique_ptr<Algorithm> &&algo, BaseStage &prev_stage, bool enabled = true) :
        BaseStage(prev_stage),
        algo_(std::move(algo)),
        event_buffer_pool_(OutputEventBufferPool::make_bounded()),
        enabled_(enabled) {
        init();
    }

    /// @brief Returns the wrapped algorithm
    /// @return @p Algorithm & the wrapped algorithm
    Algorithm &algo() {
        return *algo_;
    }

    /// @brief Returns the wrapped algorithm
    /// @return @p Algorithm & the wrapped algorithm
    const Algorithm &algo() const {
        return *algo_;
    }

    /// @brief Enables calling the process function of the algorithm in the consuming callback
    ///
    /// When the algorithm is enabled, the (default) consuming callback calls process on the
    /// consumed data and produces as output, the output of the algorithm.
    /// When the algorithm is disabled, this stage acts as a passthrough and directly produces the
    /// consumed data.
    /// Therefore, to avoid mistakes, this function is only available if the type of output event (produced)
    /// is the same as the type of input events (consumed).
    template<typename TOutputEventType = OutputEventType, typename TInputEventType = InputEventType,
             typename = std::enable_if_t<std::is_same<TOutputEventType, TInputEventType>::value>>
    void enable() {
        enabled_ = true;
    }

    /// @brief Disables calling the process function of the algorithm in the consuming callback
    ///
    /// When the algorithm is enabled, the (default) consuming callback calls process on the
    /// consumed data and produces as output, the output of the algorithm.
    /// When the algorithm is disabled, this stage acts as a passthrough and directly produces the
    /// consumed data.
    /// Therefore, to avoid mistakes, this function is only available if the type of output event (produced)
    /// is the same as the type of input events (consumed).
    template<typename TOutputEventType = OutputEventType, typename TInputEventType = InputEventType,
             typename = std::enable_if_t<std::is_same<TOutputEventType, TInputEventType>::value>>
    void disable() {
        enabled_ = false;
    }

    /// @brief Enable/disables calling the process function of the algorithm in the consuming callback
    ///
    /// This does the same as calling @ref enable or @ref disable according to the value of @p enabled.
    /// @param enabled True if the algorithm should be enabled, false otherwise
    template<typename TOutputEventType = OutputEventType, typename TInputEventType = InputEventType,
             typename = std::enable_if_t<std::is_same<TOutputEventType, TInputEventType>::value>>
    void set_enabled(bool enabled) {
        enabled_ = enabled;
    }

    /// @brief Returns the status of the wrapped algorithm
    /// @return true if the algorithm is enabled, false otherwise
    template<typename TOutputEventType = OutputEventType, typename TInputEventType = InputEventType,
             typename = std::enable_if_t<std::is_same<TOutputEventType, TInputEventType>::value>>
    bool is_enabled() const {
        return enabled_;
    }

private:
    void init() {
        set_consuming_callback([this](const boost::any &data) {
            try {
                auto buffer     = boost::any_cast<InputEventBufferPtr>(data);
                auto out_buffer = event_buffer_pool_.acquire();
                out_buffer->clear();
                if (enabled_)
                    process(buffer->cbegin(), buffer->cend(), std::back_inserter(*out_buffer), std::true_type());
                else
                    process(buffer->cbegin(), buffer->cend(), std::back_inserter(*out_buffer), std::false_type());
                produce(out_buffer);
            } catch (boost::bad_any_cast &) {}
        });
    }

    template<typename InputIt, typename OutputIt, typename T, typename TOutputEventType = OutputEventType,
             typename TInputEventType = InputEventType>
    std::enable_if_t<!std::is_same<TOutputEventType, TInputEventType>::value, void> process(InputIt begin, InputIt end,
                                                                                            OutputIt d_begin, T) {
        algo_->process_events(begin, end, d_begin);
    }

    template<typename InputIt, typename OutputIt, typename TOutputEventType = OutputEventType,
             typename TInputEventType = InputEventType>
    std::enable_if_t<std::is_same<TOutputEventType, TInputEventType>::value, void>
        process(InputIt begin, InputIt end, OutputIt d_begin, std::true_type) {
        algo_->process_events(begin, end, d_begin);
    }

    template<typename InputIt, typename OutputIt, typename TOutputEventType = OutputEventType,
             typename TInputEventType = InputEventType>
    std::enable_if_t<std::is_same<TOutputEventType, TInputEventType>::value, void>
        process(InputIt begin, InputIt end, OutputIt d_begin, std::false_type) {
        std::copy(begin, end, d_begin);
    }

    std::unique_ptr<Algorithm> algo_;
    OutputEventBufferPool event_buffer_pool_;
    std::atomic<bool> enabled_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_ALGORITHM_STAGE_H
