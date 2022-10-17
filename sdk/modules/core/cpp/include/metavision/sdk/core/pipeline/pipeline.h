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

#ifndef METAVISION_SDK_CORE_PIPELINE_H
#define METAVISION_SDK_CORE_PIPELINE_H

#include <memory>
#include <mutex>
#include <atomic>
#include <queue>
#include <vector>
#include <functional>
#include <unordered_map>

#include "metavision/sdk/base/events/event_cd.h"

namespace Metavision {

class BaseStage;
template<typename, typename, typename>
class AlgorithmStage;

template<typename T>
constexpr bool is_base_stage = std::is_base_of<BaseStage, T>::value;

template<typename T1, typename T2>
constexpr bool is_same_type = std::is_same<T1, T2>::value;

/// @brief Class that represents a pipeline of processing units (stages) and controls their execution
class Pipeline {
    class TaskScheduler;

public:
    /// @brief Constructor
    /// @param auto_detach If true, each stage added to the pipeline will automatically be detached (@ref
    /// BaseStage::detach)
    inline Pipeline(bool auto_detach = false);

    /// @brief Destructor
    ///
    /// The destructor ensures that the pipeline is stopped by calling @ref cancel
    inline ~Pipeline();

    /// @brief Adds a stage to the pipeline
    /// @note The ownership of the stage is transferred to the pipeline.
    /// If you need to further interact with this stage, use the returned reference to the stage.
    /// @param stage Stage to add
    /// @return @ref Stage "Stage&" The added stage
    template<typename Stage, typename = std::enable_if_t<std::is_base_of<BaseStage, Stage>::value>>
    Stage &add_stage(std::unique_ptr<Stage> &&stage);

    /// @brief Adds a stage to the pipeline
    ///
    /// Convenience overload
    ///
    /// This function does the same as the more verbose equivalent :
    ///   pipeline.add_stage(stage);
    ///   stage->set_previous_stage(prev_stage);
    ///
    /// @param stage Stage to add
    /// @param prev_stage Previous stage
    /// @return @ref Stage "Stage&" The added stage
    template<typename Stage, typename = std::enable_if_t<std::is_base_of<BaseStage, Stage>::value>>
    Stage &add_stage(std::unique_ptr<Stage> &&stage, BaseStage &prev_stage);

    /// @brief Adds a stage that wraps an algorithm as the consuming callback to the pipeline
    ///
    /// Convenience overload
    ///
    /// This function creates an instance of @ref AlgorithmStage and adds it to the pipeline.
    ///
    /// @warning Be mindful of the order of the template arguments: the output event type is given first, because most
    /// of the time the InputEventType is EventCD, while the OutputEventType varies.
    ///
    /// @tparam OutputEventType Type of events produced by this stage, defaults to @ref EventCD
    /// @tparam InputEventType Type of events consumed by this stage, defaults to @ref EventCD
    /// @tparam Algorithm Type of algorithm wrapped in this stage, its type is inferred from the arguments
    /// @param algo Wrapped algorithm
    /// @return @ref AlgorithmStage "AlgorithmStage&" the created stage
    template<
        typename OutputEventType = EventCD, typename InputEventType = EventCD, typename Algorithm,
        typename std::enable_if_t<!is_base_stage<Algorithm> && !is_same_type<InputEventType, OutputEventType>, int> = 0>
    AlgorithmStage<Algorithm, OutputEventType, InputEventType> &add_algorithm_stage(std::unique_ptr<Algorithm> &&algo);

    /// @brief Adds a stage that wraps an algorithm as the consuming callback to the pipeline
    ///
    /// Convenience overload only available if InputEventType == OutputEventType.
    ///
    /// This function creates an instance of @ref AlgorithmStage and adds it to the pipeline.
    ///
    /// @warning Be mindful of the order of the template arguments: the output event type is given first, because most
    /// of the time the InputEventType is EventCD, while the OutputEventType varies.
    ///
    /// @tparam OutputEventType Type of events produced by this stage, defaults to @ref EventCD
    /// @tparam InputEventType Type of events consumed by this stage, defaults to @ref EventCD
    /// @tparam Algorithm Type of algorithm wrapped in this stage, its type is inferred from the arguments
    /// @param algo Wrapped algorithm
    /// @param enabled If the stage is enabled by default
    /// @return @ref AlgorithmStage "AlgorithmStage&" the created stage
    template<
        typename OutputEventType = EventCD, typename InputEventType = EventCD, typename Algorithm,
        typename std::enable_if_t<!is_base_stage<Algorithm> && is_same_type<InputEventType, OutputEventType>, int> = 0>
    AlgorithmStage<Algorithm, OutputEventType, InputEventType> &add_algorithm_stage(std::unique_ptr<Algorithm> &&algo,
                                                                                    bool enabled = true);

    /// @brief Adds a stage that wraps an algorithm as the consuming callback to the pipeline
    ///
    /// Convenience overload that also allows setting the previous stage.
    ///
    /// This function creates an instance of @ref AlgorithmStage and adds it to the pipeline.
    ///
    /// @warning Be mindful of the order of the template arguments: the output event type is given first, because most
    /// of the time the InputEventType is EventCD, while the OutputEventType varies.
    ///
    /// @tparam OutputEventType Type of events produced by this stage, defaults to @ref EventCD
    /// @tparam InputEventType Type of events consumed by this stage, defaults to @ref EventCD
    /// @tparam Algorithm Type of algorithm wrapped in this stage, its type is inferred from the arguments
    /// @param algo Wrapped algorithm
    /// @param prev_stage Previous stage of this stage
    /// @return @ref AlgorithmStage "AlgorithmStage&" the created stage
    template<
        typename OutputEventType = EventCD, typename InputEventType = EventCD, typename Algorithm,
        typename std::enable_if_t<!is_base_stage<Algorithm> && !is_same_type<InputEventType, OutputEventType>, int> = 0>
    AlgorithmStage<Algorithm, OutputEventType, InputEventType> &add_algorithm_stage(std::unique_ptr<Algorithm> &&algo,
                                                                                    BaseStage &prev_stage);

    /// @brief Adds a stage that wraps an algorithm as the consuming callback to the pipeline
    ///
    /// Convenience overload that also allows setting the previous stage and is only available if InputEventType ==
    /// OutputEventType.
    ///
    /// This function creates an instance of @ref AlgorithmStage and adds it to the pipeline.
    ///
    /// @warning Be mindful of the order of the template arguments: the output event type is given first, because most
    /// of the time the InputEventType is EventCD, while the OutputEventType varies.
    ///
    /// @tparam OutputEventType Type of events produced by this stage, defaults to @ref EventCD
    /// @tparam InputEventType Type of events consumed by this stage, defaults to @ref EventCD
    /// @tparam Algorithm Type of algorithm wrapped in this stage, its type is inferred from the arguments
    /// @param algo Wrapped algorithm
    /// @param prev_stage Previous stage of this stage
    /// @param enabled If the stage is enabled by default, ignored if @p InputEventType != @p OutputEventType
    /// @return @ref AlgorithmStage "AlgorithmStage&" the created stage
    template<
        typename OutputEventType = EventCD, typename InputEventType = EventCD, typename Algorithm,
        typename std::enable_if_t<!is_base_stage<Algorithm> && is_same_type<InputEventType, OutputEventType>, int> = 0>
    AlgorithmStage<Algorithm, OutputEventType, InputEventType> &
        add_algorithm_stage(std::unique_ptr<Algorithm> &&algo, BaseStage &prev_stage, bool enabled = true);

    /// @brief Removes a stage from the pipeline
    ///
    /// This has no effect if the stage was not already added to the pipeline.
    /// @param stage Stage to remove
    /// @note This function will also remove the stage from the list of previous and next stages
    /// of any other stage in the pipeline. In particular, this means that proper care must be taken by the caller
    /// to make sure the consuming callbacks of the stages affected by this removal are connected to another stage
    /// in the pipeline with e.g @ref BaseStage::set_previous_stage or @ref BaseStage::set_consuming_callback
    /// before the pipeline is started, as in the following example :
    /// @code{.cpp}
    /// Pipeline p;
    /// auto& s1 = p.add_stage(std::make_unique<Stage>());
    /// auto& s2 = p.add_stage(std::make_unique<Stage>(), s1);
    /// auto& s3 = p.add_stage(std::make_unique<Stage>(), s2);
    /// p.remove_stage(s2);
    /// s3.set_previous_stage(s1); // without this line, s3 will wait forever since its previous stage has been
    ///                            // removed and will not produce any data
    /// @endcode
    inline void remove_stage(BaseStage &stage);

    /// @brief Returns the number of stages inside the pipeline
    /// @return size_t the number of stages inside the pipeline
    inline size_t count() const;

    /// @brief Checks if the pipeline does not contain any stage
    /// @return true if the pipeline is empty, false otherwise
    inline bool empty() const;

    /// @brief Enum class representing the status of the pipeline
    enum class Status {
        /// if the stage has not yet been started
        Inactive,
        /// if the pipeline has been started
        Started,
        /// if the pipeline has been cancelled with @ref cancel
        Cancelled,
        /// if the pipeline has finished running
        Completed
    };

    /// @brief Gets the status of the pipeline
    /// @return @ref Status the status of the pipeline
    inline Status status() const;

    /// @brief Executes a step of the pipeline
    ///
    /// This actually runs one of the scheduled callback on the main thread.
    /// The processing threads runs on their own, but can be blocked by the main
    /// thread if one stage needs to run on the main thread.
    ///
    /// @return true if the step was successful, false if the pipeline has no remaining steps to run
    inline bool step();

    /// @brief Runs the pipeline
    ///
    /// This will block until the execution of all the stages callbacks have been
    /// executed.
    /// This can be halted by calling @ref cancel.
    ///
    /// This is functionally equivalent to the following loop on a Pipeline p :
    ///   while (p.step()) {}
    inline void run();

    /// @brief Cancels the pipeline execution
    ///
    /// This will prevent any stage from producing any more data and will
    /// cancel all scheduled callbacks, therefore immediately exiting the pipeline.
    inline void cancel();

    /// @brief A Callback called before of after the pipeline steps
    /// @warning a StepCallback is not allowed to add another step callback otherwise the pipeline will deadlock
    using StepCallback = std::function<void()>;

    /// @brief Adds a callback that will be called before the pipeline steps
    /// @param cb The callback to call
    /// @warning This method cannot be called from a step callback
    inline void add_pre_step_callback(const StepCallback &cb);

    /// @brief Adds a callback that will be called after the pipeline steps
    /// @param cb The callback to call
    /// @warning This method cannot be called from a step callback
    inline void add_post_step_callback(const StepCallback &cb);

private:
    inline BaseStage &add_stage_priv(std::unique_ptr<BaseStage> &&stage);
    inline void check_if_started();

    inline void start();
    inline void stop();
    inline void schedule(BaseStage &stage, const std::function<void()> &task, size_t task_id, bool optional,
                         bool schedule_on_main_thread = true);

    bool auto_detach_stages_ = false;
    std::mutex mutex_;
    std::atomic<Status> status_{Status::Inactive};
    std::vector<std::unique_ptr<BaseStage>> stages_;
    std::vector<StepCallback> pre_step_cbs_;
    std::vector<StepCallback> post_step_cbs_;
    std::unique_ptr<TaskScheduler> scheduler_;

    friend class BaseStage;
};

} // namespace Metavision

#include "detail/pipeline_impl.h"

#endif // METAVISION_SDK_CORE_PIPELINE_H
