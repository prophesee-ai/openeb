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

#ifndef METAVISION_SDK_CORE_BASE_STAGE_H
#define METAVISION_SDK_CORE_BASE_STAGE_H

#include <atomic>
#include <mutex>
#include <vector>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <boost/any.hpp>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/utils/object_pool.h"

namespace Metavision {

class Pipeline;

/// @brief Base class for all stages added in the pipeline
///
/// A stage can produce and/or consume data.
/// When data are produced by a stage, they are forwarded to the next stages,
/// so that they can consume these data and also produce some data on their own.
///
/// The consumption of data is handled by the consuming callback which, by default,
/// does nothing with the data.
/// You can customize this behavior by calling @ref set_consuming_callback function.
///
/// The production of data is triggered by calling @ref produce, which will
/// call the consuming callback of the next stages. The notion of previous/next stage is
/// defined by either passing the previous stage in the constructor or by calling
/// @ref set_previous_stage.
/// It can also be expressed by customizing the consuming callback of a previous stage by
/// calling @ref set_consuming_callback.
///
/// Note that when data are produced, each producing callback is not executed synchronously.
/// Instead, the callback execution is scheduled to run either on the main thread (by default)
/// or on its own (dedicated) processing thread.
/// In any case, the callbacks of a stage are never called concurrently : the callbacks will be
/// either run synchronously in the main thread, or synchronously in a dedicated processing thread
/// @ref detach. This holds true for all consuming callbacks of a stage.
class BaseStage {
public:
    /// @brief Convenience alias for a typical buffer of events
    using EventBuffer = std::vector<EventCD>;
    /// @brief Convenience alias for a pool of buffer of events
    using EventBufferPool = SharedObjectPool<EventBuffer>;
    /// @brief Convenience alias for a pointer to a buffer of events allocated in a pool
    using EventBufferPtr = EventBufferPool::ptr_type;

protected:
    /// @brief Constructor
    /// @param detachable If this stage can be detached (i.e. can run on its own thread)
    inline BaseStage(bool detachable = true);

    /// @brief Constructor
    ///
    /// The @p prev_stage is used to setup the consuming callback
    /// that will be called when the previous stage produces data.
    /// When the previous stage produces data, it will call the consuming
    /// callback (@ref set_consuming_callback) of this stage.
    /// This behavior is automatically handled by this constructor.
    /// If you need to customize the consuming callback of one (or all) of the previous stages,
    /// you should use @ref set_consuming_callback instead.
    ///
    /// @param prev_stage the stage that is executed before the created one
    /// @param detachable If this stage can be detached (i.e. can run on its own thread)
    inline BaseStage(BaseStage &prev_stage, bool detachable = true);

public:
    /// @brief Destructor
    inline virtual ~BaseStage();

    /// @brief Sets the previous stage of this stage
    ///
    /// When called, this will setup each stage of the @p prev_stage to
    /// call the default consuming callback of this stage when data is produced.
    /// If you need to customize the consuming callback that should be called for
    /// a previous stage, you should use @ref set_consuming_callback instead.
    ///
    /// @param prev_stage the previous stage of this stage
    inline void set_previous_stage(BaseStage &prev_stage);

    /// @brief Gets the previous stages of this stage
    /// @return The previous stages
    /// @warning The reference to the set of previous stages can be invalidated by subsequent
    ///          calls to @ref set_previous_stage or @ref set_consuming_callback.
    ///          The returned value won't change once the associated pipeline is started.
    inline const std::unordered_set<BaseStage *> &previous_stages() const;

    /// @brief Gets the next stages of this stage
    /// @return The next stages
    /// @warning The reference to the set of next stages can be invalidated by subsequent
    ///          calls to @ref set_previous_stage or @ref set_consuming_callback.
    ///          The returned value won't change once the associated pipeline is started.
    inline const std::unordered_set<BaseStage *> &next_stages() const;

    /// @brief Enum class representing the status of a stage
    enum class Status {
        /// if the stage has not yet been started
        Inactive,
        /// if the stage is started
        Started,
        /// if the stage has completed its work
        Completed,
        /// if the stage has been cancelled
        Cancelled,
    };

    /// @brief Returns the status of the stage
    /// @return Status the status of the stage
    inline Status status() const;

    /// @brief Enum class representing the type of a notification sent from a stage
    enum class NotificationType {
        /// change of status
        Status,
    };

    /// @brief Sets the (generic) receiving callback for any previous stages
    ///
    /// Whenever a stage wants to notify other stages of changes (e.g. status update, etc),
    /// it can do so by calling @ref notify.
    /// When a notification is emitted at a stage, the receiving callback for each of the
    /// following stages is scheduled to be run with the corresponding type and data.
    /// This function sets the callback that will be scheduled when a notification
    /// is emitted by any previous @p stage.
    ///
    /// @param cb The callback that will be called when a notification is emitted from
    /// one of the previous stages
    inline void set_receiving_callback(
        const std::function<void(BaseStage &, const NotificationType &, const boost::any &)> &cb);

    /// @brief Sets the (generic) receiving callback for any previous stages
    ///
    /// Convenience overload to be used when the receiving callback does not need to receive the emitting stage
    /// as an argument.
    ///
    /// @param cb The callback that will be called when a notification is emitted from
    /// one of the previous stages
    ///
    /// @sa @ref set_receiving_callback(const std::function<void(BaseStage &, const NotificationType &, const
    /// boost::any &)> &cb)
    inline void set_receiving_callback(const std::function<void(const NotificationType &, const boost::any &)> &cb);

    /// @brief Sets the (specific) receiving callback for a previous stage
    ///
    /// When a notification is emitted at a stage, the receiving callback for each of the
    /// following stages is scheduled to be run with the corresponding type and data.
    /// This function sets the callback that will be scheduled when a notification is emitted
    /// by a specific @p prev_stage.
    ///
    /// According to the stage that produced the data, the generic consuming callback
    /// will be called if no specific consuming callback has been set.
    /// According to the stage that emitted the notification, the generic receiving callback
    /// will be called if no specific receiving callback has been set.
    ///
    /// @param prev_stage the previous stage for which the @p cb is set
    /// @param cb The callback that will be called when a notification is sent from
    /// one of the previous stages
    ///
    /// @sa @ref set_receiving_callback(const std::function<void(BaseStage &, const NotificationType &, const
    /// boost::any &)> &cb)
    inline void set_receiving_callback(BaseStage &prev_stage,
                                       const std::function<void(const NotificationType &, const boost::any &)> &cb);

    /// @brief Sets the starting callback
    ///
    /// The starting callback is called the first time @ref Pipeline::run or @ref Pipeline::step is called.
    /// This callback should set up a stage so that it can produce and/or consume data :
    /// for example, it can start a producing thread or configure an algorithm with
    /// parameters that were not known during construction.
    ///
    /// @warning This callback must not block, it can however create a thread to schedule some work
    /// to be done during the time the pipeline is run.
    ///
    /// @param cb The callback that will be called when this stage is started
    /// by the pipeline via @ref Pipeline::run or @ref Pipeline::step
    ///
    inline void set_starting_callback(const std::function<void()> &cb);

    /// @brief Sets the stopping callback
    ///
    /// @param cb The callback that will be called when this stage is being stopped
    /// either because the previous stages have completed or when the pipeline is cancelled
    /// via @ref Pipeline::cancel
    ///
    inline void set_stopping_callback(const std::function<void()> &cb);

    /// @brief Sets the setup callback
    ///
    /// The setup callback is called just after a valid reference to the pipeline has been set to the stage. The setup
    /// callback allows the stage to setup everything needing a valid reference to the pipeline (e.g. setting pre and
    /// post step callbacks).
    ///
    /// @param cb The callback that will be called when this stage has been set a valid reference to the pipeline
    inline void set_setup_callback(const std::function<void()> &cb);

    /// @brief Sets the (generic) consuming callback for any previous stage
    ///
    /// When data is produced at a stage, the consuming callback for each of the
    /// following steps is scheduled to be run with the produced data.
    /// This function sets the callback that will be scheduled when data
    /// is produced by any previous @p stage.
    ///
    /// The role of a consuming callback is, often, to produce data for next stages to consume.
    /// If you don't set a consuming callback, the data produced by a previous stage will not be used
    /// and, in particular, no data will be produced for following stages to consume.
    /// Therefore, for a stage to be useful, one of the @ref set_consuming_callback function must be called.
    ///
    /// @param cb The callback that will be called by the producing callback of a
    ///           previous stage
    ///
    inline void set_consuming_callback(const std::function<void(BaseStage &, const boost::any &)> &cb);

    /// @brief Sets the (generic) consuming callback for any previous stage
    ///
    /// Convenience overload to be used when the receiving callback does not need to receive the emitting stage
    /// as an argument.
    ///
    /// @param cb The callback that will be called by the producing callback of a
    ///           previous stage
    ///
    /// @sa @ref set_consuming_callback(const std::function<void(const boost::any&)> &cb)
    ///
    inline void set_consuming_callback(const std::function<void(const boost::any &)> &cb);

    /// @brief Sets the (specific) consuming callback for a previous stage
    ///
    /// When data is produced at a stage, the consuming callback for each of the
    /// following steps is scheduled to be run with the produced data.
    /// This function sets the callback that will be scheduled when data
    /// is produced by a specific @p prev_stage.
    ///
    /// According to the stage that produced the data, the generic consuming callback
    /// will be called if no specific consuming callback has been set.
    ///
    /// @param prev_stage the previous stage for which the @p cb is set
    /// @param cb The callback to be called when data is produced by @ref produce
    ///
    /// @sa @ref set_consuming_callback(const std::function<void(BaseStage&, const boost::any&)> &cb)
    ///
    inline void set_consuming_callback(BaseStage &prev_stage, const std::function<void(boost::any)> &cb);

    /// @brief Detaches this thread and schedules the execution of any callback on
    /// its own dedicated processing thread of the pipeline
    ///
    /// When @ref detach is called, the stage will now schedules the execution of
    /// any callback (the default or custom consuming callback set
    /// for this stage on any previous stages) on its own dedicated processing thread.
    /// A stage can be detached or undetached as long as the pipeline has not been started.
    ///
    /// @return true if stage has been detached (if it schedules the execution of callbacks on its own processing
    /// thread) and false otherwise
    inline bool detach();

    /// @brief Gets the detached status of this stage
    /// @return true if stage is detached (if it schedules the execution of callbacks on its own processing thread)
    /// and false if stage schedules the execution of its callback on the main thread
    inline bool is_detached() const;

    /// @brief Returns the associated pipeline
    /// Throws std::runtime_error if no pipeline has been set yet.
    /// @return @ref Pipeline "Pipeline&" the pipeline that owns this stage
    /// @warning The function throws if no pipeline has been associated (i.e. if the stage was not created by the
    ///          pipeline nor added to it).
    inline Pipeline &pipeline();

    /// @brief Returns the associated pipeline
    /// Throws std::runtime_error if no pipeline has been set yet.
    /// @return @ref Pipeline "Pipeline&" the pipeline that owns this stage
    /// @warning The function throws if no pipeline has been associated (i.e. if the stage was not created by the
    ///          pipeline nor added to it).
    inline const Pipeline &pipeline() const;

protected:
    /// @brief Notifies next stages of a change
    ///
    /// This schedules the execution of all the receiving callbacks
    ///
    /// @param type The notification type
    /// @param data The associated notification data
    inline void notify(const NotificationType &type, const boost::any &data);

    /// @brief Produces data
    ///
    /// This schedules the execution of all the producing callbacks
    ///
    /// @param data The produced data
    inline void produce(const boost::any &data);

    /// @brief Sets the stage status and notify next stages when it is done
    ///
    /// This function should be called whenever the stage will never produce any more data
    /// A stage is done when it has the status @ref Status::Completed or
    ///
    /// @ref Status::Cancelled and it has finished scheduling tasks to be
    /// processed by following stages.
    inline void complete();

private:
    std::atomic<Status> status_{Status::Inactive};
    std::atomic<bool> done_{false}, detachable_{true}, run_on_main_thread_{true};
    std::atomic<size_t> current_prod_id_{0};

    mutable std::mutex mutex_;
    Pipeline *pipeline_ = nullptr;
    std::unordered_set<BaseStage *> prev_stages_, next_stages_;

    std::mutex cbs_mutex_;
    std::function<void()> starting_cb_ = [] {};
    std::function<void()> stopping_cb_ = [] {};
    std::function<void()> setup_cb_    = [] {};
    std::unordered_map<BaseStage *, std::function<void(BaseStage &, const boost::any &)>> consuming_cbs_;
    std::unordered_map<BaseStage *, std::function<void(BaseStage &, const NotificationType, const boost::any &)>>
        receiving_cbs_;

    inline void start();
    inline void stop();
    inline void cancel();
    inline bool is_done() const;

    inline void set_pipeline(Pipeline &pipeline);
    inline bool set_status(const Status &status);

    inline void signal();
    inline void consume(BaseStage &prev_stage, const boost::any &data);
    inline void receive(BaseStage &prev_stage, const NotificationType &type, const boost::any &data);

    friend class Pipeline;
};

} // namespace Metavision

#include "detail/base_stage_impl.h"

#endif // METAVISION_SDK_CORE_BASE_STAGE_H
