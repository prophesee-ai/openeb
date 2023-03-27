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

#ifndef METAVISION_SDK_CORE_DETAIL_PIPELINE_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_PIPELINE_IMPL_H

#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "metavision/sdk/core/pipeline/pipeline.h"
#include "metavision/sdk/core/pipeline/base_stage.h"
#include "metavision/sdk/core/pipeline/algorithm_stage.h"

namespace Metavision {

struct Task {
    Task(const std::function<void()> &task = std::function<void()>(), size_t id = 0, bool optional = true,
         BaseStage *stage_ptr = nullptr) :
        task(task), id(id), optional(optional), stage_ptr(stage_ptr) {}

    bool empty() const {
        return !static_cast<bool>(task);
    }

    void operator()() const {
        task();
    }

    bool operator<(const Task &t) const {
        return id > t.id;
    }

    std::function<void()> task;
    size_t id;
    bool optional;
    BaseStage *stage_ptr;
};

class TaskQueue {
public:
    TaskQueue() : cancel_{false} {}

    void push(const Task &task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push(task);
        }
        cond_.notify_all();
    }

    Task pop() {
        Task t;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.wait(lock, [this]() { return !tasks_.empty() || cancel_; });
            if (!tasks_.empty()) {
                t = tasks_.top();
                tasks_.pop();
            }
        }
        return t;
    }

    void cancel() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cancel_ = true;
        }
        cond_.notify_one();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!tasks_.empty())
            tasks_.pop();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.size();
    }

private:
    std::atomic<bool> cancel_{false};
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    std::priority_queue<Task> tasks_;
};

class Pipeline::TaskScheduler {
public:
    TaskScheduler() : running_(false), exited_(false), main_tasks_(std::make_unique<TaskQueue>()) {}

    ~TaskScheduler() {}

    void complete_stage_if_done(BaseStage &stage, bool decrement_num_task = false) {
        bool stage_done = false;
        {
            std::lock_guard<std::mutex> lock(stage_tasks_mutex_);
            if (decrement_num_task)
                --stages_num_tasks_[&stage];
            if (stages_num_tasks_[&stage] == 0 && !stage.previous_stages().empty()) {
                stage_done = are_previous_stages_done(stage);
            }
        }
        if (stage_done) {
            stage.complete();
        }
    }

    void schedule(BaseStage &stage, const Task &task, bool schedule_on_main_thread = true) {
        if (!running_)
            return;
        if (schedule_on_main_thread) {
            if (std::this_thread::get_id() != main_thread_id_) {
                {
                    std::lock_guard<std::mutex> lock(stage_tasks_mutex_);
                    ++stages_num_tasks_[&stage];
                }
                main_tasks_->push(task);
            } else {
                task();
                complete_stage_if_done(*task.stage_ptr);
            }
        } else {
            {
                std::lock_guard<std::mutex> lock(stage_tasks_mutex_);
                ++stages_num_tasks_[&stage];
            }
            size_t id = 0;
            {
                std::lock_guard<std::mutex> lock(processing_map_id_mutex_);
                auto it = processing_map_id_.find(&stage);
                if (processing_map_id_.find(&stage) == processing_map_id_.end()) {
                    id = processing_map_id_[&stage] = processing_current_map_id_++;
                } else {
                    id = it->second;
                }
            }
            processing_tasks_[id]->push(task);
        }
    }

    // no need for concurrent access checks or double start logic protection : this function
    // can only be called from Pipeline::start() which already does the controls
    void init(bool main_thread_will_have_tasks, size_t num_processing_threads) {
        running_ = true;

        main_thread_will_have_tasks_ = main_thread_will_have_tasks;
        processing_tasks_.resize(num_processing_threads);
        for (auto &q : processing_tasks_)
            q = std::make_unique<TaskQueue>();
        processing_threads_.resize(num_processing_threads);
    }

    void start() {
        size_t num_processing_threads = processing_threads_.size();
        for (size_t i = 0; i < num_processing_threads; ++i) {
            processing_threads_[i] = std::thread([this, i]() {
                while (running_) {
                    // This will get an already queued task or wait for one to be scheduled
                    // This will also return an empty task if a call to cancel() or exit() is
                    // made on the pipeline by the user or one of the other threads that
                    // exited the processing loop (when running_ = false)
                    Task task = processing_tasks_[i]->pop();
                    if (!task.empty()) {
                        if (!exited_ || !task.optional) {
                            task();
                        }
                    }

                    if (task.stage_ptr) {
                        complete_stage_if_done(*task.stage_ptr, true);
                    }
                }
                cancel();
            });
        }
    }

    // no need for concurrent access checks or double start logic protection : this function
    // can only be called from Pipeline::stop() which already does the controls
    void stop() {
        for (auto &thread : processing_threads_) {
            if (thread.joinable())
                thread.join();
        }
    }

    bool is_running() const {
        return running_;
    }

    // no need for concurrent access checks, this function is thread safe
    bool step() {
        if (!running_)
            return false;

        if (main_thread_will_have_tasks_) {
            if (main_tasks_->empty()) {
                // We have to be careful to not try to pop a task unless there really
                // is one to pop, otherwise we could block forever
                std::this_thread::yield();
            } else {
                // This will get an already queued task or wait for one to be scheduled
                // This will also return an empty task if a call to cancel() or exit() is
                // made on the pipeline by the user or one of the other threads that
                // exited the processing loop (when running_ = false)
                Task task = main_tasks_->pop();
                if (!task.empty()) {
                    if (!exited_ || !task.optional) {
                        task();
                    }
                }

                if (task.stage_ptr) {
                    complete_stage_if_done(*task.stage_ptr, true);
                }
            }
        } else if (processing_tasks_.empty()) {
            // We have no tasks to process at all
            cancel();
            return false;
        } else {
            std::this_thread::yield();
        }
        return true;
    }

    // no need for concurrent access checks, this function is thread safe
    void cancel() {
        running_ = false;
        main_tasks_->cancel();
        for (auto &q : processing_tasks_)
            q->cancel();
    }

    // no need for concurrent access checks, this function is thread safe
    void exit() {
        exited_ = true;
        // process mandatory tasks while queues are not empty
        while (running_ && !empty()) {
            step();
        }
        cancel();
    }

    bool empty() const {
        if (!main_tasks_->empty())
            return false;
        for (auto &q : processing_tasks_) {
            if (!q->empty())
                return false;
        }
        return true;
    }

    void set_main_thread_id(std::thread::id id) {
        main_thread_id_ = id;
    }

    std::thread::id main_thread_id() const {
        return main_thread_id_;
    }

private:
    bool are_previous_stages_done(const BaseStage &stage) {
        bool done                   = true;
        const auto &prev_stage_ptrs = stage.previous_stages();
        for (auto &prev_stage_ptr : prev_stage_ptrs) {
            // previous stage must have completed and finished scheduling all tasks
            if (prev_stage_ptr->status() != BaseStage::Status::Completed || !prev_stage_ptr->is_done()) {
                done = false;
                break;
            }
        }
        return done;
    }

    std::atomic<bool> running_;
    std::atomic<bool> exited_;
    std::unique_ptr<TaskQueue> main_tasks_;
    std::thread::id main_thread_id_;

    std::mutex processing_map_id_mutex_;
    size_t processing_current_map_id_ = 0;
    std::unordered_map<BaseStage *, size_t> processing_map_id_;

    std::vector<std::unique_ptr<TaskQueue>> processing_tasks_;
    std::vector<std::thread> processing_threads_;

    mutable std::mutex stage_tasks_mutex_;
    bool main_thread_will_have_tasks_;
    std::unordered_map<BaseStage *, size_t> stages_num_tasks_;
};

Pipeline::Pipeline(bool auto_detach) :
    auto_detach_stages_(auto_detach), scheduler_(std::make_unique<TaskScheduler>()) {}

Pipeline::~Pipeline() {
    cancel();
    step();
}

inline void Pipeline::check_if_started() {
    if (status_ == Status::Started)
        throw std::runtime_error("Pipeline : Can't modify pipeline after it has started!");
}

template<typename Stage, typename>
Stage &Pipeline::add_stage(std::unique_ptr<Stage> &&stage) {
    check_if_started();
    return static_cast<Stage &>(add_stage_priv(std::move(stage)));
}

template<typename Stage, typename>
Stage &Pipeline::add_stage(std::unique_ptr<Stage> &&stage, BaseStage &prev_stage) {
    check_if_started();
    stage->set_previous_stage(prev_stage);
    return static_cast<Stage &>(add_stage_priv(std::move(stage)));
}

template<typename OutputEventType, typename InputEventType, typename Algorithm,
         typename std::enable_if_t<!is_base_stage<Algorithm> && !is_same_type<InputEventType, OutputEventType>, int>>
AlgorithmStage<Algorithm, OutputEventType, InputEventType> &
    Pipeline::add_algorithm_stage(std::unique_ptr<Algorithm> &&algo) {
    check_if_started();
    auto stage = std::make_unique<AlgorithmStage<Algorithm, OutputEventType, InputEventType>>(std::move(algo));
    return static_cast<AlgorithmStage<Algorithm, OutputEventType, InputEventType> &>(add_stage_priv(std::move(stage)));
}

template<typename OutputEventType, typename InputEventType, typename Algorithm,
         typename std::enable_if_t<!is_base_stage<Algorithm> && is_same_type<InputEventType, OutputEventType>, int>>
AlgorithmStage<Algorithm, OutputEventType, InputEventType> &
    Pipeline::add_algorithm_stage(std::unique_ptr<Algorithm> &&algo, bool enabled) {
    check_if_started();
    auto stage = std::make_unique<AlgorithmStage<Algorithm, OutputEventType, InputEventType>>(std::move(algo), enabled);
    return static_cast<AlgorithmStage<Algorithm, OutputEventType, InputEventType> &>(add_stage_priv(std::move(stage)));
}

template<typename OutputEventType, typename InputEventType, typename Algorithm,
         typename std::enable_if_t<!is_base_stage<Algorithm> && !is_same_type<InputEventType, OutputEventType>, int>>
AlgorithmStage<Algorithm, OutputEventType, InputEventType> &
    Pipeline::add_algorithm_stage(std::unique_ptr<Algorithm> &&algo, BaseStage &prev_stage) {
    check_if_started();
    auto stage =
        std::make_unique<AlgorithmStage<Algorithm, OutputEventType, InputEventType>>(std::move(algo), prev_stage);
    return static_cast<AlgorithmStage<Algorithm, OutputEventType, InputEventType> &>(add_stage_priv(std::move(stage)));
}

template<typename OutputEventType, typename InputEventType, typename Algorithm,
         typename std::enable_if_t<!is_base_stage<Algorithm> && is_same_type<InputEventType, OutputEventType>, int>>
AlgorithmStage<Algorithm, OutputEventType, InputEventType> &
    Pipeline::add_algorithm_stage(std::unique_ptr<Algorithm> &&algo, BaseStage &prev_stage, bool enabled) {
    check_if_started();
    auto stage = std::make_unique<AlgorithmStage<Algorithm, OutputEventType, InputEventType>>(std::move(algo),
                                                                                              prev_stage, enabled);
    return static_cast<AlgorithmStage<Algorithm, OutputEventType, InputEventType> &>(add_stage_priv(std::move(stage)));
}

BaseStage &Pipeline::add_stage_priv(std::unique_ptr<BaseStage> &&stage) {
    stage->set_pipeline(*this);
    if (auto_detach_stages_) {
        stage->detach();
    }
    stages_.emplace_back(std::move(stage));
    return *stages_.back();
}

void Pipeline::remove_stage(BaseStage &stage) {
    check_if_started();
    auto it = std::find_if(stages_.begin(), stages_.end(),
                           [&stage](const std::unique_ptr<BaseStage> &p) { return p.get() == &stage; });
    if (it != stages_.end()) {
        stages_.erase(it);
    }

    for (auto &s : stages_) {
        s->prev_stages_.erase(&stage);
        s->next_stages_.erase(&stage);
    }
}

size_t Pipeline::count() const {
    return stages_.size();
}

bool Pipeline::empty() const {
    return stages_.empty();
}

Pipeline::Status Pipeline::status() const {
    return status_;
}

void Pipeline::start() {
    if (scheduler_->main_thread_id() == std::thread::id()) {
        scheduler_->set_main_thread_id(std::this_thread::get_id());
    }
    bool main_thread_will_have_tasks = false;
    int num_processing_threads       = 0;
    for (auto &stage : stages_) {
        // producing only stages don't count
        const auto &prev_stages = stage->previous_stages();
        if (!prev_stages.empty()) {
            if (stage->is_detached()) {
                ++num_processing_threads;
            } else {
                main_thread_will_have_tasks = true;
            }
        }
    }
    scheduler_->init(main_thread_will_have_tasks, num_processing_threads);
    for (auto &stage : stages_) {
        stage->start();
    }
    scheduler_->start();
}

void Pipeline::stop() {
    for (auto &stage_ptr : stages_) {
        stage_ptr->stop();
    }
    scheduler_->exit();
    scheduler_->stop();
    if (status_ != Status::Cancelled) {
        status_ = Status::Completed;
    }
}

bool Pipeline::step() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (status_ == Status::Inactive) {
        status_ = Status::Started;
        start();
    }

    bool done = true;
    for (auto &stage_ptr : stages_) {
        if (!stage_ptr->is_done()) {
            done = false;
            break;
        }
    }

    bool ret = true;
    if (status_ == Status::Cancelled || done) {
        stop();
        ret = false;
    } else {
        for (const auto &pre_cb : pre_step_cbs_)
            pre_cb();

        ret = scheduler_->step();

        for (const auto &post_cb : post_step_cbs_)
            post_cb();
    }

    return ret;
}

void Pipeline::run() {
    if (status_ == Status::Started)
        return;

    while (step()) {}
}

void Pipeline::cancel() {
    // can be called concurrently, no need for a mutex
    status_ = Status::Cancelled;
    for (auto &stage_ptr : stages_) {
        if (stage_ptr->status() != BaseStage::Status::Completed) {
            stage_ptr->cancel();
        }
    }
}

void Pipeline::add_pre_step_callback(const StepCallback &cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    pre_step_cbs_.emplace_back(cb);
}

void Pipeline::add_post_step_callback(const StepCallback &cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    post_step_cbs_.emplace_back(cb);
}

void Pipeline::schedule(BaseStage &stage, const std::function<void()> &task, size_t task_id, bool optional,
                        bool schedule_on_main_thread) {
    scheduler_->schedule(stage, {task, task_id, optional, &stage}, schedule_on_main_thread);
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_PIPELINE_IMPL_H
