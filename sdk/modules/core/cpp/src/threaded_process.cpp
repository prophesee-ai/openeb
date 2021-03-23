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

#include "metavision/sdk/core/utils/threaded_process.h"

namespace Metavision {

ThreadedProcess::~ThreadedProcess() {
    stop();
}

void ThreadedProcess::add_task(Task task) {
    if (abort_) {
        return;
    }

    std::lock_guard<std::mutex> lock(process_mutex_);
    tasks_.push(task);
    process_cond_.notify_all();
}

void ThreadedProcess::add_repeating_task(RepeatingTask task) {
    add_task([this, task]() {
        if (task()) {
            add_repeating_task(task);
        }
    });
}

bool ThreadedProcess::start() {
    std::unique_lock<std::mutex> lock(process_mutex_);
    if (processing_thread_.joinable()) {
        // If already active, do not start the thread
        return false;
    }

    // Clears tasks list
    std::queue<Task> tasks;
    tasks.swap(tasks_);

    processing_thread_ = std::thread(&ThreadedProcess::processing_thread, this);
    process_cond_.wait(lock, [this]() { return !stop_ && !abort_; });
    return true;
}

void ThreadedProcess::stop() {
    stop(false);
}

void ThreadedProcess::abort() {
    stop(true);
}

void ThreadedProcess::stop(bool abort) {
    {
        std::lock_guard<std::mutex> lock(process_mutex_);
        if (!processing_thread_.joinable()) {
            return;
        }

        stop_  = true;
        abort_ = abort;
        process_cond_.notify_all();
    }

    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

bool ThreadedProcess::is_active() {
    std::lock_guard<std::mutex> lock(process_mutex_);
    return processing_thread_.joinable();
}

void ThreadedProcess::processing_thread() {
    {
        std::lock_guard<std::mutex> lock(process_mutex_);
        stop_  = false;
        abort_ = false;
        process_cond_.notify_all();
    }

    while (!abort_) {
        std::queue<Task> tasks;
        {
            std::unique_lock<std::mutex> lock(process_mutex_);
            // Waits for tasks
            process_cond_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
            if (tasks_.empty()) {
                // Stop call and no pending tasks
                break;
            }

            // Swap tasks
            tasks.swap(tasks_);
        }

        // Run tasks
        while (!tasks.empty() && !abort_) {
            tasks.front()();
            tasks.pop();
        }
    }
}

} // namespace Metavision
