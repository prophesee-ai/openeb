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

#ifndef METAVISION_SDK_CORE_THREADED_PROCESS_H
#define METAVISION_SDK_CORE_THREADED_PROCESS_H

#include <memory>
#include <condition_variable>
#include <thread>
#include <queue>
#include <atomic>
#include <functional>

#include "metavision/sdk/base/utils/object_pool.h"

namespace Metavision {

/// @brief A convenient object whose purpose is to queue and dequeue tasks in a thread
class ThreadedProcess {
public:
    using Task          = std::function<void()>;
    using RepeatingTask = std::function<bool()>;

    /// @brief Destructor
    ///
    /// Waits for all pending tasks completion before leaving the thread.
    /// Use @ref abort before destruction to abort the processing.
    ~ThreadedProcess();

    /// @brief Adds a task to the processing queue.
    void add_task(Task task);

    /// @brief Adds a task that is repeated once it is done if and only if its result returns true.
    void add_repeating_task(RepeatingTask task);

    /// @brief Starts the processing thread
    /// @return false if the processing thread is already started
    bool start();

    /// @brief Requests the processing thread to stop and join
    ///
    /// The processing thread remains active until all pending tasks have been processed
    void stop();

    /// @brief Requests the processing thread to abort and join
    ///
    /// The threads leaves when it is no longer processing a task. All remaining tasks are dropped.
    void abort();

    /// @brief Returns if the processing thread is ongoing
    bool is_active();

private:
    void stop(bool abort);
    void processing_thread();

private:
    std::queue<Task> tasks_;
    std::thread processing_thread_;
    std::mutex process_mutex_;
    std::condition_variable process_cond_;
    std::atomic<bool> stop_{true}, abort_{true};
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_THREADED_PROCESS_H
