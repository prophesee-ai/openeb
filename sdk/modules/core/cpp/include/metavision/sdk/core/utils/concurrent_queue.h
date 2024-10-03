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

#ifndef METAVISION_SDK_CORE_CONCURRENT_QUEUE_H
#define METAVISION_SDK_CORE_CONCURRENT_QUEUE_H

#include <mutex>
#include <condition_variable>
#include <queue>
#include <optional>

namespace Metavision {

/// @brief Class that implements a concurrent queue facility
///
/// This class can be used to connect tasks together. For example, a task that produces some data will push it into the
/// queue while another one will pop the data from it to consume/process it. If the queue is empty, the consuming task,
/// when trying to pop data, will be blocked until new data is pushed in the queue by the producing task.
///
/// In addition to that, this class maintains a state indicating whether the queue is opened or closed (i.e. enabled or
/// disabled). In the closed state, no data can be pushed (i.e. the call is ignored). This is useful to properly
/// handle the termination of tasks where some of them can be waiting for objects allocated from a memory pool to come
/// back before stopping.
///
/// For example, let's imagine a task T1 that is allocating data from a pool and pushing it to a queue that another task
/// T2 is pulling. For the task T1 to terminate all the allocated data need to be back in the pool, otherwise the task
/// will wait and this can lead to deadlocks (e.g. if the task T2 is also waiting for new data to be produced).
/// A simple way to fix this problem is to:
///  - stop the task T2,
///  - clear and disable the queue (all the data will go back to the memory pool of the task T1)
///  - stop the task T1, if the task is executed in a separate thread then it may still try to push the data it has just
///  produced. In that case the call to push the data won't succeed because the queue would have been disabled before
///  and the data will directly go back to the memory pool. It will then be possible for the task T1 to terminate
///  properly without blocking.
/// @tparam T Type of the elements stored in the queue
template<typename T>
class ConcurrentQueue {
public:
    /// @brief Constructor
    explicit ConcurrentQueue(size_t max_size = 0);

    /// @brief Destructor
    ~ConcurrentQueue();

    ConcurrentQueue(const ConcurrentQueue &) = delete;
    ConcurrentQueue(ConcurrentQueue &&)      = delete;
    ConcurrentQueue &operator=(const ConcurrentQueue &) = delete;
    ConcurrentQueue &operator=(ConcurrentQueue &&) = delete;

    /// @brief Retrieves the front element of the queue (i.e. the oldest one).
    ///
    /// If the queue is empty, this method waits until a new element is pushed to the queue, if the queue is closed in
    /// the meantime, then this method returns false and the front element is not retrieved.
    /// This method can be made non-blocking by setting the @p wait parameter to false.
    /// @param wait If false, the method will return immediately if the queue is empty
    /// @return The front element of the queue if this call succeeds
    std::optional<T> pop_front(bool wait = true);

    /// @brief Opens (i.e. enables) the queue. After the call it will be possible to push new elements
    void open();

    /// @brief Closes (i.e. disables) the queue. After the call it won't be possible to push new elements
    void close();

    /// @brief Pushes a new element to the queue.
    ///
    /// If the queue is full, this methods waits until a new element is popped out or until the queue is closed (in that
    /// latter case the element is not pushed).
    /// This method can be made non-blocking by setting the @p wait parameter to false.
    /// @param[in] elt The new element to push
    /// @param wait If false, the method will return immediately if the queue is full
    /// @return True if the element was successfully added, false otherwise
    bool emplace(T &&elt, bool wait = true);

    /// @brief Retrieves the size of the queue
    /// @return The size of the queue
    size_t size() const;

    /// @brief Clears the queue
    void clear();

private:
    const size_t max_size_;
    std::queue<T> q_;
    mutable std::mutex mtx_;
    std::condition_variable cond_;
    bool enabled_{false};
};
} // namespace Metavision

#include "metavision/sdk/core/utils/detail/concurrent_queue_impl.h"

#endif // METAVISION_SDK_CORE_CONCURRENT_QUEUE_H