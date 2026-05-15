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

#ifndef METAVISION_SDK_CORE_CONCURRENT_QUEUE_IMPL_H
#define METAVISION_SDK_CORE_CONCURRENT_QUEUE_IMPL_H

#include "metavision/sdk/base/utils/log.h"

namespace Metavision {

template<typename T>
ConcurrentQueue<T>::ConcurrentQueue(size_t max_size) : max_size_(max_size) {
    open();
}

template<typename T>
ConcurrentQueue<T>::~ConcurrentQueue() {
    close();
}

template<typename T>
std::optional<T> ConcurrentQueue<T>::pop_front(bool wait) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (q_.empty() && enabled_) {
        if (!wait) {
            return std::nullopt;
        }
        cond_.wait(lock, [&]() { return !q_.empty() || !enabled_; });
    }

    if (q_.empty() && !enabled_) {
        return std::nullopt;
    }

    auto front = std::move(q_.front());
    q_.pop();
    cond_.notify_one();

    return std::move(front);
}

template<typename T>
void ConcurrentQueue<T>::open() {
    std::lock_guard<std::mutex> lock(mtx_);
    enabled_ = true;
}

template<typename T>
void ConcurrentQueue<T>::close() {
    std::lock_guard<std::mutex> lock(mtx_);

    enabled_ = false;

    cond_.notify_all();
}

template<typename T>
bool ConcurrentQueue<T>::emplace(T &&elt, bool wait) {
    std::unique_lock<std::mutex> lock(mtx_);

    const bool queue_is_full = max_size_ != 0 && q_.size() == max_size_;
    if (enabled_ && queue_is_full) {
        if (!wait) {
            return false;
        }
        cond_.wait(lock, [&]() { return q_.size() < max_size_ || !enabled_; });
    }

    if (!enabled_) {
        return false;
    }

    q_.emplace(std::forward<T>(elt));
    cond_.notify_one();
    return true;
}

template<typename T>
size_t ConcurrentQueue<T>::size() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return q_.size();
}

template<typename T>
void ConcurrentQueue<T>::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    q_ = {};

    cond_.notify_one();
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_CONCURRENT_QUEUE_IMPL_H