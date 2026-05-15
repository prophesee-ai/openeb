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

#ifndef METAVISION_SDK_CORE_COUNTER_MAP_H
#define METAVISION_SDK_CORE_COUNTER_MAP_H

#include <map>
#include <mutex>
#include <iostream>

namespace Metavision {

/// @brief Associate a counter to a key
template<typename KeyT>
class CounterMap {
public:
    /// @brief Constructs a CounterMap
    CounterMap() {}

    /// @brief Destructs a CounterMap
    ~CounterMap() {}

    /// @brief Increments the reference counter associated to the input key
    ///
    /// If the key was not existing, then it is added and its associated reference count is initialized to 1
    ///
    /// @return The current count for the key
    size_t tag(KeyT key) {
        std::unique_lock<std::mutex> lock(tag_mutex_);
        return ++tag_[key];
    }

    /// @brief Decrements the reference counter associated to the input key
    ///
    /// If the counter goes to 0, the reference counter for the key is erased from the map until it is tagged again
    ///
    /// @return The current count for the key
    size_t untag(KeyT key) {
        std::unique_lock<std::mutex> lock(tag_mutex_);
        auto it = tag_.find(key);
        if (it != tag_.end()) {
            --(it->second);
            if (it->second == 0) {
                tag_.erase(it);
                return 0;
            }
            return it->second;
        }

        return 0;
    }

    /// @brief Returns the input key reference counter value
    /// @return The current count for the key
    size_t tag_count(KeyT key) const {
        std::unique_lock<std::mutex> lock(tag_mutex_);
        auto it = tag_.find(key);
        if (it != tag_.end()) {
            return it->second;
        }
        return 0;
    }

    /// @brief Checks if the map is empty
    /// @return true if empty, false otherwise
    bool empty() const {
        return tag_.empty();
    }

private:
    mutable std::mutex tag_mutex_;
    std::map<KeyT, size_t> tag_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_COUNTER_MAP_H
