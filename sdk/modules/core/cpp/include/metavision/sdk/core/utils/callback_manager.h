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

#ifndef METAVISION_SDK_CORE_CALLBACK_MANAGER_H
#define METAVISION_SDK_CORE_CALLBACK_MANAGER_H

#include <atomic>
#include <map>
#include <mutex>
#include <vector>

#include "metavision/sdk/core/utils/index_manager.h"

namespace Metavision {

template<class EventsCallback, typename TagType = uint8_t>
class CallbackManager {
public:
    CallbackManager(IndexManager &index_manager) : index_manager_(index_manager) {}

    CallbackManager(IndexManager &index_manager, uint8_t tag_id) : index_manager_(index_manager), tag_id_(tag_id) {}

    virtual ~CallbackManager() {}

    size_t add_callback(const EventsCallback &cb) {
        std::unique_lock<std::mutex> lock(cbs_mutex_);
        auto idx = index_manager_.index_generator_.get_next_index();
        index_manager_.counter_map_.tag(tag_id_);
        cbs_map_[idx]  = cb;
        cbs_vec_dirty_ = true;
        return idx;
    }

    bool remove_callback(size_t callback_id) {
        std::unique_lock<std::mutex> lock(cbs_mutex_);
        auto it = cbs_map_.find(callback_id);
        if (it != cbs_map_.end()) {
            cbs_map_.erase(it);
            index_manager_.counter_map_.untag(tag_id_);
            cbs_vec_dirty_ = true;
            return true;
        }
        return false;
    }

    const std::vector<EventsCallback> &get_cbs() const {
        if (cbs_vec_dirty_) {
            std::unique_lock<std::mutex> lock(cbs_mutex_);
            cbs_vec_.clear();
            for (auto &&p : cbs_map_) {
                cbs_vec_.push_back(p.second);
            }
            cbs_vec_dirty_ = false;
        }
        return cbs_vec_;
    }

    template<typename... Args>
    void operator()(Args &&...params) {
        auto cbs = get_cbs();

        for (auto &cb : cbs)
            cb(std::forward<Args>(params)...);
    }

private:
    IndexManager &index_manager_;
    TagType tag_id_ = std::numeric_limits<TagType>::max();
    mutable std::mutex cbs_mutex_;
    mutable std::atomic<bool> cbs_vec_dirty_{false};
    std::map<size_t, EventsCallback> cbs_map_;
    mutable std::vector<EventsCallback> cbs_vec_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_CALLBACK_MANAGER_H
