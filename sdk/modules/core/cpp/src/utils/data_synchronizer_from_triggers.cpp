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

#include "metavision/sdk/core/utils/data_synchronizer_from_triggers.h"

namespace Metavision {
DataSynchronizerFromTriggers::Parameters::Parameters(uint32_t period_us) : period_us_(period_us) {
    if (period_us_ == 0) {
        throw std::invalid_argument("In Events synchronizer from triggers, period must be strictly greater than 0.");
    }
}

DataSynchronizerFromTriggers::DataSynchronizerFromTriggers(const Parameters &parameters) : parameters_(parameters) {
    reset_synchronization();
}

DataSynchronizerFromTriggers::~DataSynchronizerFromTriggers() {
    set_synchronization_as_done();
}

void DataSynchronizerFromTriggers::reset_synchronization() {
    set_synchronization_as_done();
    std::lock_guard<std::mutex> lock(triggers_updated_mutex_);
    synchronization_information_deque_.clear();
    first_trigger_indexed_      = false;
    triggers_source_is_done_    = false;
    last_synchronization_index_ = 0;
    last_synchronization_ts_us_ = 0;
}

void DataSynchronizerFromTriggers::set_synchronization_as_done() {
    std::lock_guard<std::mutex> lock(triggers_updated_mutex_);
    triggers_source_is_done_ = true;
    wait_for_triggers_cond_.notify_all();
    wait_for_triggers_consumed_cond_.notify_all();
}

void DataSynchronizerFromTriggers::wait_for_triggers_consumed(uint32_t max_remaining_to_be_consumed) {
    std::unique_lock<std::mutex> lock(triggers_updated_mutex_);
    if (triggers_source_is_done_)
        return;
    wait_for_triggers_consumed_cond_.wait(lock, [this, max_remaining_to_be_consumed]() {
        return triggers_source_is_done_ || synchronization_information_deque_.size() <= max_remaining_to_be_consumed;
    });
}
} // namespace Metavision
