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

#ifndef METAVISION_SDK_CORE_DETAIL_DATA_SYNCHRONIZER_FROM_TRIGGERS_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_DATA_SYNCHRONIZER_FROM_TRIGGERS_IMPL_H

#include <vector>

namespace Metavision {

template<typename ExtTriggerIterator>
size_t DataSynchronizerFromTriggers::index_triggers(ExtTriggerIterator trigger_it, ExtTriggerIterator trigger_it_end) {
    std::vector<EventExtTrigger> out;
    auto back_ins = std::back_inserter(out);
    index_triggers(trigger_it, trigger_it_end, back_ins);
    return out.size();
}

template<typename ExtTriggerIterator, typename IndexTriggerInserterIterator>
void DataSynchronizerFromTriggers::index_triggers(ExtTriggerIterator trigger_it, ExtTriggerIterator trigger_it_end,
                                                  IndexTriggerInserterIterator indexed_trigger_inserter_it) {
    static_assert(is_const_iterator_over<ExtTriggerIterator, EventExtTrigger>::value,
                  "Requires an iterator over EventExtTrigger element.");

    static_assert(detail::is_back_inserter_iterator_v<IndexTriggerInserterIterator>,
                  "Requires a back inserter iterator.");

    static_assert(
        std::is_same<typename iterator_traits<IndexTriggerInserterIterator>::value_type, EventExtTrigger>::value,
        "Requires an output back inserter iterator over EventExtTrigger element.");

    std::lock_guard<std::mutex> lock(triggers_updated_mutex_);
    for (; trigger_it != trigger_it_end; ++trigger_it, ++indexed_trigger_inserter_it) {
        // Consider the trigger only if polarity matches the reference one
        if (trigger_it->p != parameters_.reference_polarity_) {
            continue;
        }

        // Discard trigger if requested
        if (parameters_.to_discard_count_ != 0) {
            --parameters_.to_discard_count_;
            continue;
        }

        if (first_trigger_indexed_) {
            // We expect the new trigger to be at least (around) period_us in the future

            // This trigger is in the past or is a duplicate.
            // This should never happen unless input trigger stream has been reordered
            if (trigger_it->t <=
                (last_synchronization_ts_us_ + parameters_.periodicity_tolerance_factor_ * parameters_.period_us_)) {
                continue;
            }

            // ------------------------------
            // Compute the new trigger index
            const uint32_t new_synchronization_index =
                last_synchronization_index_ +
                std::round(static_cast<double>(trigger_it->t - last_synchronization_ts_us_) / parameters_.period_us_);

            // If trigger index > current_index + 1, then we missed some triggers. We need to interpolate
            // synchronization data.
            for (uint32_t interpolated_index = last_synchronization_index_ + 1;
                 interpolated_index < new_synchronization_index; ++interpolated_index, ++indexed_trigger_inserter_it) {
                // Interpolate the timestamp from the last  trigger received timestamp
                last_synchronization_ts_us_ += parameters_.period_us_;

                // Push the trigger in the queue for synchronization
                synchronization_information_deque_.push_back({last_synchronization_ts_us_, interpolated_index});
                *indexed_trigger_inserter_it =
                    EventExtTrigger(trigger_it->p, last_synchronization_ts_us_, trigger_it->id);
            }

            // Keep in memory last trigger index for interpolation
            last_synchronization_index_ = new_synchronization_index;
        } else {
            // ------------------------------
            // First trigger received. Index starts from user offset.
            first_trigger_indexed_      = true;
            last_synchronization_index_ = parameters_.index_offset_;
        }

        last_synchronization_ts_us_ = trigger_it->t;
        synchronization_information_deque_.push_back({last_synchronization_ts_us_, last_synchronization_index_});
        *indexed_trigger_inserter_it = EventExtTrigger(trigger_it->p, last_synchronization_ts_us_, trigger_it->id);
    }

    wait_for_triggers_cond_.notify_all();
}

template<typename DataIterator>
uint32_t DataSynchronizerFromTriggers::synchronize_data_from_triggers(
    DataIterator data_it_begin, DataIterator data_it_end,
    std::function<timestamp &(detail::value_t<DataIterator> &)> data_timestamp_accessor,
    std::function<uint32_t(const detail::value_t<DataIterator> &)> data_index_accessor) {
    auto data_it = data_it_begin;
    std::unique_lock<std::mutex> lock(triggers_updated_mutex_);

    for (; data_it != data_it_end; ++data_it) {
        // ------------------------------
        // Check if the last sync info is older than the one to be synchronized.
        // Possible cases:
        // 1- There is sync information in the queue and one has the same index as the current data
        // 2- No sync info is available in the queue -> we need to wait for some
        // 3- Sync info are available in the queue but the most recent index is lower than the input data to
        // synchronize
        // 4- Sync info are available in the queue but the oldest one's index is greater than the input data to
        // synchronize

        // Cases 2 & 3 -> Need to wait for sync info
        bool has_synchronization_info = false;

        // We wait for new synchronization information if needed.
        wait_for_triggers_cond_.wait(lock, [&]() {
            // We check here if we have enough information to proceed with the synchronization
            has_synchronization_info = !synchronization_information_deque_.empty() &&
                                       last_synchronization_index_ >= data_index_accessor(*data_it);

            if (!has_synchronization_info) {
                if (!synchronization_information_deque_.empty()) {
                    // The last index is lower than the idx that we are searching. We consume the triggers to avoid
                    // a deadlock
                    synchronization_information_deque_.clear();
                }

                // If not, all triggers have been used for synchronization. We notify to unlock any process
                // waiting on this condition
                wait_for_triggers_consumed_cond_.notify_all();
            }
            return has_synchronization_info || triggers_source_is_done_;
        });

        if (!has_synchronization_info) {
            // Source is done and no sync info remains
            break;
        }
        // Here, even if source is done but we have triggers to synchronize, we use all we have.

        // Case 4 -> interpolates sync info in the past
        if (data_index_accessor(*data_it) < synchronization_information_deque_.front().index) {
            SynchronizationInformation sync_info_past_interpolated = synchronization_information_deque_.front();
            const auto data_index                                  = data_index_accessor(*data_it);
            while (data_index < sync_info_past_interpolated.index) {
                --sync_info_past_interpolated.index;
                sync_info_past_interpolated.t -= parameters_.period_us_;
                synchronization_information_deque_.push_front(sync_info_past_interpolated);
            }
        }

        // Case 1 -> We have enough triggers
        SynchronizationInformation sync_information = synchronization_information_deque_.front();
        synchronization_information_deque_.pop_front();
        while (data_index_accessor(*data_it) !=
               sync_information.index) { // No need to check on size: the above checks ensure that the
            // synchronization information is in the queue.
            sync_information = synchronization_information_deque_.front();
            synchronization_information_deque_.pop_front();
        }

        data_timestamp_accessor(*data_it) = sync_information.t;
    }

    wait_for_triggers_consumed_cond_.notify_all();

    // compute the amount of data synchronized
    return std::distance(data_it_begin, data_it);
}
} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_DATA_SYNCHRONIZER_FROM_TRIGGERS_IMPL_H
