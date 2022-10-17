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

#ifndef METAVISION_SDK_CORE_DETAIL_SHARED_EVENTS_BUFFER_PRODUCER_ALGORITHM_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_SHARED_EVENTS_BUFFER_PRODUCER_ALGORITHM_IMPL_H

namespace Metavision {

template<typename EventT>
SharedEventsBufferProducerAlgorithm<EventT>::SharedEventsBufferProducerAlgorithm(
    SharedEventsBufferProducerParameters params, SharedEventsBufferProducedCb buffer_produced_cb) :
    buffer_produced_cb_(buffer_produced_cb),
    buffers_pool_(params.bounded_memory_pool_ ? EventsBufferPool::make_bounded(params.buffers_pool_size_) :
                                                EventsBufferPool::make_unbounded(params.buffers_pool_size_)),
    params_(params) {
    // Preallocate the memory of each buffer in the object pool
    {
        std::vector<SharedEventsBuffer> acquired_for_allocation;
        while (!buffers_pool_.empty()) {
            acquired_for_allocation.push_back(buffers_pool_.acquire());
            acquired_for_allocation.back()->reserve(params.buffers_preallocation_size_);
        }
    }

    current_shared_buffer_ = buffers_pool_.acquire();
    current_shared_buffer_->reserve(params_.buffers_preallocation_size_);

    if (params.buffers_events_count_ == 0 && params.buffers_time_slice_us_ != 0) {
        this->set_processing_n_us(params.buffers_time_slice_us_);
    } else if (params.buffers_time_slice_us_ == 0 && params.buffers_events_count_ != 0) {
        this->set_processing_n_events(params.buffers_events_count_);
    } else if (params.buffers_events_count_ != 0 && params.buffers_time_slice_us_ != 0) {
        this->set_processing_mixed(params.buffers_events_count_, params.buffers_time_slice_us_);
    } else {
        this->set_processing_external();
    }
}

template<typename EventT>
SharedEventsBufferProducerParameters SharedEventsBufferProducerAlgorithm<EventT>::params() const {
    return params_;
}

template<typename EventT>
void SharedEventsBufferProducerAlgorithm<EventT>::clear() {
    current_shared_buffer_->clear();
}

template<typename EventT>
template<typename InputIt>
void SharedEventsBufferProducerAlgorithm<EventT>::process_online(InputIt it_begin, InputIt it_end) {
    current_shared_buffer_->insert(current_shared_buffer_->end(), it_begin, it_end);
}

template<typename EventT>
void SharedEventsBufferProducerAlgorithm<EventT>::process_async(const timestamp processing_ts,
                                                                const size_t n_processed_events) {
    buffer_produced_cb_(processing_ts, current_shared_buffer_);

    current_shared_buffer_ = buffers_pool_.acquire();

    // In the bounded memory case, the memory is already allocated and this is equivalent to a single if if memory is
    // already allocated.
    // In the unbounded memory case, if needs to reallocate a new buffer, this will reserve the requested memory
    current_shared_buffer_->reserve(params_.buffers_preallocation_size_);
    current_shared_buffer_->clear();
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_SHARED_EVENTS_BUFFER_PRODUCER_ALGORITHM_IMPL_H
