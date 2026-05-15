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

#ifndef METAVISION_SDK_CORE_SHARED_EVENTS_BUFFER_PRODUCER_ALGORITHM_H
#define METAVISION_SDK_CORE_SHARED_EVENTS_BUFFER_PRODUCER_ALGORITHM_H

#include <vector>
#include <memory>
#include <algorithm>
#include <functional>

#include "metavision/sdk/core/algorithms/async_algorithm.h"
#include "metavision/sdk/base/utils/object_pool.h"

namespace Metavision {

/// @brief Default parameters for default policies
struct SharedEventsBufferProducerParameters {
    uint32_t buffers_events_count_{0};
    uint32_t buffers_time_slice_us_{5000};
    uint32_t buffers_pool_size_{32};
    uint32_t buffers_preallocation_size_{
        0}; ///< number of events to preallocate buffer with for efficiency purpose at insertion
    bool bounded_memory_pool_{true};
};

/// @brief A utility class to generate shared ptr around a vector of events according to a processing policy
/// (e.g. AsyncAlgorithm::Processing from @ref AsyncAlgorithm)
///
/// The events buffers are allocated within a bounded memory pool (@ref SharedObjectPool) to reuse the memory and
/// avoid memory allocation.
///
/// @tparam EventT The type of events contained in the buffer.
template<typename EventT>
class SharedEventsBufferProducerAlgorithm : public AsyncAlgorithm<SharedEventsBufferProducerAlgorithm<EventT>> {
public:
    // aliases
    using EventsBuffer = std::vector<EventT>;
    using EventsBufferPool =
        ObjectPool<EventsBuffer, true>; ///< @ref ObjectPool. We use a bounded one to avoid reallocation
    using SharedEventsBuffer = typename EventsBufferPool::ptr_type;
    using SharedEventsBufferProducedCb =
        std::function<void(timestamp, const SharedEventsBuffer &)>; ///< Alias of callback to process a
                                                                    ///< generated @ref SharedEventsBuffer

    /// @brief Constructor
    ///
    /// Supported mode from (e.g. AsyncAlgorithm::Processing from @ref AsyncAlgorithm): N_EVENTS, N_US,
    /// MIXED, NONE.
    ///
    /// Setting @ref SharedEventsBufferProducerParameters::buffers_events_count_ to 0 calls @ref set_processing_n_us
    /// (unless buffers_time_slice_us_ is 0 as well).
    ///
    /// Setting @ref SharedEventsBufferProducerParameters::buffers_time_slice_us_ to 0
    /// calls @ref set_processing_n_events (unless @ref SharedEventsBufferProducerParameters::buffers_events_count_ is 0
    /// as well).
    ///
    /// Setting @ref SharedEventsBufferProducerParameters::buffers_events_count_ and @ref
    /// SharedEventsBufferProducerParameters::buffers_time_slice_us_ to 0 calls @ref set_processing_external.
    ///
    /// Setting non zero value to both @ref SharedEventsBufferProducerParameters::buffers_events_count_ and @ref
    /// SharedEventsBufferProducerParameters::buffers_time_slice_us_ calls @ref set_processing_mixed.
    ///
    /// The mode can be overridden after calling the constructor.
    ///
    /// @param params An @ref SharedEventsBufferProducerParameters object containing the parameters.
    /// @param buffer_produced_cb A callback called (@ref SharedEventsBufferProducedCb) whenever a buffer is created.
    SharedEventsBufferProducerAlgorithm(SharedEventsBufferProducerParameters params,
                                        SharedEventsBufferProducedCb buffer_produced_cb);

    /// @brief Get the parameters used to construct this object
    inline SharedEventsBufferProducerParameters params() const;

    /// @brief Resets the internal states of the policy
    inline void clear();

private:
    /// @brief Function to process directly the events
    template<typename InputIt>
    inline void process_online(InputIt it_begin, InputIt it_end);

    /// @brief Function to process the state that is called every n_events or n_us
    inline void process_async(const timestamp processing_ts, const size_t n_processed_events);

    SharedEventsBufferProducedCb buffer_produced_cb_;
    EventsBufferPool buffers_pool_;
    SharedEventsBuffer current_shared_buffer_;
    SharedEventsBufferProducerParameters params_;

    friend AsyncAlgorithm<SharedEventsBufferProducerAlgorithm>;
};

} // namespace Metavision

#include "metavision/sdk/core/algorithms/detail/shared_events_buffer_producer_algorithm_impl.h"

#endif // METAVISION_SDK_CORE_SHARED_EVENTS_BUFFER_PRODUCER_ALGORITHM_H
