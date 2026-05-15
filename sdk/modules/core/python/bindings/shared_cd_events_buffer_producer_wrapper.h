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

#ifndef METAVISION_SDK_CORE_SHARED_CD_EVENTS_BUFFER_PRODUCER_WRAPPER_H
#define METAVISION_SDK_CORE_SHARED_CD_EVENTS_BUFFER_PRODUCER_WRAPPER_H

#include <vector>
#include <memory>
#include <algorithm>
#include <functional>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/core/algorithms/shared_cd_events_buffer_producer_algorithm.h"

namespace Metavision {

// aliases

using EventsBufferPtr = SharedCdEventsBufferProducerAlgorithm::SharedEventsBuffer;
using BufferCallback  = SharedCdEventsBufferProducerAlgorithm::SharedEventsBufferProducedCb;

/// @brief A wrapping class (@ref
/// SharedCdEventsBufferProducer)
///
/// The events buffers are allocated within an unbounded memory pool (@ref ObjectPool) to reuse the memory
/// and avoid memory allocation.
class SharedCdEventsBufferProducerWrapper : public SharedCdEventsBufferProducerAlgorithm {
public:
    SharedCdEventsBufferProducerWrapper(SharedEventsBufferProducerParameters params,
                                        BufferCallback buffer_produced_cb) :
        SharedCdEventsBufferProducerAlgorithm{params, buffer_produced_cb} {}
    /// @brief Returns the cpp callback to put it in a decoder loop from metavision HAL.
    std::function<void(const EventCD *, const EventCD *)> get_process_events_callback();
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_SHARED_CD_EVENTS_BUFFER_PRODUCER_WRAPPER_H
