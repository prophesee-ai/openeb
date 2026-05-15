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

#ifndef METAVISION_SDK_CORE_SHARED_CD_EVENTS_BUFFER_PRODUCER_ALGORITHM_H
#define METAVISION_SDK_CORE_SHARED_CD_EVENTS_BUFFER_PRODUCER_ALGORITHM_H

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/core/algorithms/shared_events_buffer_producer_algorithm.h"

namespace Metavision {

using SharedCdEventsBufferProducerAlgorithm = SharedEventsBufferProducerAlgorithm<EventCD>;

} // namespace Metavision

#endif // METAVISION_SDK_CORE_SHARED_CD_EVENTS_BUFFER_PRODUCER_ALGORITHM_H
