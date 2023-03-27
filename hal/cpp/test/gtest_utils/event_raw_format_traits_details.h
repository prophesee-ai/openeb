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

#ifndef METAVISION_HAL_EVENT_RAW_FORMAT_TRAITS_DETAILS_H
#define METAVISION_HAL_EVENT_RAW_FORMAT_TRAITS_DETAILS_H

#include <cstddef>

namespace Metavision {
namespace detail {

template<typename EvtFormat, typename DecodedEventType>
struct encoded_event_type {};

// Common class for Evt 1, 2 and 2.1 (evt 3 is too different, need to be handled separately)
template<typename EvtFormat, typename BaseRaw>
struct event_raw_format_common_base {
    using BaseEventType = BaseRaw;

    // Encoded Event
    template<typename DecodedEventType>
    struct EncodedEvent {
        using Type = typename detail::encoded_event_type<EvtFormat, DecodedEventType>::Type;
        static constexpr size_t number_of_words() {
            return std::max(sizeof(Type), sizeof(BaseEventType)) / sizeof(BaseEventType); // Because of EVT2.1
        };
        static constexpr size_t number_of_bytes() {
            return number_of_words() * sizeof(BaseEventType);
        };
    };
};
} // namespace detail
} // namespace Metavision

#endif // METAVISION_HAL_EVENT_RAW_FORMAT_TRAITS_DETAILS_H
