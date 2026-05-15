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

#ifndef METAVISION_HAL_EVT2_RAW_FORMAT_H
#define METAVISION_HAL_EVT2_RAW_FORMAT_H

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/hal/decoders/base/event_base.h"
#include "metavision/hal/decoders/evt2/evt2_event_types.h"
#include "event_raw_format_traits.h"
#include "event_raw_format_traits_details.h"

namespace Metavision {

struct Evt2RawFormat {};

namespace detail {

template<>
struct encoded_event_type<Evt2RawFormat, Metavision::EventCD> {
    using Type = EVT2Event2D;
};

template<>
struct encoded_event_type<Evt2RawFormat, Metavision::EventExtTrigger> {
    using Type = EVT2EventExtTrigger;
};

// By default Event2d for CD
template<>
struct encoded_event_type<Evt2RawFormat, Metavision::Event2d> {
    using Type = typename encoded_event_type<Evt2RawFormat, Metavision::EventCD>::Type;
};

} // namespace detail

template<>
struct event_raw_format_traits<Evt2RawFormat>
    : public detail::event_raw_format_common_base<Evt2RawFormat, EventBase::RawEvent> {
    using EnumType = EVT2EventTypes;

    // number of lower bits of TIME HIGH
    static const char NLowerBitsTH                      = EVT2EventsTimeStampBits;
    static constexpr Metavision::timestamp MaxTimestamp = Metavision::timestamp((1 << 28) - 1)
                                                          << EVT2EventsTimeStampBits;
};

} // namespace Metavision

#endif // METAVISION_HAL_EVT2_RAW_FORMAT_H
