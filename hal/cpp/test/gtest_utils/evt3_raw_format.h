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

#ifndef METAVISION_HAL_EVT3_RAW_FORMAT_H
#define METAVISION_HAL_EVT3_RAW_FORMAT_H

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/hal/decoders/base/event_base.h"
#include "metavision/hal/decoders/evt3/evt3_event_types.h"
#include "event_raw_format_traits.h"
#include "event_raw_format_traits_details.h"

namespace Metavision {

struct Evt3RawFormat {};

template<>
struct event_raw_format_traits<Evt3RawFormat> {
    using BaseEventType             = Evt3Raw::RawEvent;
    using EnumType                  = Evt3EventTypes_4bits;
    static const char NLowerBitsTH  = 12;
    static const char NHigherBitsTH = 12;
    static constexpr Metavision::timestamp MaxTimestamp =
        Metavision::timestamp((1 << (NLowerBitsTH + NHigherBitsTH)) - 1);
};
constexpr Metavision::timestamp event_raw_format_traits<Evt3RawFormat>::MaxTimestamp;

} // namespace Metavision

#endif // METAVISION_HAL_EVT3_RAW_FORMAT_H
