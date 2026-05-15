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

#ifndef METAVISION_HAL_EVENT_ENCODER_IMPL_H
#define METAVISION_HAL_EVENT_ENCODER_IMPL_H

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/hal/decoders/base/base_event_types.h"

namespace Metavision {

template<typename DecodedEventtype>
template<typename EvtFormat>
void EventEncoder<DecodedEventtype>::encode_event( // Default is Event CD
    typename event_raw_format_traits<EvtFormat>::template EncodedEvent<DecodedEventtype>::Type *ev_encoded,
    const DecodedEventtype *ev_decoded) {
    ev_encoded->x         = ev_decoded->x;
    ev_encoded->y         = ev_decoded->y;
    ev_encoded->timestamp = ev_decoded->t;
    ev_encoded->type      = ev_decoded->p ?
                                static_cast<EventTypesUnderlying_t>(event_raw_format_traits<EvtFormat>::EnumType::CD_ON) :
                                static_cast<EventTypesUnderlying_t>(event_raw_format_traits<EvtFormat>::EnumType::CD_OFF);
}

template<>
template<typename EvtFormat>
void EventEncoder<Metavision::EventExtTrigger>::encode_event(
    typename event_raw_format_traits<EvtFormat>::template EncodedEvent<Metavision::EventExtTrigger>::Type *ev_encoded,
    const Metavision::EventExtTrigger *ev_decoded) {
    ev_encoded->timestamp = ev_decoded->t;
    ev_encoded->id        = ev_decoded->id;
    ev_encoded->value     = ev_decoded->p;
    ev_encoded->type = static_cast<EventTypesUnderlying_t>(event_raw_format_traits<EvtFormat>::EnumType::EXT_TRIGGER);
}
} // namespace Metavision

#endif // METAVISION_HAL_EVENT_ENCODER_IMPL_H
