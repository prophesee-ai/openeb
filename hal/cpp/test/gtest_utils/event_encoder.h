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

#ifndef METAVISION_HAL_EVENT_ENCODER_H
#define METAVISION_HAL_EVENT_ENCODER_H

#include <type_traits>
#include <iterator>
#include <stddef.h>

#include "metavision/sdk/base/utils/timestamp.h"
#include "event_raw_format_traits.h"

namespace Metavision {

template<typename DecodedEventtype>
struct EventEncoder {
    template<typename EvtFormat>
    static void encode_event(
        typename event_raw_format_traits<EvtFormat>::template EncodedEvent<DecodedEventtype>::Type *ev_encoded,
        const DecodedEventtype *ev_decoded);
};

template<typename EventIteratorType>
struct BatchEventEncoder {
    using DecodedEventType = decltype(typename std::iterator_traits<EventIteratorType>::value_type());

    void register_buffer(EventIteratorType begin, EventIteratorType end) {
        next_to_encode_ = begin;
        last_           = end;
    }

    inline Metavision::timestamp get_next_timestamp_to_encode() {
        return (next_to_encode_ == last_) ? std::numeric_limits<Metavision::timestamp>::max() : next_to_encode_->t;
    }

    template<typename EvtFormat>
    static constexpr size_t get_size_encoded() {
        return event_raw_format_traits<EvtFormat>::template EncodedEvent<DecodedEventType>::number_of_bytes();
    }

    template<typename EvtFormat>
    inline void encode_next_event(uint8_t *encoded_ev) {
        auto ev_td = reinterpret_cast<
            typename event_raw_format_traits<EvtFormat>::template EncodedEvent<DecodedEventType>::Type *>(encoded_ev);
        EventEncoder<DecodedEventType>::template encode_event<EvtFormat>(ev_td, &*next_to_encode_);
        ++next_to_encode_;
    }

    inline bool is_done() const {
        return next_to_encode_ == last_;
    }

private:
    EventIteratorType next_to_encode_;
    EventIteratorType last_;
};

} // namespace Metavision

#include "event_encoder_impl.h"

#endif // METAVISION_HAL_EVENT_ENCODER_H
