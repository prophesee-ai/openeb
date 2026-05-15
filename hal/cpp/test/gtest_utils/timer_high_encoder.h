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

#ifndef METAVISION_HAL_TIMER_HIGH_ENCODER_H
#define METAVISION_HAL_TIMER_HIGH_ENCODER_H

#include <type_traits>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/hal/decoders/base/base_event_types.h"
#include "event_raw_format_traits.h"

namespace Metavision {

template<typename EvtFormat, typename TimerHighRedundancyPolicy>
struct TimerHighEncoder {
    static constexpr char N_LOWER_BITS_TH               = event_raw_format_traits<EvtFormat>::NLowerBitsTH;
    static constexpr Metavision::timestamp TH_STEP      = (1ul << N_LOWER_BITS_TH);
    static constexpr Metavision::timestamp TH_NEXT_STEP = TH_STEP / TimerHighRedundancyPolicy::REDUNDANCY_FACTOR;

    // This function does not need to be template, but doing so we can use the same syntax as for the BatchEventEncoder,
    // thus generalizing better the class TupleOfEncoders (because we do not need to handle the special case of
    // time high, it will be automatically done).
    // However, it does not make sense for the template type to be != from EvtFormat, so we enforce it
    template<typename Format = EvtFormat, typename = std::enable_if<std::is_same<Format, EvtFormat>::value>>
    static constexpr size_t get_size_encoded() {
        return sizeof(typename event_raw_format_traits<EvtFormat>::BaseEventType);
    }

    TimerHighEncoder() {}

    /// Initialize next time high
    void initialize(Metavision::timestamp base) {
        next_th_ = (base / TH_NEXT_STEP) * TH_NEXT_STEP;
        update_current_th<TimerHighRedundancyPolicy::REDUNDANCY_FACTOR>();
    }

    // This function does not need to be template, but doing so we can use the same syntax as for the BatchEventEncoder,
    // thus generalizing better the class TupleOfEncoders (because we do not need to handle the special case of
    // time high, it will be automatically done).
    // However, it does not make sense for the template type to be != from EvtFormat, so we enforce it
    template<typename Format = EvtFormat, typename = std::enable_if<std::is_same<Format, EvtFormat>::value>>
    void encode_next_event(uint8_t *encoded_ev) {
        auto ev_th   = reinterpret_cast<typename event_raw_format_traits<Format>::BaseEventType *>(encoded_ev);
        ev_th->trail = next_th_ >> N_LOWER_BITS_TH;
        ev_th->type  = static_cast<EventTypesUnderlying_t>(event_raw_format_traits<Format>::EnumType::EVT_TIME_HIGH);

        update_current_th<TimerHighRedundancyPolicy::REDUNDANCY_FACTOR>();
        next_th_ += TH_NEXT_STEP;
    }

    Metavision::timestamp get_next_timestamp_to_encode() const {
        return next_th_;
    }

    Metavision::timestamp get_current_time_high() const {
        return current_th_;
    }

private:
    template<unsigned int FACTOR>
    typename std::enable_if<(FACTOR <= 1), void>::type update_current_th() {
        current_th_ = next_th_;
    }

    template<unsigned int FACTOR>
    typename std::enable_if<!(FACTOR <= 1), void>::type update_current_th() {
        current_th_ = next_th_ & ~((static_cast<Metavision::timestamp>(1) << N_LOWER_BITS_TH) - 1);
    }

    Metavision::timestamp current_th_ = 0;
    Metavision::timestamp next_th_    = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_TIMER_HIGH_ENCODER_H
