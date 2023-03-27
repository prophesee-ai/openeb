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

#ifndef METAVISION_HAL_TENCODER_DETAILS_H
#define METAVISION_HAL_TENCODER_DETAILS_H

#include <limits>
#include <tuple>

#include "event_encoder.h"

namespace Metavision {

namespace detail {

template<typename>
struct TupleofEventEncodersMaker {
    using Type = std::tuple<>;
};

template<typename EventIteratorType1, typename EventIteratorType2, typename... EventIteratorTypeRest>
struct TupleofEventEncodersMaker<std::tuple<EventIteratorType1, EventIteratorType2, EventIteratorTypeRest...>> {
    using TypeRest = typename TupleofEventEncodersMaker<std::tuple<EventIteratorTypeRest...>>::Type;
    using Type     = decltype(
        std::tuple_cat(std::declval<std::tuple<BatchEventEncoder<EventIteratorType1>>>(), std::declval<TypeRest>()));
};

template<typename TimeHighEncoderType, typename EventIteratorType1, typename EventIteratorType2,
         typename... EventIteratorTypeRest>
struct TupleofEncodersMaker {
    using TypeRest = typename TupleofEventEncodersMaker<
        std::tuple<EventIteratorType1, EventIteratorType2, EventIteratorTypeRest...>>::Type;
    using Type = decltype(std::tuple_cat(std::declval<std::tuple<TimeHighEncoderType>>(), std::declval<TypeRest>()));
};

template<typename TimeHighEncoderType, typename... EventIteratorTypeN>
struct TupleOfEncodersWrapper {
    using TupleOfEncodersType = typename TupleofEncodersMaker<TimeHighEncoderType, EventIteratorTypeN...>::Type;
    static constexpr size_t NUMBER_OF_ENCODERS = std::tuple_size<TupleOfEncodersType>::value;

    TupleOfEncodersWrapper(EventIteratorTypeN... pn) {
        register_encoders<1>(pn...);
        update_time<1>(); // From 1 and not 0 because we want to set next_timestamp_to_encode_ to time of the oldest of
                          // the events, in order to be able later to initialize the time high decoder
    }

    void init_time_high(Metavision::timestamp t) {
        // We set the time high decoder to t
        std::get<0>(t_).initialize(t);
        next_timestamp_to_encode_ = std::numeric_limits<Metavision::timestamp>::max();
        update_time<0>();
    }

    // Return true if it finished encoding, false if not (because no more available size)
    template<typename EvtFormat>
    bool encode_up_to(Metavision::timestamp upper_limit, uint8_t *&encoded_events_current_ptr,
                      const uint8_t *encoded_events_end) {
        while (next_timestamp_to_encode_ < upper_limit && !done()) {
            if (!encode_next<EvtFormat, 0>(encoded_events_current_ptr, encoded_events_end)) {
                return false;
            }
        }

        return true;
    }

    Metavision::timestamp get_next_timer_high() const {
        return std::get<0>(t_).get_next_timestamp_to_encode();
    }
    Metavision::timestamp next_timestamp_to_encode() const {
        return next_timestamp_to_encode_;
    }

    bool done() {
        return are_done<1>(); // From 1 because we do not consider timehigh
    }

private:
    template<std::size_t I>
    inline typename std::enable_if<I == NUMBER_OF_ENCODERS, bool>::type are_done() {
        return true;
    }

    template<std::size_t I>
    inline typename std::enable_if<(I > 0 && I < NUMBER_OF_ENCODERS), bool>::type are_done() {
        if (!std::get<I>(t_).is_done()) {
            return false;
        }
        return are_done<I + 1>();
    }

    template<std::size_t I>
    inline typename std::enable_if<I == NUMBER_OF_ENCODERS, void>::type update_time() {}

    template<std::size_t I>
    inline typename std::enable_if<(I < NUMBER_OF_ENCODERS), void>::type update_time() {
        Metavision::timestamp next_timestamp_to_encode = std::get<I>(t_).get_next_timestamp_to_encode();
        if (next_timestamp_to_encode < next_timestamp_to_encode_) {
            next_timestamp_to_encode_ = next_timestamp_to_encode;
            idx_next_encoder_         = I;
        }
        update_time<I + 1>();
    }

    template<std::size_t I, typename EventIteratorType1>
    inline void register_encoders(EventIteratorType1 p1, EventIteratorType1 p2) {
        std::get<I>(t_).register_buffer(p1, p2);
    }

    template<std::size_t I, typename EventIteratorType1, typename... EventIteratorTypeRest>
    inline void register_encoders(EventIteratorType1 p1, EventIteratorType1 p2, EventIteratorTypeRest... pn) {
        std::get<I>(t_).register_buffer(p1, p2);
        register_encoders<I + 1>(pn...);
    }

    template<typename EvtFormat, std::size_t I>
    inline typename std::enable_if<I == NUMBER_OF_ENCODERS, bool>::type
        encode_next(uint8_t *&encoded_events_current_ptr, const uint8_t *encoded_events_end) {
        return true;
    }

    template<typename EvtFormat, std::size_t I>
    inline typename std::enable_if<(I < NUMBER_OF_ENCODERS), bool>::type
        encode_next(uint8_t *&encoded_events_current_ptr, const uint8_t *encoded_events_end) {
        if (idx_next_encoder_ == I) {
            if (encoded_events_current_ptr + std::get<I>(t_).template get_size_encoded<EvtFormat>() >
                encoded_events_end) {
                return false;
            }
            std::get<I>(t_).template encode_next_event<EvtFormat>(encoded_events_current_ptr);
            encoded_events_current_ptr += std::get<I>(t_).template get_size_encoded<EvtFormat>();

            // NEED TO UPDATE NEXT TIMESTAMP
            next_timestamp_to_encode_ = std::numeric_limits<Metavision::timestamp>::max();
            update_time<0>();
            return true;
        }

        return encode_next<EvtFormat, I + 1>(encoded_events_current_ptr, encoded_events_end);
    }

    TupleOfEncodersType t_;
    int idx_next_encoder_{0};
    Metavision::timestamp next_timestamp_to_encode_{std::numeric_limits<Metavision::timestamp>::max()};
};

} // namespace detail
} // namespace Metavision

#endif // METAVISION_HAL_TENCODER_DETAILS_H
