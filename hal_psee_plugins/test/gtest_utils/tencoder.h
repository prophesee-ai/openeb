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

#ifndef METAVISION_HAL_TENCODER_H
#define METAVISION_HAL_TENCODER_H

#include <functional>
#include <type_traits>
#include <vector>

#include "tencoder_details.h"
#include "encoding_policies.h"
#include "event_raw_format_traits.h"
#include "timer_high_encoder.h"

namespace Metavision {
struct Evt21RawFormat;

template<typename EvtFormat, typename TimerHighRedundancyPolicy = TimerHighRedundancyNone>
class TEncoder {
public:
    using VoidTypeCallbackRawWord = std::function<void(const uint8_t *, const uint8_t *)>;

    TEncoder(Metavision::timestamp t_step = 5000) : current_size_(SIZE_BUFFER_BEGIN_), step_(t_step) {
        encoded_events_.resize(current_size_);
        init_pointers();
    }

    ~TEncoder() {}

    void set_encode_event_callback(const VoidTypeCallbackRawWord &cb) {
        cb_on_encode_ = cb;
    }

    void reset() {
        is_initialized = false;
    }

    template<typename... EventIteratorTypeN>
    void encode(EventIteratorTypeN... pn) {
        static_assert(
            sizeof...(EventIteratorTypeN) > 1,
            "Error : need to call encode() with at least 2 parameters, representing the range of the events to encode");
        static_assert(sizeof...(EventIteratorTypeN) % 2 == 0,
                      "Error : need to call encode() with an even number of parameters.");
        detail::TupleOfEncodersWrapper<TimerHighEncoder<EvtFormat, TimerHighRedundancyPolicy>, EventIteratorTypeN...>
            ts(pn...);

        if (ts.done()) {
            return;
        }

        if (!is_initialized) {
            ts.init_time_high(ts.next_timestamp_to_encode());
            next_upper_limit = (ts.next_timestamp_to_encode() / step_ + 1) * step_;

            init_pointers();

            is_initialized = true;
        } else {
            ts.init_time_high(last_next_th);
        }

        while (true) {
            // Encode events
            if (ts.template encode_up_to<EvtFormat>(next_upper_limit, encoded_event_current_ptr_,
                                                    encoded_event_end_ptr_)) {
                if (!ts.done()) {
                    write_packet();
                } else {
                    break;
                }
            } else {
                // Means we need to allocate more memory
                double_size();
            }
        }

        last_next_th = ts.get_next_timer_high();
    }

    // To update last packet
    void flush() {
        if (encoded_event_current_ptr_ != encoded_events_.data()) {
            write_packet();
        }
    }

private:
    void write_packet() {
        uint8_t *end = encoded_event_current_ptr_;

        // Call the cb and reset pointers :
        cb_on_encode_(encoded_events_.data(), end);

        // Reset :
        init_pointers();
        next_upper_limit += step_;
    }

    void init_pointers() {
        // Add begin manual event :
        // Cannot encode it right away because we do not know the size yet of the buffer
        // so we just store the place for now :
        encoded_event_current_ptr_ = encoded_events_.data();
        encoded_event_end_ptr_     = encoded_events_.data() + current_size_; // Keep the place for the last manual event
    }

    void double_size() {
        auto old_position = std::distance(encoded_events_.data(), encoded_event_current_ptr_);
        current_size_     = 2 * current_size_;
        encoded_events_.resize(current_size_);
        encoded_event_current_ptr_ = encoded_events_.data() + old_position;
        encoded_event_end_ptr_     = encoded_events_.data() + current_size_;
    }

    bool is_initialized = false;

    static void callback_on_encode_default(const uint8_t *, const uint8_t *) {}
    VoidTypeCallbackRawWord cb_on_encode_ = callback_on_encode_default;

    static constexpr size_t SIZE_BUFFER_BEGIN_ = 1000;
    size_t current_size_;
    std::vector<uint8_t> encoded_events_;
    uint8_t *encoded_event_current_ptr_;
    uint8_t *encoded_event_end_ptr_;

    Metavision::timestamp last_next_th = 0;

    const Metavision::timestamp step_;
    Metavision::timestamp next_upper_limit{0};
};

} // namespace Metavision

#endif // METAVISION_HAL_TENCODER_H
