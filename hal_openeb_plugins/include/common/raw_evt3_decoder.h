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

#ifndef RAW_EVT3_DECODER_H
#define RAW_EVT3_DECODER_H

#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/hal/facilities/i_decoder.h>
#include <metavision/hal/facilities/i_event_decoder.h>

#include "raw_evt_base.h"

namespace Metavision {
namespace EVT3 {

union Mask {
    uint32_t valid;
    struct MaskVect32 {
        uint32_t valid1 : 12;
        uint32_t valid2 : 12;
        uint32_t valid3 : 8;
    } m;
};

struct EventPosX {
    uint16_t x : 11;
    uint16_t pol : 1;
    uint16_t type : 4;
};

struct EventVect12_12_8 {
    uint16_t valid1 : 12;
    uint16_t type1 : 4;
    uint16_t valid2 : 12;
    uint16_t type2 : 4;
    uint16_t valid3 : 8;
    uint16_t unused3 : 4;
    uint16_t type3 : 4;
};

struct EventExtTrigger {
    uint16_t pol : 1;
    uint16_t unused : 7;
    uint16_t id : 4;
    uint16_t type : 4;
};

struct RawEvent {
    uint16_t content : 12;
    uint16_t type : 4;
};

struct EventTime {
    uint16_t time : 12;
    uint16_t type : 4;
    static size_t decode_time_high(const uint16_t *ev, Metavision::timestamp &cur_t) {
        const EventTime *ev_timehigh = reinterpret_cast<const EventTime *>(ev);
        cur_t                        = (cur_t & ~(0b111111111111ull << 12)) | (ev_timehigh->time << 12);
        return 1;
    }
};

enum class TypesEnum : EventTypesUnderlying_t {
    EVT_ADDR_Y   = 0x0,
    EVT_ADDR_X   = 0x2,
    VECT_BASE_X  = 0x3,
    VECT_12      = 0x4,
    VECT_8       = 0x5,
    TIME_LOW     = 0x6,
    CONTINUED_4  = 0x7,
    TIME_HIGH    = 0x8,
    EXT_TRIGGER  = 0xA,
    OTHERS       = 0xE,
    CONTINUED_12 = 0xF
};

class Decoder : public Metavision::I_Decoder {
public:
    Decoder(bool time_shifting_enabled, int height,
            const std::shared_ptr<Metavision::I_EventDecoder<Metavision::EventCD>> &event_cd_decoder =
                std::shared_ptr<Metavision::I_EventDecoder<Metavision::EventCD>>(),
            const std::shared_ptr<Metavision::I_EventDecoder<Metavision::EventExtTrigger>> &event_ext_trigger_decoder =
                std::shared_ptr<Metavision::I_EventDecoder<Metavision::EventExtTrigger>>()) :
        Metavision::I_Decoder(time_shifting_enabled, event_cd_decoder, event_ext_trigger_decoder), height_(height) {}

    virtual bool get_timestamp_shift(Metavision::timestamp &ts_shift) const override {
        ts_shift = timestamp_shift_;
        return base_time_set_;
    }

    virtual Metavision::timestamp get_last_timestamp() const override final {
        return is_time_shifting_enabled() ? last_timestamp<true>() : last_timestamp<false>();
    }

    uint8_t get_raw_event_size_bytes() const override {
        return sizeof(RawEvent);
    }

private:
    template<bool DO_TIMESHIFT>
    Metavision::timestamp last_timestamp() const {
        return DO_TIMESHIFT ? last_timestamp_.time - timestamp_shift_ : last_timestamp_.time;
    }

    virtual void decode_impl(RawData *cur_raw_data, RawData *raw_data_end) override {
        RawEvent *cur_raw_ev       = reinterpret_cast<RawEvent *>(cur_raw_data);
        RawEvent *const raw_ev_end = reinterpret_cast<RawEvent *>(raw_data_end);

        if (!base_time_set_) {
            for (; cur_raw_ev != raw_ev_end; ++cur_raw_ev) {
                if (static_cast<EventTypesUnderlying_t>(TypesEnum::TIME_HIGH) == cur_raw_ev->type) {
                    EventTime *ev_timehigh = reinterpret_cast<EventTime *>(cur_raw_ev);

                    Metavision::timestamp t = ev_timehigh->time;
                    if (t > 0) {
                        --t;
                    }
                    timestamp_shift_                   = t << NumBitsInTimestampLSB;
                    last_timestamp_.bitfield_time.high = t;
                    base_time_set_                     = true;
                    break;
                }
            }
        }

        // We first try to decode incomplete multiword event from previous decode call, if any
        if (raw_events_missing_count_ > 0) {
            // 1- Computes how many raw event from this input need to be copied to get a complete multiword raw event
            // and append them to the incomplete raw event
            const auto raw_events_to_insert_count =
                std::min(raw_events_missing_count_, std::distance(cur_raw_ev, raw_ev_end));
            incomplete_multiword_raw_event_.insert(incomplete_multiword_raw_event_.end(), cur_raw_ev,
                                                   cur_raw_ev + raw_events_to_insert_count);
            cur_raw_ev += raw_events_to_insert_count;
            raw_events_missing_count_ -= raw_events_to_insert_count;

            // 2- If the necessary amount of data is present in the input, decode the now complete multiword event
            if (raw_events_missing_count_ == 0) {
                auto multi_word_raw_ev_begin = incomplete_multiword_raw_event_.data();
                auto multi_word_raw_ev_end =
                    incomplete_multiword_raw_event_.data() + incomplete_multiword_raw_event_.size();
                is_time_shifting_enabled() ?
                    decode_events_buffer<true>(multi_word_raw_ev_begin, multi_word_raw_ev_end) :
                    decode_events_buffer<false>(multi_word_raw_ev_begin, multi_word_raw_ev_end);
                incomplete_multiword_raw_event_.clear();
            } else {
                // Not enough events in the inputs to complete the missing multiword event. All input data were
                // processed without any actual decoding. We can return safely.
                return;
            }
        }

        raw_events_missing_count_ = is_time_shifting_enabled() ? decode_events_buffer<true>(cur_raw_ev, raw_ev_end) :
                                                                 decode_events_buffer<false>(cur_raw_ev, raw_ev_end);
        incomplete_multiword_raw_event_.insert(incomplete_multiword_raw_event_.end(), cur_raw_ev, raw_ev_end);
    }

    template<bool DO_TIMESHIFT>
    uint32_t decode_events_buffer(RawEvent *&cur_raw_ev, RawEvent *const raw_ev_end) {
        auto &cd_forwarder      = cd_event_forwarder();
        auto &trigger_forwarder = trigger_event_forwarder();
        for (; cur_raw_ev != raw_ev_end;) {
            const uint16_t type = cur_raw_ev->type;
            if (type == static_cast<EventTypesUnderlying_t>(TypesEnum::EVT_ADDR_X)) {
                if (is_valid) {
                    EventPosX *ev_posx = reinterpret_cast<EventPosX *>(cur_raw_ev);
                    cd_forwarder.forward(static_cast<unsigned short>(ev_posx->x), state[(int)TypesEnum::EVT_ADDR_Y],
                                         static_cast<short>(ev_posx->pol), last_timestamp<DO_TIMESHIFT>());
                }
                ++cur_raw_ev;

            } else if (type == static_cast<EventTypesUnderlying_t>(TypesEnum::VECT_12)) {
                constexpr uint32_t vect12_size = sizeof(EventVect12_12_8) / sizeof(RawEvent);
                if (cur_raw_ev + vect12_size > raw_ev_end) {
                    // Not enough raw data to decode the vect12_12_8 events. Stop decoding this buffer and return the
                    // amount of data missing to wait for to be able to decode on the next call
                    return static_cast<uint32_t>(std::distance(raw_ev_end, cur_raw_ev + vect12_size));
                }
                if (!is_valid) {
                    cur_raw_ev += vect12_size;
                    continue;
                }

                cd_forwarder.reserve(32);

                EventVect12_12_8 *ev_vect12_12_8 = reinterpret_cast<EventVect12_12_8 *>(cur_raw_ev);

                Mask m;
                m.m.valid1 = ev_vect12_12_8->valid1;
                m.m.valid2 = ev_vect12_12_8->valid2;
                m.m.valid3 = ev_vect12_12_8->valid3;

                uint32_t valid = m.valid;

                uint16_t last_x  = state[(int)TypesEnum::VECT_BASE_X] & NOT_POLARITY_MASK;
                uint16_t nb_bits = 32;
#if defined(__x86_64__) || defined(__aarch64__)
                uint16_t off = 0;
                while (valid) {
                    off = __builtin_ctz(valid);
                    valid &= ~(1 << off);
                    cd_forwarder.forward_unsafe(last_x + off, state[(int)TypesEnum::EVT_ADDR_Y],
                                                (bool)(state[(int)TypesEnum::VECT_BASE_X] & POLARITY_MASK),
                                                last_timestamp<DO_TIMESHIFT>());
                }
#else
                uint16_t end = last_x + nb_bits;
                for (uint16_t i = last_x; i != end; ++i) {
                    if (valid & 0x1) {
                        cd_forwarder.forward_unsafe(i, state[(int)TypesEnum::EVT_ADDR_Y],
                                                    (bool)(state[(int)TypesEnum::VECT_BASE_X] & POLARITY_MASK),
                                                    last_timestamp<DO_TIMESHIFT>());
                    }
                    valid >>= 1;
                }
#endif
                state[(int)TypesEnum::VECT_BASE_X] += nb_bits;
                cur_raw_ev += vect12_size;

            } else if (type == static_cast<EventTypesUnderlying_t>(TypesEnum::TIME_HIGH)) {
                EventTime *ev_timehigh                   = reinterpret_cast<EventTime *>(cur_raw_ev);
                static constexpr uint64_t max_timestamp_ = 1ULL << 11;
                last_timestamp_.bitfield_time.loop +=
                    (bool)(last_timestamp_.bitfield_time.high >= max_timestamp_ + ev_timehigh->time);
                last_timestamp_.bitfield_time.high = ev_timehigh->time;
                ++cur_raw_ev;
            } else if (type == static_cast<EventTypesUnderlying_t>(TypesEnum::EXT_TRIGGER)) {
                EventExtTrigger *ev_exttrigger = reinterpret_cast<EventExtTrigger *>(cur_raw_ev);
                trigger_forwarder.forward(static_cast<short>(ev_exttrigger->pol), last_timestamp<DO_TIMESHIFT>(),
                                          static_cast<short>(ev_exttrigger->id));
                ++cur_raw_ev;
            } else {
                // The objective is to reduce the number of possible cases
                // The content of each type is store into a state because the encoding is stateful
                state[type] = cur_raw_ev->content;
                // Some event outside of the sensor may occur, to limit the number of test the check is done
                // every EVT_ADDR_Y
                is_valid = state[(int)TypesEnum::EVT_ADDR_Y] < height_;

                last_timestamp_.bitfield_time.low = type != static_cast<EventTypesUnderlying_t>(TypesEnum::TIME_LOW) ?
                                                        last_timestamp_.bitfield_time.low :
                                                        state[static_cast<EventTypesUnderlying_t>(TypesEnum::TIME_LOW)];

                ++cur_raw_ev;
            }
        }

        // All raw events have been fully decoded: no data missing to decode a full event
        return 0;
    }

    bool base_time_set_ = false;

    constexpr static int SIZE_EVTYPE                    = 16;
    constexpr static uint16_t NumBitsInTimestampLSB     = 12;
    constexpr static uint16_t NumBitsInHighTimestampLSB = 12;
    constexpr static uint16_t POLARITY_MASK             = 1 << (NumBitsInTimestampLSB - 1);
    constexpr static uint16_t NOT_POLARITY_MASK         = ~(1 << (NumBitsInTimestampLSB - 1));

    uint32_t state[SIZE_EVTYPE];
    bool is_valid = false;
    bool is_td    = false;

    struct bitfield_timestamp {
        uint64_t low : NumBitsInTimestampLSB;
        uint64_t high : NumBitsInHighTimestampLSB;
        uint64_t loop : 64 - NumBitsInHighTimestampLSB - NumBitsInTimestampLSB;
    };

    struct evt3_timestamp {
        union {
            uint64_t time;
            bitfield_timestamp bitfield_time;
        };
    };

    evt3_timestamp last_timestamp_         = {0};
    Metavision::timestamp timestamp_shift_ = 0;
    uint32_t height_                       = 65536;
    std::vector<RawEvent> incomplete_multiword_raw_event_;
    std::ptrdiff_t raw_events_missing_count_{0};
};

} // namespace EVT3
} // namespace Metavision

#endif // RAW_EVT3_DECODER_H
