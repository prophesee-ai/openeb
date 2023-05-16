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

#ifndef METAVISION_HAL_EVT3_DECODER_H
#define METAVISION_HAL_EVT3_DECODER_H

#include <atomic>
#include <algorithm>
#include <mutex>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_erc_counter.h"
#include "metavision/sdk/base/utils/detail/bitinstructions.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/decoders/base/event_base.h"
#include "metavision/hal/facilities/i_geometry.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/decoders/evt3/evt3_event_types.h"
#include "metavision/hal/decoders/evt3/evt3_validator.h"

namespace Metavision {
namespace detail {

template<class Validator>
class EVT3Decoder : public I_EventsStreamDecoder {
public:
    using RawEvent       = Evt3Raw::RawEvent;
    using EventTypesEnum = Evt3EventTypes_4bits;
    using ValidatorType  = Validator;

private:
    // Parameters related to time loop policy
    std::atomic<bool> timestamp_loop_enabled_{true};

    std::mutex is_decoding_mut_;
    Validator validator;

public:
    EVT3Decoder(
        bool time_shifting_enabled, int height, int width,
        const std::shared_ptr<I_EventDecoder<EventCD>> &event_cd_decoder = std::shared_ptr<I_EventDecoder<EventCD>>(),
        const std::shared_ptr<I_EventDecoder<EventExtTrigger>> &event_ext_trigger_decoder =
            std::shared_ptr<I_EventDecoder<EventExtTrigger>>(),
        const std::shared_ptr<I_EventDecoder<EventERCCounter>> &erc_count_event_decoder =
            std::shared_ptr<I_EventDecoder<EventERCCounter>>()) :
        I_EventsStreamDecoder(time_shifting_enabled, event_cd_decoder, event_ext_trigger_decoder,
                              erc_count_event_decoder),
        height_(height),
        validator(height, width) {}

    virtual bool get_timestamp_shift(timestamp &ts_shift) const override {
        ts_shift = timestamp_shift_;
        return timestamp_shift_set_;
    }

    virtual timestamp get_last_timestamp() const override final {
        if (!last_timestamp_set_) {
            return -1;
        }
        return is_time_shifting_enabled() ? last_timestamp<true>() : last_timestamp<false>();
    }

    uint8_t get_raw_event_size_bytes() const override {
        return sizeof(RawEvent);
    }

    virtual size_t add_protocol_violation_callback(const ProtocolViolationCallback_t &cb) override {
        return validator.add_protocol_violation_callback(cb);
    }

    virtual bool remove_protocol_violation_callback(size_t callback_id) override {
        return validator.remove_protocol_violation_callback(callback_id);
    }

private:
    template<bool DO_TIMESHIFT>
    timestamp last_timestamp() const {
        return DO_TIMESHIFT ? last_timestamp_.time - timestamp_shift_ : last_timestamp_.time;
    }

    virtual void decode_impl(const RawData *const cur_raw_data, const RawData *const raw_data_end) override {
        const RawEvent *cur_raw_ev       = reinterpret_cast<const RawEvent *>(cur_raw_data);
        const RawEvent *const raw_ev_end = reinterpret_cast<const RawEvent *>(raw_data_end);

        if (!base_time_set_) {
            for (; cur_raw_ev != raw_ev_end; ++cur_raw_ev) {
                if (static_cast<EventTypesUnderlying_t>(EventTypesEnum::EVT_TIME_HIGH) == cur_raw_ev->type) {
                    const Evt3Raw::Event_Time *ev_timehigh = reinterpret_cast<const Evt3Raw::Event_Time *>(cur_raw_ev);

                    timestamp t = ev_timehigh->time;
                    if (t > 0) {
                        --t;
                    }
                    if (!timestamp_shift_set_) {
                        timestamp_shift_     = t << NumBitsInTimestampLSB;
                        timestamp_shift_set_ = true;
                    }
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
                const RawEvent *multi_word_raw_ev_begin = incomplete_multiword_raw_event_.data();
                const RawEvent *const multi_word_raw_ev_end =
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
    uint32_t decode_events_buffer(const RawEvent *&cur_raw_ev, const RawEvent *const raw_ev_end) {
        auto &cd_forwarder        = cd_event_forwarder();
        auto &trigger_forwarder   = trigger_event_forwarder();
        auto &erc_count_forwarder = erc_count_event_forwarder();
        for (; cur_raw_ev != raw_ev_end;) {
            const uint16_t type = cur_raw_ev->type;
            if (type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::EVT_ADDR_X)) {
                if (is_valid) {
                    const Evt3Raw::Event_PosX *ev_posx = reinterpret_cast<const Evt3Raw::Event_PosX *>(cur_raw_ev);
                    if (validator.validate_event_cd(cur_raw_ev)) {
                        cd_forwarder.forward(static_cast<unsigned short>(ev_posx->x),
                                             state[(int)EventTypesEnum::EVT_ADDR_Y], static_cast<short>(ev_posx->pol),
                                             last_timestamp<DO_TIMESHIFT>());
                    }
                }
                ++cur_raw_ev;

            } else if (type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::VECT_12)) {
                constexpr uint32_t vect12_size = sizeof(Evt3Raw::Event_Vect12_12_8) / sizeof(RawEvent);
                if (cur_raw_ev + vect12_size > raw_ev_end) {
                    // Not enough raw data to decode the vect12_12_8 events. Stop decoding this buffer and return the
                    // amount of data missing to wait for to be able to decode on the next call
                    return std::distance(raw_ev_end, cur_raw_ev + vect12_size);
                }
                if (!is_valid) {
                    cur_raw_ev += vect12_size;
                    continue;
                }

                const uint16_t nb_bits = 32;
                int next_offset;
                if (validator.validate_vect_12_12_8_pattern(
                        cur_raw_ev, state[(int)EventTypesEnum::VECT_BASE_X] & NOT_POLARITY_MASK, next_offset)) {
                    cd_forwarder.reserve(32);

                    const Evt3Raw::Event_Vect12_12_8 *ev_vect12_12_8 =
                        reinterpret_cast<const Evt3Raw::Event_Vect12_12_8 *>(cur_raw_ev);

                    Evt3Raw::Mask m;
                    m.m.valid1 = ev_vect12_12_8->valid1;
                    m.m.valid2 = ev_vect12_12_8->valid2;
                    m.m.valid3 = ev_vect12_12_8->valid3;

                    uint32_t valid = m.valid;

                    uint16_t last_x = state[(int)EventTypesEnum::VECT_BASE_X] & NOT_POLARITY_MASK;
                    uint16_t off    = 0;
                    while (valid) {
                        off = ctz_not_zero(valid);
                        valid &= ~(1 << off);
                        cd_forwarder.forward_unsafe(last_x + off, state[(int)EventTypesEnum::EVT_ADDR_Y],
                                                    (bool)(state[(int)EventTypesEnum::VECT_BASE_X] & POLARITY_MASK),
                                                    last_timestamp<DO_TIMESHIFT>());
                    }
                }
                if (validator.has_valid_vect_base()) {
                    state[(int)EventTypesEnum::VECT_BASE_X] += nb_bits;
                }
                cur_raw_ev += next_offset;
            } else if (type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::EVT_TIME_HIGH)) {
                const Evt3Raw::Event_Time *ev_timehigh    = reinterpret_cast<const Evt3Raw::Event_Time *>(cur_raw_ev);
                static constexpr timestamp max_timestamp_ = 1ULL << 11;

                validator.validate_time_high(last_timestamp_.bitfield_time.high, ev_timehigh->time);

                last_timestamp_.bitfield_time.loop +=
                    (bool)(last_timestamp_.bitfield_time.high >= max_timestamp_ + ev_timehigh->time);
                last_timestamp_.bitfield_time.low =
                    (last_timestamp_.bitfield_time.high == ev_timehigh->time ?
                         last_timestamp_.bitfield_time.low :
                         0); // avoid momentary time discrepancies when decoding event per events. Time low comes
                             // right after to correct the value (note that the timestamp here is not good if we don't
                             // do that either)
                last_timestamp_.bitfield_time.high = ev_timehigh->time;

                ++cur_raw_ev;
            } else if (type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::EXT_TRIGGER)) {
                if (validator.validate_ext_trigger(cur_raw_ev)) {
                    const Evt3Raw::Event_ExtTrigger *ev_exttrigger =
                        reinterpret_cast<const Evt3Raw::Event_ExtTrigger *>(cur_raw_ev);
                    trigger_forwarder.forward(static_cast<short>(ev_exttrigger->pol), last_timestamp<DO_TIMESHIFT>(),
                                              static_cast<short>(ev_exttrigger->id));
                }
                ++cur_raw_ev;
            } else if (type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::OTHERS)) {
                const uint16_t master_type = cur_raw_ev->content;
                bool is_out_count_evt      = false;

                switch (master_type) {
                case static_cast<uint16_t>(Evt3MasterEventTypes::MASTER_RATE_CONTROL_CD_EVENT_COUNT):
                    is_out_count_evt = true;
                    [[fallthrough]];
                case static_cast<uint16_t>(Evt3MasterEventTypes::MASTER_IN_CD_EVENT_COUNT): {
                    constexpr uint32_t evt_size = 1 + sizeof(Evt3Raw::Event_Continue12_12_4) / sizeof(RawEvent);
                    if (cur_raw_ev + evt_size > raw_ev_end) {
                        // Not enough raw data to decode the continue events. Stop decoding this buffer and return the
                        // amount of data missing to wait for to be able to decode on the next call
                        return std::distance(raw_ev_end, cur_raw_ev + evt_size);
                    }
                    ++cur_raw_ev;
                    int next_offset;
                    if (validator.validate_continue_12_12_4_pattern(cur_raw_ev, next_offset)) {
                        const Evt3Raw::Event_Continue12_12_4 *data =
                            reinterpret_cast<const Evt3Raw::Event_Continue12_12_4 *>(cur_raw_ev);
                        erc_count_forwarder.forward(last_timestamp<DO_TIMESHIFT>(),
                                                    Evt3Raw::Event_Continue12_12_4::decode(*data), is_out_count_evt);
                    }
                    cur_raw_ev += next_offset;
                    break;
                }
                default:
                    // Unhandled sys event
                    ++cur_raw_ev;
                    break;
                }
            } else {
                // The objective is to reduce the number of possible cases
                // The content of each type is store into a state because the encoding is stateful
                state[type] = cur_raw_ev->content;
                // Here the type of event is saved (CD vs EM) to know when a EVT_ADDR_X or VECT_BASE_X arrives
                // if the event is a CD or EM
                is_cd = type >= 2 ? is_cd : !(bool)type;
                // Some event outside of the sensor may occur, to limit the number of test the check is done
                // every EVT_ADDR_Y
                is_valid = is_cd && state[(int)EventTypesEnum::EVT_ADDR_Y] < height_;

                last_timestamp_.bitfield_time.low =
                    type != static_cast<EventTypesUnderlying_t>(EventTypesEnum::EVT_TIME_LOW) ?
                        last_timestamp_.bitfield_time.low :
                        state[static_cast<EventTypesUnderlying_t>(EventTypesEnum::EVT_TIME_LOW)];
                last_timestamp_set_ = true;

                validator.state_update(cur_raw_ev);

                ++cur_raw_ev;
            }
        }

        // All raw events have been fully decoded: no data missing to decode a full event
        return 0;
    }

    /// @brief Resets the decoder last timestamp
    /// @param t Timestamp to reset the decoder to
    /// @return True if the reset operation could complete, false otherwise.
    /// @note It is expected after this call has succeeded, that @ref get_last_timestamp returns @p timestamp
    /// @warning If time shifting is enabled, the @p timestamp must be in the shifted time reference
    /// @warning After this function has been called with @p timestamp >= 0, it is assumed that the next buffers of
    ///          raw data to decode contain events with timestamps >= @p timestamp
    /// @warning After this function has been called, the internal state of the decoder is reset.
    ///          The next buffers of raw data to decode should contain events that fully restore the internal state
    ///          of the decoder to a valid state (such as 'EVT_ADDR_Y', 'VECT_BASE_X', etc.) before any other
    ///          type of events can be decoded
    bool reset_timestamp_impl(const timestamp &t) override {
        if (is_time_shifting_enabled() && !timestamp_shift_set_) {
            return false;
        }

        // reset internal state
        std::fill(state, state + SIZE_EVTYPE, 0);
        is_valid = false;
        is_cd    = false;
        incomplete_multiword_raw_event_.clear();
        raw_events_missing_count_ = 0;

        if (t >= 0) {
            static constexpr timestamp max_timestamp = 0xFFFFFF;
            const auto shifted_time                  = t + (is_time_shifting_enabled() ? timestamp_shift_ : 0);
            last_timestamp_.bitfield_time.high       = (shifted_time & 0xFFF000) >> NumBitsInTimestampLSB;
            last_timestamp_.bitfield_time.low        = shifted_time & 0xFFF;
            last_timestamp_.bitfield_time.loop       = shifted_time / max_timestamp;
            base_time_set_                           = true;
            last_timestamp_set_                      = true;
            return true;
        } else {
            base_time_set_      = false;
            last_timestamp_set_ = false;
            return true;
        }
        return false;
    }

    bool reset_timestamp_shift_impl(const timestamp &shift) override {
        if (shift >= 0 && is_time_shifting_enabled()) {
            timestamp_shift_     = shift;
            timestamp_shift_set_ = true;
            return true;
        }
        return false;
    }

    constexpr static int SIZE_EVTYPE                    = 16;
    constexpr static uint16_t NumBitsInTimestampLSB     = 12;
    constexpr static uint16_t NumBitsInHighTimestampLSB = 12;
    constexpr static uint16_t POLARITY_MASK             = 1 << (NumBitsInTimestampLSB - 1);
    constexpr static uint16_t NOT_POLARITY_MASK         = ~(1 << (NumBitsInTimestampLSB - 1));
    uint32_t state[SIZE_EVTYPE]                         = {0};
    bool is_valid                                       = false;
    bool is_cd                                          = false;
    struct bitfield_timestamp {
        uint64_t low : NumBitsInTimestampLSB;
        uint64_t high : NumBitsInHighTimestampLSB;
        uint64_t loop : 64 - NumBitsInHighTimestampLSB - NumBitsInTimestampLSB;
    };
    struct evt3_timestamp {
        union {
            bitfield_timestamp bitfield_time;
            uint64_t time;
        };
    };
    evt3_timestamp last_timestamp_ = {0};

    bool timestamp_shift_set_  = false;
    bool base_time_set_        = false;
    bool last_timestamp_set_   = false;
    timestamp timestamp_shift_ = 0;
    uint32_t height_           = 65536;
    std::vector<RawEvent> incomplete_multiword_raw_event_;
    std::ptrdiff_t raw_events_missing_count_{0};
};

} // namespace detail

using EVT3Decoder       = detail::EVT3Decoder<decoder::evt3::BasicCheckValidator>;
using UnsafeEVT3Decoder = detail::EVT3Decoder<decoder::evt3::NullCheckValidator>;
using RobustEVT3Decoder = detail::EVT3Decoder<decoder::evt3::GrammarValidator>;

namespace {
void throw_on_non_monotonic_time_high(const DecoderProtocolViolation &protocol_violation_type) {
    std::ostringstream oss;
    oss << "Evt3 protocol violation detected : " << protocol_violation_type;

    switch (protocol_violation_type) {
    case DecoderProtocolViolation::NonMonotonicTimeHigh:
        throw(HalException(protocol_violation_type, oss.str()));
    default:
        break;
    }
};
} // namespace

inline std::unique_ptr<I_EventsStreamDecoder> make_evt3_decoder(
    bool time_shifting_enabled, int height, int width,
    const std::shared_ptr<I_EventDecoder<EventCD>> &event_cd_decoder = std::shared_ptr<I_EventDecoder<EventCD>>(),
    const std::shared_ptr<I_EventDecoder<EventExtTrigger>> &event_ext_trigger_decoder =
        std::shared_ptr<I_EventDecoder<EventExtTrigger>>(),
    const std::shared_ptr<I_EventDecoder<EventERCCounter>> &erc_count_event_decoder =
        std::shared_ptr<I_EventDecoder<EventERCCounter>>()) {
    std::unique_ptr<I_EventsStreamDecoder> decoder = std::make_unique<EVT3Decoder>(
        time_shifting_enabled, height, width, event_cd_decoder, event_ext_trigger_decoder, erc_count_event_decoder);

    if (std::getenv("MV_FLAGS_EVT3_THROW_ON_NON_MONOTONIC_TIME_HIGH") || std::getenv("MV_FLAGS_EVT3_ROBUST_DECODER")) {
        MV_HAL_LOG_INFO() << "Using EVT3 Robust decoder.";
        decoder = std::make_unique<RobustEVT3Decoder>(time_shifting_enabled, height, width, event_cd_decoder,
                                                      event_ext_trigger_decoder, erc_count_event_decoder);

    } else if (std::getenv("MV_FLAGS_EVT3_UNSAFE_DECODER")) {
        MV_HAL_LOG_INFO() << "Using EVT3 Unsafe decoder.";
        decoder = std::make_unique<UnsafeEVT3Decoder>(time_shifting_enabled, height, width, event_cd_decoder,
                                                      event_ext_trigger_decoder, erc_count_event_decoder);
    }

    if (std::getenv("MV_FLAGS_EVT3_THROW_ON_NON_MONOTONIC_TIME_HIGH")) {
        MV_HAL_LOG_INFO() << "Decoder will raise exception upon EVT3 Non Monotonic Time High violation.";
        decoder->add_protocol_violation_callback(throw_on_non_monotonic_time_high);
    }

    return decoder;
}

} // namespace Metavision

#endif // METAVISION_HAL_EVT3_DECODER_H
