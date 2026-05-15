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

#ifndef METAVISION_HAL_EVT4_DECODER_H
#define METAVISION_HAL_EVT4_DECODER_H

#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/sdk/base/utils/detail/bitinstructions.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/decoders/evt4/evt4_event_types.h"
#include "metavision/hal/decoders/evt4/evt4_validator.h"

#include <cstdint>
#include <cstdlib>
#include <optional>

namespace Metavision {

namespace detail {

template<class Validator>
class EVT4Decoder : public I_EventsStreamDecoder {
public:
    using RawEvent        = Evt4Raw::RawEvent;
    using EventTypesEnum  = EVT4EventTypes;
    using Event_Word_Type = std::uint32_t;

    static constexpr std::uint8_t NumBitsInTimestampLSB{6};
    static constexpr std::uint8_t NumBitsInTimestampMSB{28};
    static constexpr timestamp MaxTimestamp{((static_cast<timestamp>(1) << NumBitsInTimestampMSB) - 1)
                                            << NumBitsInTimestampLSB};
    static constexpr timestamp LoopThreshold{(MaxTimestamp >> 1) + 1};
    static constexpr timestamp TimeLoop{MaxTimestamp + (1 << NumBitsInTimestampLSB)};
    static constexpr std::uint32_t MaxWidth{1 << 11};
    static constexpr std::uint32_t MaxHeight{1 << 11};

    EVT4Decoder(
        bool time_shifting_enabled, const std::optional<std::uint32_t> &width = std::nullopt,
        const std::optional<std::uint32_t> &height                       = std::nullopt,
        const std::shared_ptr<I_EventDecoder<EventCD>> &event_cd_decoder = std::shared_ptr<I_EventDecoder<EventCD>>(),
        const std::shared_ptr<I_EventDecoder<EventExtTrigger>> &event_ext_trigger_decoder =
            std::shared_ptr<I_EventDecoder<EventExtTrigger>>(),
        const std::shared_ptr<I_EventDecoder<EventERCCounter>> &erc_count_event_decoder =
            std::shared_ptr<I_EventDecoder<EventERCCounter>>()) :
        I_EventsStreamDecoder(time_shifting_enabled, event_cd_decoder, event_ext_trigger_decoder,
                              erc_count_event_decoder),
        validator_(width.value_or(MaxWidth), height.value_or(MaxHeight)) {
        ev_other_.subtype = static_cast<std::uint16_t>(EVT4EventSubTypes::UNUSED);
    }

    virtual bool get_timestamp_shift(timestamp &ts_shift) const override {
        ts_shift = shift_th_;
        return shift_set_;
    }

    virtual timestamp get_last_timestamp() const override {
        return last_timestamp_;
    }

    std::uint8_t get_raw_event_size_bytes() const override {
        return sizeof(RawEvent);
    }

private:
    virtual void decode_impl(const RawData *const cur_raw_data, const RawData *const raw_data_end) override {
        const RawEvent *cur_raw_ev = reinterpret_cast<const RawEvent *>(cur_raw_data);
        const RawEvent *raw_ev_end = reinterpret_cast<const RawEvent *>(raw_data_end);

        if (!base_time_set_) {
            for (; !base_time_set_ && cur_raw_ev < raw_ev_end; ++cur_raw_ev) {
                const RawEvent *ev = reinterpret_cast<const RawEvent *>(cur_raw_ev);
                switch (ev->type) {
                case static_cast<EventTypesUnderlying_t>(EventTypesEnum::CD_VEC_OFF):
                case static_cast<EventTypesUnderlying_t>(EventTypesEnum::CD_VEC_ON):
                    cur_raw_ev++;
                    break;
                case static_cast<EventTypesUnderlying_t>(EventTypesEnum::EVT_TIME_HIGH): {
                    timestamp t = static_cast<timestamp>(ev->trail) << NumBitsInTimestampLSB;
                    if (!shift_set_) {
                        shift_th_   = is_time_shifting_enabled() ? t : 0;
                        full_shift_ = -shift_th_;
                        shift_set_  = true;
                    }
                    base_time_      = t + full_shift_;
                    base_time_set_  = true;
                    last_timestamp_ = base_time_;
                } break;
                }
            }
        }

        decode_events_buffer(cur_raw_ev, raw_ev_end);
    }

    inline void decode_event_vector(DecodedEventForwarder<EventCD> &cd_forwarder, const Evt4Raw::EVT4EventCD *ev_cd,
                                    uint32_t pol, const uint32_t *vect_data) {
        if (!validator_.validate_event_cd_vec(ev_cd, vect_data)) {
            return;
        }
        std::uint32_t valid   = *vect_data;
        const std::uint32_t x = ev_cd->x;
        const std::uint32_t y = ev_cd->y;
        while (valid) {
            auto off = ctz_not_zero(valid);
            valid &= valid - 1; // Reset LSB set bit to zero
            cd_forwarder.forward(x + off, y, pol, last_timestamp_);
        }
    }

    void decode_events_buffer(const RawEvent *&cur_raw_ev, const RawEvent *const raw_ev_end) {
        auto &cd_forwarder        = cd_event_forwarder();
        auto &trigger_forwarder   = trigger_event_forwarder();
        auto &erc_count_forwarder = erc_count_event_forwarder();
        if (cd_vec_open_ && cur_raw_ev < raw_ev_end) {
            decode_event_vector(cd_forwarder, &ev_cd_, ev_cd_.type & 0x1,
                                reinterpret_cast<const std::uint32_t *>(cur_raw_ev));
            ++cur_raw_ev;
            cd_vec_open_ = false;
        }
        for (; cur_raw_ev < raw_ev_end; ++cur_raw_ev) {
            const RawEvent *ev        = reinterpret_cast<const RawEvent *>(cur_raw_ev);
            const std::uint32_t &type = ev->type;
            if (type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::CD_OFF) ||
                type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::CD_ON)) { // CD
                const auto *ev_cd = reinterpret_cast<const Evt4Raw::EVT4EventCD *>(ev);
                if (!validator_.validate_event_cd(ev_cd)) {
                    continue;
                }
                last_timestamp_ = base_time_ + ev_cd->timestamp;
                cd_forwarder.forward(static_cast<std::uint16_t>(ev_cd->x), static_cast<std::uint16_t>(ev_cd->y),
                                     type & 1, last_timestamp_);
            } else if (type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::CD_VEC_OFF) ||
                       type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::CD_VEC_ON)) { // CD Vector
                const auto *ev_cd = reinterpret_cast<const Evt4Raw::EVT4EventCD *>(ev);
                last_timestamp_   = base_time_ + ev_cd->timestamp;
                if (++cur_raw_ev >= raw_ev_end) {
                    cd_vec_open_ = true;
                    ev_cd_       = *ev_cd;
                    break;
                }
                decode_event_vector(cd_forwarder, ev_cd, type & 0x1,
                                    reinterpret_cast<const std::uint32_t *>(cur_raw_ev));
            } else if (type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::EVT_TIME_HIGH)) {
                timestamp new_th          = static_cast<timestamp>(ev->trail) << NumBitsInTimestampLSB;
                const auto last_base_time = base_time_;
                auto full_shift           = full_shift_;
                new_th += full_shift;
                if (base_time_ >= new_th + LoopThreshold) {
                    full_shift += TimeLoop;
                    new_th += TimeLoop;
                }

                if (!validator_.validate_time_high(base_time_, new_th)) {
                    continue;
                }

                base_time_  = new_th;
                full_shift_ = full_shift;

                // Avoid momentary time discrepancies when decoding event per events, time low comes
                // right after (in an event of another type) to correct the value
                if (base_time_ != last_base_time) {
                    last_timestamp_ = base_time_;
                }
            } else if (type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::EXT_TRIGGER)) {
                if (!validator_.validate_ext_trigger(ev)) {
                    continue;
                }
                const auto *ev_ext_raw = reinterpret_cast<const Evt4Raw::EVT4EventExtTrigger *>(ev);
                last_timestamp_        = base_time_ + ev_ext_raw->timestamp;
                trigger_forwarder.forward(static_cast<std::uint16_t>(ev_ext_raw->value), last_timestamp_,
                                          static_cast<std::uint16_t>(ev_ext_raw->id));
            } else if (type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::OTHER)) {
                if (!validator_.validate_event_other(ev)) {
                    continue;
                }
                ev_other_       = *reinterpret_cast<const Evt4Raw::EVT4EventMonitor *>(ev);
                last_timestamp_ = base_time_ + ev_other_.timestamp;
            } else if (type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::CONTINUED)) {
                if (!validator_.validate_event_continued(ev)) {
                    continue;
                }
                const std::uint16_t &subtype = ev_other_.subtype;
                if (subtype == static_cast<std::uint16_t>(EVT4EventSubTypes::MASTER_IN_CD_EVENT_COUNT) ||
                    subtype == static_cast<std::uint16_t>(EVT4EventSubTypes::MASTER_RATE_CONTROL_CD_EVENT_COUNT)) {
                    const auto *evssf = reinterpret_cast<const Evt4Raw::EVT4EventMonitorMasterInCdEventCount *>(ev);
                    erc_count_forwarder.forward(
                        last_timestamp_, evssf->count,
                        subtype == static_cast<std::uint16_t>(EVT4EventSubTypes::MASTER_RATE_CONTROL_CD_EVENT_COUNT));
                }
                ev_other_.subtype = static_cast<std::uint16_t>(EVT4EventSubTypes::UNUSED);
            }
        }
    }

    /// @brief Resets the decoder last timestamp
    /// @param t Timestamp to reset the decoder to
    /// @return True if the reset operation could complete, false otherwise.
    /// @note It is expected after this call has succeeded, that @ref get_last_timestamp returns @p timestamp
    /// @warning If time shifting is enabled, the @p timestamp must be in the shifted time reference
    /// @warning After this function has been called with @p timestamp >= 0, it is assumed that the next buffers of
    ///          raw data to decode contain events with timestamps >= @p timestamp
    bool reset_last_timestamp_impl(const timestamp &t) override {
        if (is_time_shifting_enabled() && !shift_set_) {
            return false;
        }
        cd_vec_open_ = false;
        if (t >= 0) {
            constexpr timestamp min_timer_high_val = (1 << NumBitsInTimestampLSB);
            base_time_                             = min_timer_high_val * (t / min_timer_high_val);
            last_timestamp_                        = base_time_ + t % min_timer_high_val;
            full_shift_                            = -shift_th_ + TimeLoop * (base_time_ / TimeLoop);
            base_time_set_                         = true;
            return true;
        } else {
            base_time_set_ = false;
            return true;
        }
        return false;
    }

    bool reset_timestamp_shift_impl(const timestamp &shift) override {
        if (shift >= 0 && is_time_shifting_enabled()) {
            shift_th_   = shift;
            full_shift_ = -shift_th_;
            shift_set_  = true;
            return true;
        }
        return false;
    }

    bool base_time_set_{false};
    timestamp base_time_{0};       // Base time to add non timer high events' time stamp low to
    timestamp shift_th_{0};        // First time high decoded
    timestamp last_timestamp_{-1}; // The timestamp of the last event
    timestamp full_shift_{0};      // Includes loop and shift_th in one single variable
    bool shift_set_{false};
    Evt4Raw::EVT4EventCD ev_cd_;
    bool cd_vec_open_{false};
    Evt4Raw::EVT4EventMonitor ev_other_{};
    Validator validator_;
};

} // namespace detail

using EVT4Decoder       = detail::EVT4Decoder<decoder::evt4::NotifyValidator>;
using UnsafeEVT4Decoder = detail::EVT4Decoder<decoder::evt4::NullCheckValidator>;
using RobustEVT4Decoder = detail::EVT4Decoder<decoder::evt4::RobustValidator>;

inline std::unique_ptr<I_EventsStreamDecoder> make_evt4_decoder(
    bool time_shifting_enabled, const std::optional<std::uint32_t> &width = std::nullopt,
    const std::optional<std::uint32_t> &height                       = std::nullopt,
    const std::shared_ptr<I_EventDecoder<EventCD>> &event_cd_decoder = std::shared_ptr<I_EventDecoder<EventCD>>(),
    const std::shared_ptr<I_EventDecoder<EventExtTrigger>> &event_ext_trigger_decoder =
        std::shared_ptr<I_EventDecoder<EventExtTrigger>>(),
    const std::shared_ptr<I_EventDecoder<EventERCCounter>> &erc_count_event_decoder =
        std::shared_ptr<I_EventDecoder<EventERCCounter>>()) {
    std::unique_ptr<I_EventsStreamDecoder> decoder;

    if (std::getenv("MV_FLAGS_EVT4_ROBUST_DECODER")) {
        MV_HAL_LOG_INFO() << "Using EVT4 Robust decoder.";
        decoder = std::make_unique<RobustEVT4Decoder>(time_shifting_enabled, width, height, event_cd_decoder,
                                                      event_ext_trigger_decoder, erc_count_event_decoder);

    } else if (std::getenv("MV_FLAGS_EVT4_UNSAFE_DECODER")) {
        MV_HAL_LOG_INFO() << "Using EVT4 Unsafe decoder.";
        decoder = std::make_unique<UnsafeEVT4Decoder>(time_shifting_enabled, width, height, event_cd_decoder,
                                                      event_ext_trigger_decoder, erc_count_event_decoder);
    } else {
        decoder = std::make_unique<EVT4Decoder>(time_shifting_enabled, width, height, event_cd_decoder,
                                                event_ext_trigger_decoder, erc_count_event_decoder);
    }

    return decoder;
}

} // namespace Metavision

#endif // METAVISION_HAL_EVT4_DECODER_H
