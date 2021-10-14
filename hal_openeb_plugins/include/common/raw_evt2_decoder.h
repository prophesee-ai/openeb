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

#ifndef RAW_EVT2_DECODER_H
#define RAW_EVT2_DECODER_H

#include <metavision/hal/facilities/i_decoder.h>
#include <metavision/hal/facilities/i_event_decoder.h>
#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/events/event_ext_trigger.h>

#include "raw_evt_base.h"

namespace Metavision {
namespace EVT2 {

constexpr uint8_t EventsTimeStampBits = 6;

// See https://docs.prophesee.ai/stable/data_formats/data_encoding_formats/evt2.html
enum class TypesEnum : EventTypesUnderlying_t {
    CD_OFF        = 0x00, // camera CD event, decrease in illumination (polarity '0')
    CD_ON         = 0x01, // camera CD event, increase in illumination (polarity '1')
    EVT_TIME_HIGH = 0x08, // Timer high bits, also used to synchronize different event flows in the FPGA.
    EXT_TRIGGER   = 0x0A, // External trigger output
    OTHER         = 0x0E, // To be used for extensions in the event types
    CONTINUED     = 0X0F, // Extra data to previous events
};

// Works for both CD_ON and CD_OFF events
struct Event2D {
    unsigned int y : 11;
    unsigned int x : 11;
    unsigned int timestamp : 6;
    unsigned int type : 4;
};

struct EventExtTrigger {
    unsigned int value : 1;
    unsigned int unused2 : 7;
    unsigned int id : 5;
    unsigned int unused1 : 9;
    unsigned int timestamp : 6;
    unsigned int type : 4;
};

class Decoder : public Metavision::I_Decoder {
public:
    struct RawEvent {
        unsigned int trail : 28;
        unsigned int type : 4;
    };
    using Event_Word_Type = uint32_t;

    static constexpr std::uint8_t NumBitsInTimestampLSB = EventsTimeStampBits;
    static constexpr Metavision::timestamp MaxTimestamp = Metavision::timestamp((1 << 28) - 1) << NumBitsInTimestampLSB;
    static constexpr Metavision::timestamp LoopThreshold = 10000;
    static constexpr Metavision::timestamp TimeLoop      = MaxTimestamp + (1 << NumBitsInTimestampLSB);

    Decoder(bool time_shifting_enabled,
            const std::shared_ptr<Metavision::I_EventDecoder<Metavision::EventCD>> &event_cd_decoder =
                std::shared_ptr<Metavision::I_EventDecoder<Metavision::EventCD>>(),
            const std::shared_ptr<Metavision::I_EventDecoder<Metavision::EventExtTrigger>> &event_ext_trigger_decoder =
                std::shared_ptr<Metavision::I_EventDecoder<Metavision::EventExtTrigger>>()) :
        Metavision::I_Decoder(time_shifting_enabled, event_cd_decoder, event_ext_trigger_decoder) {}

    virtual bool get_timestamp_shift(Metavision::timestamp &ts_shift) const override {
        ts_shift = shift_th_;
        return base_time_set_;
    }

    virtual Metavision::timestamp get_last_timestamp() const override {
        return last_timestamp_;
    }

    uint8_t get_raw_event_size_bytes() const override {
        return sizeof(RawEvent);
    }

private:
    bool base_time_set_ = false;

    virtual void decode_impl(RawData *cur_raw_data, RawData *raw_data_end) override {
        RawEvent *cur_raw_ev = reinterpret_cast<RawEvent *>(cur_raw_data);
        RawEvent *raw_ev_end = reinterpret_cast<RawEvent *>(raw_data_end);

        if (!base_time_set_) {
            for (; cur_raw_ev != raw_ev_end; cur_raw_ev++) {
                if (cur_raw_ev->type == static_cast<EventTypesUnderlying_t>(TypesEnum::EVT_TIME_HIGH)) {
                    uint64_t t = cur_raw_ev->trail;
                    t <<= NumBitsInTimestampLSB;
                    base_time_     = t;
                    shift_th_      = is_time_shifting_enabled() ? t : 0;
                    full_shift_    = -shift_th_;
                    base_time_set_ = true;
                    break;
                }
            }
        }

        if (!buffer_has_time_loop(cur_raw_ev, raw_ev_end, base_time_, full_shift_)) {
            // In the general case: if no time shift is to be applied and there is no time loop yet, do not apply
            // any shifting on the new timer high decoded.
            if (full_shift_ == 0) {
                decode_events_buffer<false, false>(cur_raw_ev, raw_ev_end);
            } else {
                decode_events_buffer<false, true>(cur_raw_ev, raw_ev_end);
            }
        } else {
            decode_events_buffer<true, true>(cur_raw_ev, raw_ev_end);
        }
    }

    template<bool UPDATE_LOOP, bool APPLY_TIMESHIFT>
    void decode_events_buffer(RawEvent *&cur_raw_ev, RawEvent *const raw_ev_end) {
        auto &cd_forwarder      = cd_event_forwarder();
        auto &trigger_forwarder = trigger_event_forwarder();
        for (; cur_raw_ev != raw_ev_end; ++cur_raw_ev) {
            const unsigned int type = cur_raw_ev->type;
            if (type == static_cast<EventTypesUnderlying_t>(TypesEnum::EVT_TIME_HIGH)) {
                Metavision::timestamp new_th = Metavision::timestamp(cur_raw_ev->trail) << NumBitsInTimestampLSB;
                if (UPDATE_LOOP) {
                    new_th += full_shift_;
                    if (has_time_loop(new_th, base_time_)) {
                        full_shift_ += TimeLoop;
                        new_th += TimeLoop;
                    }
                    base_time_ = new_th;
                } else {
                    base_time_ = APPLY_TIMESHIFT ? new_th + full_shift_ : new_th;
                }
            } else if (type == static_cast<EventTypesUnderlying_t>(TypesEnum::CD_OFF) ||
                       type == static_cast<EventTypesUnderlying_t>(TypesEnum::CD_ON)) {
                Event2D *ev_cd  = reinterpret_cast<Event2D *>(cur_raw_ev);
                last_timestamp_ = base_time_ + ev_cd->timestamp;
                cd_forwarder.forward(static_cast<unsigned short>(ev_cd->x), static_cast<unsigned short>(ev_cd->y),
                                     ev_cd->type & 1, last_timestamp_);
            } else if (type == static_cast<EventTypesUnderlying_t>(TypesEnum::EXT_TRIGGER)) {
                EventExtTrigger *ev_ext_raw = reinterpret_cast<EventExtTrigger *>(cur_raw_ev);
                last_timestamp_             = base_time_ + ev_ext_raw->timestamp;
                trigger_forwarder.forward(static_cast<short>(ev_ext_raw->value), last_timestamp_,
                                          static_cast<short>(ev_ext_raw->id));
            }
        }
    }

    static bool buffer_has_time_loop(RawEvent *const cur_raw_ev, RawEvent *raw_ev_end,
                                     const Metavision::timestamp base_time_us,
                                     const Metavision::timestamp timeshift_us) {
        for (; raw_ev_end != cur_raw_ev;) {
            --raw_ev_end; // If cur_ev_end == cur_ev, we don't enter so cur_ev_end is always valid
            if (raw_ev_end->type == static_cast<EventTypesUnderlying_t>(TypesEnum::EVT_TIME_HIGH)) {
                const Metavision::timestamp timer_high =
                    (Metavision::timestamp(raw_ev_end->trail) << NumBitsInTimestampLSB) + timeshift_us;
                return has_time_loop(timer_high, base_time_us);
            }
        }
        return false;
    }

    static bool has_time_loop(const Metavision::timestamp current_time_us, const Metavision::timestamp base_time_us) {
        return (current_time_us < base_time_us) && ((base_time_us - current_time_us) >= (MaxTimestamp - LoopThreshold));
    }

    Metavision::timestamp base_time_;      // base time to add non timer high events' ts to
    Metavision::timestamp shift_th_{0};    // first th decoded
    Metavision::timestamp last_timestamp_; // ts of the last event
    Metavision::timestamp full_shift_{
        0}; // includes loop and shift_th in one single variable. Must be signed typed as shift can be negative.
};

} // namespace EVT2
} // namespace Metavision

#endif // RAW_EVT2_DECODER_H
