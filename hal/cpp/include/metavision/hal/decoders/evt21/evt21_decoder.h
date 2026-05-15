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

#ifndef METAVISION_HAL_EVT21_DECODER_H
#define METAVISION_HAL_EVT21_DECODER_H

#include <cstdint>
#include <set>
#include <variant>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_cd_vector.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/sdk/base/events/event_erc_counter.h"
#include "metavision/sdk/base/events/event_monitoring.h"
#include "metavision/sdk/base/utils/detail/bitinstructions.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/decoders/base/event_base.h"
#include "metavision/hal/decoders/evt21/evt21_event_types.h"
#include "metavision/hal/utils/detail/type_check.h"

namespace Metavision {

template<typename RawEvent, typename Event_TIME_HIGH, typename Event_2D, typename Event_EXT_TRIGGER,
         typename Event_OTHERS, typename OutputCDType = EventCD>
class EVT21GenericDecoder : public I_EventsStreamDecoder {
public:
    using EventTypesEnum   = Evt21EventTypes_4bits;
    using EventCDForwarder = I_EventsStreamDecoder::DecodedEventForwarder<OutputCDType>;
    using OutputCDTypes    = std::variant<EventCD, EventCDVector>;

    EVT21GenericDecoder(bool time_shifting_enabled,
                        const std::shared_ptr<I_EventDecoder<OutputCDType>> &event_cd_decoder =
                            std::shared_ptr<I_EventDecoder<OutputCDType>>(),
                        const std::shared_ptr<I_EventDecoder<EventExtTrigger>> &event_ext_trigger_decoder =
                            std::shared_ptr<I_EventDecoder<EventExtTrigger>>(),
                        const std::shared_ptr<I_EventDecoder<EventERCCounter>> &erc_count_event_decoder =
                            std::shared_ptr<I_EventDecoder<EventERCCounter>>(),
                        const std::shared_ptr<I_EventDecoder<EventMonitoring>> &event_monitoring_decoder =
                            std::shared_ptr<I_EventDecoder<EventMonitoring>>(),
                        const std::set<uint16_t> &monitoring_id_blacklist = std::set<uint16_t>()) :
        I_EventsStreamDecoder(time_shifting_enabled, event_cd_decoder, event_ext_trigger_decoder,
                              erc_count_event_decoder),
        event_monitoring_decoder_(event_monitoring_decoder),
        monitoring_id_blacklist_(monitoring_id_blacklist) {
        static_assert(Metavision::detail::is_in_type_list_v<OutputCDType, OutputCDTypes>,
                      "Error, cannot construct EVT21GenericDecoder with specified OutputCDType... Supported types are: "
                      "{EventCD, EventCDVector}.");

        if (event_monitoring_decoder_) {
            monitoring_event_forwarder_.reset(
                new DecodedEventForwarder<EventMonitoring, 1>(event_monitoring_decoder_.get()));
        }

    }

    virtual timestamp get_last_timestamp() const override final {
        if (!last_timestamp_set_) {
            return -1;
        }
        return is_time_shifting_enabled() ? last_timestamp<true>() : last_timestamp<false>();
    }

    virtual bool get_timestamp_shift(Metavision::timestamp &timestamp_shift) const override {
        if (time_shifting_set_) {
            timestamp_shift = timestamp_shift_;
        }
        return time_shifting_set_;
    }

    virtual uint8_t get_raw_event_size_bytes() const override {
        return sizeof(RawEvent);
    }

private:
    template<bool DO_TIMESHIFT>
    timestamp last_timestamp() const {
        if constexpr (DO_TIMESHIFT) {
            return last_timestamp_ - timestamp_shift_;
        } else {
            return last_timestamp_;
        }
    }

    void decode_impl(const RawData *const cur_raw_data, const RawData *const raw_data_end) override {
        const RawEvent *cur_raw_ev = reinterpret_cast<const RawEvent *>(cur_raw_data);
        const RawEvent *raw_ev_end = reinterpret_cast<const RawEvent *>(raw_data_end);

        if (base_time_set_ == false) {
            for (; cur_raw_ev != raw_ev_end; cur_raw_ev++) {
                if (static_cast<EventTypesUnderlying_t>(Evt21EventTypes_4bits::EVT_TIME_HIGH) == cur_raw_ev->type) {
                    const Event_TIME_HIGH *ev_timehigh = reinterpret_cast<const Event_TIME_HIGH *>(cur_raw_ev);

                    // Prevent integer convertion from happening when bits 34-32 are not set
                    timestamp t = static_cast<uint64_t>(ev_timehigh->ts) << 6;
                    set_last_high_timestamp(t);
                    if (!time_shifting_set_ && is_time_shifting_enabled()) {
                        timestamp_shift_   = t;
                        time_shifting_set_ = true;
                    }

                    base_time_set_ = true;
                    break;
                }
            }
            if (cur_raw_ev == raw_ev_end) {
                // There was no TIME_HIGH in the buffer, let's drop it since we don't have a base time to work with...
                return;
            }
        }
        is_time_shifting_enabled() ? decode_events_buffer<true>(cur_raw_ev, raw_ev_end) :
                                     decode_events_buffer<false>(cur_raw_ev, raw_ev_end);
    }

    template<bool DO_TIMESHIFT>
    void decode_events_buffer(const RawEvent *&cur_raw_ev, const RawEvent *const raw_ev_end) {
        auto &cd_forwarder        = cd_event_forwarder<OutputCDType>();
        auto &trigger_forwarder   = trigger_event_forwarder();
        auto &erc_count_forwarder = erc_count_event_forwarder();
        auto &monitoring_forwarder = *monitoring_event_forwarder_;

        const RawEvent *&cur_ev = cur_raw_ev;
        // We should stop on equality but we test difference here to be sure we don't overflow
        for (; cur_ev < raw_ev_end;) {
            const RawEvent *ev      = reinterpret_cast<const RawEvent *>(cur_ev);
            const unsigned int type = ev->type;

            if (type == static_cast<EventTypesUnderlying_t>(Evt21EventTypes_4bits::EVT_POS) ||
                type == static_cast<EventTypesUnderlying_t>(Evt21EventTypes_4bits::EVT_NEG)) {
                const Event_2D *ev_td = reinterpret_cast<const Event_2D *>(cur_ev);
                uint16_t base_x       = ev_td->x;
                uint16_t y            = ev_td->y;

                last_timestamp_     = (last_timestamp_ & ~((1ULL << 6) - 1)) + ev_td->ts;
                last_timestamp_set_ = true;

                uint32_t vector_mask     = ev_td->valid;
                const RawEvent *ev = reinterpret_cast<const RawEvent *>(cur_ev);
                bool polarity            = ev->type == static_cast<EventTypesUnderlying_t>(Evt21EventTypes_4bits::EVT_POS);

                if constexpr (std::is_same_v<OutputCDType, EventCD>) {
                    uint16_t offset = 0;
                    while (vector_mask) {
                        offset = ctz_not_zero(vector_mask);
                        vector_mask &= ~(1 << offset);
                        cd_forwarder.forward(base_x + offset, y, static_cast<short>(polarity), last_timestamp<DO_TIMESHIFT>());
                    }
                }

                if constexpr (std::is_same_v<OutputCDType, EventCDVector>) {
                    cd_forwarder.forward(base_x, y, polarity, vector_mask, last_timestamp<DO_TIMESHIFT>());
                }

                ++cur_ev;
            } else if (type == static_cast<EventTypesUnderlying_t>(Evt21EventTypes_4bits::EVT_TIME_HIGH)) {
                const Event_TIME_HIGH *ev_timehigh = reinterpret_cast<const Event_TIME_HIGH *>(cur_ev);
                // Prevent integer convertion from happening when bits 34-32 are not set
                set_last_high_timestamp(static_cast<uint64_t>(ev_timehigh->ts) << 6);
                ++cur_ev;
            } else if (type == static_cast<EventTypesUnderlying_t>(Evt21EventTypes_4bits::EXT_TRIGGER)) {
                const Event_EXT_TRIGGER *ev_exttrigger = reinterpret_cast<const Event_EXT_TRIGGER *>(cur_ev);
                last_timestamp_                        = (last_timestamp_ & ~((1ULL << 6) - 1)) + ev_exttrigger->ts;
                last_timestamp_set_                    = true;
                trigger_forwarder.forward(static_cast<short>(ev_exttrigger->p), last_timestamp<DO_TIMESHIFT>(),
                                          static_cast<short>(ev_exttrigger->id));
                ++cur_ev;
            } else if (type == static_cast<EventTypesUnderlying_t>(Evt21EventTypes_4bits::OTHERS)) {
                const Event_OTHERS *ev_other = reinterpret_cast<const Event_OTHERS *>(cur_ev);
                if (monitoring_id_blacklist_.count(ev_other->subtype) == 0) {
                    const Evt21EventMasterEventTypes subtype =
                        static_cast<Evt21EventMasterEventTypes>(ev_other->subtype);

                    last_timestamp_     = (last_timestamp_ & ~((1ULL << 6) - 1)) + ev_other->ts;
                    last_timestamp_set_ = true;

                    if (subtype == Evt21EventMasterEventTypes::MASTER_IN_CD_EVENT_COUNT ||
                        subtype == Evt21EventMasterEventTypes::MASTER_RATE_CONTROL_CD_EVENT_COUNT) {
                        uint32_t count = ev_other->payload & ((1 << 22) - 1);
                        erc_count_forwarder.forward(last_timestamp<DO_TIMESHIFT>(), count,
                                                    subtype ==
                                                        Evt21EventMasterEventTypes::MASTER_RATE_CONTROL_CD_EVENT_COUNT);
                    }

                    uint32_t payload = 0x0;
                    // Has CONTINUED event?
                    if (ev_other->payload >> 28 == 0xF) {
                        payload = ev_other->payload & ~(0xF << 28);
                    }
                    monitoring_forwarder.forward(last_timestamp<DO_TIMESHIFT>(), ev_other->subtype, payload);
                }
                ++cur_ev;
            } else {
                ++cur_ev;
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
        if (is_time_shifting_enabled() && !time_shifting_set_) {
            return false;
        }
        if (t >= 0) {
            const timestamp shifted_time = t + (is_time_shifting_enabled() ? timestamp_shift_ : 0);
            last_timestamp_              = shifted_time;
            base_time_set_               = true;
            last_timestamp_set_          = true;
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
            timestamp_shift_   = shift;
            time_shifting_set_ = true;
            return true;
        }
        return false;
    }

    bool base_time_set_                       = false;
    bool last_timestamp_set_                  = false;
    timestamp last_timestamp_                 = 0;
    timestamp timestamp_shift_                = 0;
    bool time_shifting_set_                   = false;
    std::shared_ptr<I_EventDecoder<EventMonitoring>> event_monitoring_decoder_;
    std::unique_ptr<DecodedEventForwarder<EventMonitoring, 1>> monitoring_event_forwarder_;
    std::set<uint16_t> monitoring_id_blacklist_;
    static constexpr uint64_t max_time_high_  = ((1ULL << 28) - 1) << 6;
    static constexpr int loop_shift_          = 34;
    static constexpr uint64_t time_high_mask_ = ((1ULL << 28) - 1) << 6;

    void set_last_high_timestamp(uint64_t t) {
        const uint64_t last_high_ts = last_timestamp_ & time_high_mask_;
        uint64_t n_loop             = last_timestamp_ >> loop_shift_;
        if (t < last_high_ts) {
            if (last_high_ts - t >= max_time_high_) {
                ++n_loop;
            } else {
                MV_LOG_ERROR() << "Error TimeHigh discrepancy";
            }
        }
        if (t != last_high_ts) {
            last_timestamp_ = (t & time_high_mask_) | (n_loop << loop_shift_);
        }
    }
};

using EVT21Decoder = EVT21GenericDecoder<Evt21Raw::RawEvent, Evt21Raw::Event_TIME_HIGH, Evt21Raw::Event_2D,
                                         Evt21Raw::Event_EXT_TRIGGER, Evt21Raw::Event_OTHERS>;

using EVT21LegacyDecoder =
    EVT21GenericDecoder<Evt21LegacyRaw::RawEvent, Evt21LegacyRaw::Event_TIME_HIGH, Evt21LegacyRaw::Event_2D,
                        Evt21LegacyRaw::Event_EXT_TRIGGER, Evt21LegacyRaw::Event_OTHERS>;

using EVT21VectorizedDecoder = EVT21GenericDecoder<Evt21Raw::RawEvent, Evt21Raw::Event_TIME_HIGH, Evt21Raw::Event_2D,
                                                   Evt21Raw::Event_EXT_TRIGGER, Evt21Raw::Event_OTHERS, EventCDVector>;

} // namespace Metavision

#endif // METAVISION_HAL_EVT21_DECODER_H
