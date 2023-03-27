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

#ifndef METAVISION_HAL_AER_DECODER_H
#define METAVISION_HAL_AER_DECODER_H

#include <chrono>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/decoders/base/event_base.h"
#include "metavision/hal/facilities/i_geometry.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"

namespace Metavision {

template<bool HAS_4_BITS_INTERFACE>
class AERDecoder : public I_EventsStreamDecoder {
public:
    AERDecoder(bool time_shifting_enabled, const std::shared_ptr<I_EventDecoder<EventCD>> &event_cd_decoder =
                                               std::shared_ptr<I_EventDecoder<EventCD>>()) :
        I_EventsStreamDecoder(time_shifting_enabled, event_cd_decoder), ts_start_(std::chrono::system_clock::now()) {}

    virtual bool get_timestamp_shift(timestamp &ts_shift) const override {
        ts_shift = timestamp_shift_;
        return timestamp_shift_set_;
    }

    virtual timestamp get_last_timestamp() const override final {
        return last_timestamp_;
    }

    uint8_t get_raw_event_size_bytes() const override {
        return sizeof(RawData);
    }

private:
    template<bool DO_TIMESHIFT>
    timestamp do_timestamp() {
        const timestamp ts =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - ts_start_).count();
        last_timestamp_ = DO_TIMESHIFT ? ts - timestamp_shift_ : ts;
        return last_timestamp_;
    }

    static uint16_t decode_x(uint32_t data) {
        return (data >> 9) & CoordMask;
    }

    static uint16_t decode_y(uint32_t data) {
        return data & CoordMask;
    }

    static int16_t decode_pol(uint32_t data) {
        return (data >> 18) & 0x1;
    }

    virtual void decode_impl(const RawData *const cur_raw_data, const RawData *const raw_data_end) override {
        auto &cd_forwarder          = cd_event_forwarder();
        const RawData *raw_data_ptr = cur_raw_data;

        for (; raw_data_ptr != raw_data_end;) {
            decode_data_ |= *raw_data_ptr << decode_shift_;
            decode_shift_ += 8;

            if (decode_shift_ >= kNumBitsInEvent) {
                cd_forwarder.forward(decode_x(decode_data_), decode_y(decode_data_), decode_pol(decode_data_),
                                     is_time_shifting_enabled() ? do_timestamp<true>() : do_timestamp<true>());

                if (HAS_4_BITS_INTERFACE && decode_shift_ > kNumBitsInEvent) {
                    decode_data_ >>= kNumBitsInEvent;
                    decode_shift_ -= kNumBitsInEvent;
                } else {
                    decode_data_  = 0;
                    decode_shift_ = 0;
                }
            }

            ++raw_data_ptr;
        }
    }

    bool reset_timestamp_impl(const timestamp &t) override {
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

    virtual bool is_decoded_event_stream_indexable() const override {
        return false;
    }

    constexpr static uint32_t CoordMask       = (1 << 9) - 1;
    constexpr static uint32_t kNumBitsInEvent = HAS_4_BITS_INTERFACE ? 20 : 24;
    bool timestamp_shift_set_                 = true;
    timestamp timestamp_shift_                = 0;
    timestamp last_timestamp_                 = 0;
    std::chrono::time_point<std::chrono::system_clock> ts_start_;
    uint32_t decode_data_      = 0;
    unsigned int decode_shift_ = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_AER_DECODER_H
