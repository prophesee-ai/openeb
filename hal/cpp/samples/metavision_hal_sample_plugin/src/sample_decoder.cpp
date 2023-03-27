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

#include <algorithm>

#include "sample_decoder.h"
#include "sample_events_format.h"

SampleDecoder::SampleDecoder(bool do_time_shift,
                             const std::shared_ptr<Metavision::I_EventDecoder<Metavision::EventCD>> &cd_event_decoder) :
    Metavision::I_EventsStreamDecoder(do_time_shift, cd_event_decoder) {}

void SampleDecoder::decode_impl(const RawData *const ev, const RawData *const evend) {
    if (ev == evend) {
        return;
    }

    // Note: Input guarantees std::distance(ev, evend) % sizeof(SampleEventsFormat) = 0
    const SampleEventsFormat *current_ev = reinterpret_cast<const SampleEventsFormat *>(ev);
    const SampleEventsFormat *ev_end     = reinterpret_cast<const SampleEventsFormat *>(evend);
    Metavision::EventCD event_decoded(0, 0, 0, last_timestamp_);
    auto &cd_forwarder = cd_event_forwarder();

    // If the time shift is enabled, check if we already set it. If not, set it now
    if (is_time_shifting_enabled()) {
        if (!time_shift_set_) {
            time_shift_     = (*current_ev) & TS_MASK;
            time_shift_set_ = true;
        }
    }

    // Remark : we have the guarantee that the input buffer length is a multiple of sizeof(SampleEventsFormat),
    // so we can use != in the exit condition of the for loop below
    for (; current_ev != ev_end; ++current_ev) {
        decode_sample_format(*current_ev, event_decoded, time_shift_);
        cd_forwarder.forward(event_decoded);
    }

    last_timestamp_ = event_decoded.t;
}

Metavision::timestamp SampleDecoder::get_last_timestamp() const {
    return last_timestamp_;
}

bool SampleDecoder::get_timestamp_shift(Metavision::timestamp &ts_shift) const {
    ts_shift = time_shift_;
    return time_shift_set_;
}

uint8_t SampleDecoder::get_raw_event_size_bytes() const {
    return sizeof(SampleEventsFormat);
}

bool SampleDecoder::reset_timestamp_impl(const Metavision::timestamp &t) {
    if (is_time_shifting_enabled() && !time_shift_set_) {
        return false;
    }
    last_timestamp_ = t;
    return true;
}

bool SampleDecoder::reset_timestamp_shift_impl(const Metavision::timestamp &shift) {
    if (shift >= 0 && is_time_shifting_enabled()) {
        time_shift_     = shift;
        time_shift_set_ = true;
        return true;
    }
    return false;
}
