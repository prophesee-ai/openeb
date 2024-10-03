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

#ifndef METAVISION_HAL_SAMPLE_DECODER_H
#define METAVISION_HAL_SAMPLE_DECODER_H

#include <metavision/hal/facilities/i_events_stream_decoder.h>

/// @brief Interface for decoding events.
///
/// This class is the implementation of HAL's facility @ref Metavision::I_EventsStreamDecoder
/// The implementation must have the following feature :
///
/// - support of time shifting : If enabled, the timestamp of the decoded events will be shifted of the value of the
///   time of the first event
class SampleDecoder : public Metavision::I_EventsStreamDecoder {
public:
    SampleDecoder(bool do_time_shift,
                  const std::shared_ptr<Metavision::I_EventDecoder<Metavision::EventCD>> &cd_event_decoder);
    Metavision::timestamp get_last_timestamp() const override final;
    bool get_timestamp_shift(Metavision::timestamp &timestamp_shift) const override final;
    uint8_t get_raw_event_size_bytes() const override final;

private:
    void decode_impl(const RawData *const ev, const RawData *const evend) override final;
    bool reset_last_timestamp_impl(const Metavision::timestamp &timestamp) override final;
    bool reset_timestamp_shift_impl(const Metavision::timestamp &shift) override final;

    Metavision::timestamp last_timestamp_{0};
    Metavision::timestamp time_shift_{0};
    bool time_shift_set_{false};
};

#endif // METAVISION_HAL_SAMPLE_DECODER_H
