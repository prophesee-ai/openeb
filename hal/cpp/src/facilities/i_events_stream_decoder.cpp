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

#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/utils/hal_exception.h"

namespace Metavision {

I_EventsStreamDecoder::I_EventsStreamDecoder(
    bool time_shifting_enabled, const std::shared_ptr<I_EventDecoder<EventCD>> &cd_event_decoder,
    const std::shared_ptr<I_EventDecoder<EventExtTrigger>> &ext_trigger_event_decoder,
    const std::shared_ptr<I_EventDecoder<EventERCCounter>> &erc_count_event_decoder) :
    is_time_shifting_enabled_(time_shifting_enabled),
    cd_event_decoder_(cd_event_decoder),
    ext_trigger_event_decoder_(ext_trigger_event_decoder),
    erc_count_event_decoder_(erc_count_event_decoder) {
    if (cd_event_decoder_) {
        cd_event_forwarder_.reset(new DecodedEventForwarder<EventCD>(cd_event_decoder_.get()));
    }
    if (ext_trigger_event_decoder_) {
        trigger_event_forwarder_.reset(new DecodedEventForwarder<EventExtTrigger, 1>(ext_trigger_event_decoder_.get()));
    }
    if (erc_count_event_decoder_) {
        erc_count_event_forwarder_.reset(new DecodedEventForwarder<EventERCCounter, 1>(erc_count_event_decoder_.get()));
    }
}

I_EventsStreamDecoder::I_EventsStreamDecoder(
    bool time_shifting_enabled, const std::shared_ptr<I_EventDecoder<EventCDVector>> &cd_vector_event_decoder,
    const std::shared_ptr<I_EventDecoder<EventExtTrigger>> &ext_trigger_event_decoder,
    const std::shared_ptr<I_EventDecoder<EventERCCounter>> &erc_count_event_decoder) :
    is_time_shifting_enabled_(time_shifting_enabled),
    cd_event_vector_decoder_(cd_vector_event_decoder),
    ext_trigger_event_decoder_(ext_trigger_event_decoder),
    erc_count_event_decoder_(erc_count_event_decoder) {
    if (cd_event_vector_decoder_) {
        cd_event_vector_forwarder_.reset(new DecodedEventForwarder<EventCDVector>(cd_event_vector_decoder_.get()));
    }
    if (ext_trigger_event_decoder_) {
        trigger_event_forwarder_.reset(new DecodedEventForwarder<EventExtTrigger, 1>(ext_trigger_event_decoder_.get()));
    }
    if (erc_count_event_decoder_) {
        erc_count_event_forwarder_.reset(new DecodedEventForwarder<EventERCCounter, 1>(erc_count_event_decoder_.get()));
    }
}

bool I_EventsStreamDecoder::is_time_shifting_enabled() const {
    return is_time_shifting_enabled_;
}

void I_EventsStreamDecoder::decode(const RawData *const raw_data_begin, const RawData *const raw_data_end) {
    const RawData *cur_raw_data = raw_data_begin;

    // We first decode incomplete data from previous decode call
    if (!incomplete_raw_data_.empty()) {
        // Computes how many raw data from this input need to be copied to get a complete raw event and append
        // them to the incomplete raw data..
        const int raw_data_to_insert_count = get_raw_event_size_bytes() - incomplete_raw_data_.size();

        // Check that the input buffer has enough data to complete the raw event
        if (raw_data_to_insert_count > std::distance(cur_raw_data, raw_data_end)) {
            incomplete_raw_data_.insert(incomplete_raw_data_.end(), cur_raw_data, raw_data_end);
            return;
        }

        // The necessary amount of data is present in the input, decode the now complete raw event
        incomplete_raw_data_.insert(incomplete_raw_data_.end(), cur_raw_data, cur_raw_data + raw_data_to_insert_count);
        decode_impl(incomplete_raw_data_.data(), incomplete_raw_data_.data() + incomplete_raw_data_.size());
        incomplete_raw_data_.clear();

        cur_raw_data += raw_data_to_insert_count;
    }

    // Computes a valid end iterator from the input data so that the data size to decode is a multiple of raw event size
    const auto raw_data_end_decodable_range =
        cur_raw_data +
        get_raw_event_size_bytes() * (std::distance(cur_raw_data, raw_data_end) / get_raw_event_size_bytes());

    // Decode the data
    decode_impl(cur_raw_data, raw_data_end_decodable_range);

    if (raw_data_end_decodable_range != raw_data_end) {
        // If the decodable range was not the same as the input (i.e. not a multiple of event bytes size) then we
        // keep the remaining truncated data in memory. They are inserted in the incomplete data.
        incomplete_raw_data_.insert(incomplete_raw_data_.end(), raw_data_end_decodable_range, raw_data_end);
    }

    // Flush the decoders and call time callbacks
    if (cd_event_forwarder_) {
        cd_event_forwarder_->flush();
    }
    if (cd_event_vector_forwarder_) {
        cd_event_vector_forwarder_->flush();
    }
    if (trigger_event_forwarder_) {
        trigger_event_forwarder_->flush();
    }
    if (erc_count_event_forwarder_) {
        erc_count_event_forwarder_->flush();
    }
    timestamp last_ts = get_last_timestamp();
    for (auto it = time_cbs_map_.begin(), it_end = time_cbs_map_.end(); it != it_end; ++it) {
        it->second(last_ts);
    }
}

void I_EventsStreamDecoder::decode(const DataTransfer::BufferPtr &raw_buffer) {
    decode(raw_buffer.begin(), raw_buffer.end());
}

size_t I_EventsStreamDecoder::add_time_callback(const TimeCallback_t &cb) {
    time_cbs_map_[next_cb_idx_] = cb;
    return next_cb_idx_++;
}

bool I_EventsStreamDecoder::remove_time_callback(size_t callback_id) {
    auto it = time_cbs_map_.find(callback_id);
    if (it != time_cbs_map_.end()) {
        time_cbs_map_.erase(it);
        return true;
    }
    return false;
}

bool I_EventsStreamDecoder::reset_last_timestamp(const timestamp &timestamp) {
    incomplete_raw_data_.clear();
    return reset_last_timestamp_impl(timestamp);
}

bool I_EventsStreamDecoder::reset_timestamp_shift(const timestamp &t) {
    return reset_timestamp_shift_impl(t);
}

bool I_EventsStreamDecoder::is_decoded_event_stream_indexable() const {
    return true;
}

} // namespace Metavision
