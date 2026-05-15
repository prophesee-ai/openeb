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

#include "metavision/hal/facilities/i_event_frame_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/sdk/base/utils/generic_header.h"
#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/internal/event_file_reader_internal.h"
#include "metavision/sdk/stream/raw_event_file_reader.h"

namespace Metavision {

class RAWEventFileReader::Private {
public:
    Private(RAWEventFileReader &reader, Device &device) :
        reader_(reader), raw_data_cb_mgr_(get_parent_pimpl().cb_id_mgr_, 0), device_(device) {
        i_events_stream_         = device.get_facility<I_EventsStream>();
        i_events_stream_decoder_ = device.get_facility<I_EventsStreamDecoder>();
        i_decoder_               = device.get_facility<I_Decoder>();

        I_EventDecoder<EventCD> *i_cd_events_decoder = device.get_facility<I_EventDecoder<EventCD>>();
        if (i_cd_events_decoder) {
            i_cd_events_decoder->add_event_buffer_callback(
                [this](const EventCD *begin, const EventCD *end) { reader_.notify_events_buffer(begin, end); });
        }

        I_EventDecoder<EventExtTrigger> *i_ext_trigger_events_decoder =
            device.get_facility<I_EventDecoder<EventExtTrigger>>();
        if (i_ext_trigger_events_decoder) {
            i_ext_trigger_events_decoder->add_event_buffer_callback(
                [this](const EventExtTrigger *begin, const EventExtTrigger *end) {
                    reader_.notify_events_buffer(begin, end);
                });
        }

        I_EventDecoder<EventERCCounter> *i_erc_counter_events_decoder =
            device.get_facility<I_EventDecoder<EventERCCounter>>();
        if (i_erc_counter_events_decoder) {
            i_erc_counter_events_decoder->add_event_buffer_callback(
                [this](const EventERCCounter *begin, const EventERCCounter *end) {
                    reader_.notify_events_buffer(begin, end);
                });
        }

        I_EventDecoder<EventMonitoring> *i_monitoring_events_decoder =
            device.get_facility<I_EventDecoder<EventMonitoring>>();
        if (i_monitoring_events_decoder) {
            i_monitoring_events_decoder->add_event_buffer_callback(
                [this](const EventMonitoring *begin, const EventMonitoring *end) {
                    reader_.notify_events_buffer(begin, end);
                });
        }

        auto i_histogram_decoder = device.get_facility<I_EventFrameDecoder<RawEventFrameHisto>>();
        if (i_histogram_decoder) {
            i_histogram_decoder->add_event_frame_callback(
                [this](const RawEventFrameHisto &h) { reader_.notify_event_frame(h); });
        }

        auto i_diff_decoder = device.get_facility<I_EventFrameDecoder<RawEventFrameDiff>>();
        if (i_diff_decoder) {
            i_diff_decoder->add_event_frame_callback(
                [this](const RawEventFrameDiff &d) { reader_.notify_event_frame(d); });
        }
    }

    size_t add_raw_read_callback(const RawDataBufferReadCallback &cb) {
        return raw_data_cb_mgr_.add_callback(cb);
    }

    bool has_raw_read_callbacks() const {
        return get_parent_pimpl().cb_id_mgr_.counter_map_.tag_count(0) > 0;
    }

    void notify_raw_data_buffer(const std::uint8_t *begin, const std::uint8_t *end) {
        auto cbs = raw_data_cb_mgr_.get_cbs();
        for (auto &cb : cbs) {
            cb(begin, end);
        }
    }

    void remove_raw_callback(size_t id) {
        if (raw_data_cb_mgr_.remove_callback(id)) {
            return;
        }
    }

    void start() {
        if (!started_) {
            if (i_events_stream_) {
                i_events_stream_->start();
            }
            started_ = true;
        }
    }

    void stop() {
        if (started_) {
            if (i_events_stream_) {
                i_events_stream_->stop();
            }
            started_ = false;
        }
    }

    bool read_impl() {
        if (!started_) {
            start();
        }

        if (raw_data_cur_ptr_ == raw_data_end_ptr_) {
            int res = 0;
            if (i_events_stream_) {
                res = i_events_stream_->wait_next_buffer();
            }
            if (res <= 0) {
                return false;
            }
            if (i_events_stream_) {
                data_buffer_      = i_events_stream_->get_latest_raw_data();
                raw_data_cur_ptr_ = data_buffer_.data();
                raw_data_end_ptr_ = data_buffer_.end();
            }
        }

        uint32_t num_bytes_to_decode;
        if (i_events_stream_decoder_) {
            // Decode events chunk by chunk to allow early stop and better cadencing when emulating real time
            constexpr uint32_t num_events_to_decode = 1024;
            num_bytes_to_decode = i_events_stream_decoder_->get_raw_event_size_bytes() * num_events_to_decode;
        } else {
            num_bytes_to_decode = i_decoder_->get_raw_event_size_bytes();
        }

        const uint32_t num_remaining_bytes = std::distance(raw_data_cur_ptr_, raw_data_end_ptr_);
        num_bytes_to_decode                = std::min(num_remaining_bytes, num_bytes_to_decode);

        // we first decode the buffer and call the corresponding events callback ...
        if (reader_.has_read_callbacks()) {
            i_decoder_->decode(raw_data_cur_ptr_, raw_data_cur_ptr_ + num_bytes_to_decode);
        }

        // ... then we call the raw buffer callback so that a user has access to some info (e.g last
        // decoded timestamp) when the raw callback is called
        if (reader_.has_raw_read_callbacks()) {
            reader_.notify_raw_data_buffer(raw_data_cur_ptr_, raw_data_cur_ptr_ + num_bytes_to_decode);
        }

        raw_data_cur_ptr_ += num_bytes_to_decode;

        return true;
    }

    bool seekable() const {
        return i_events_stream_decoder_ != nullptr;
    }

    bool get_seek_range_impl(timestamp &min_t, timestamp &max_t) const {
        if (i_events_stream_) {
            return i_events_stream_->get_seek_range(min_t, max_t) == I_EventsStream::IndexStatus::Good;
        }
        return false;
    }

    timestamp get_duration_impl() const {
        if (!i_events_stream_decoder_) {
            return -1;
        }
        timestamp duration    = -1;
        FileConfigHints hints = FileConfigHints().real_time_playback(false);
        Camera cam            = Camera::from_file(reader_.get_path(), hints);
        try {
            cam.get_device().get_facility<I_EventsStreamDecoder>()->add_protocol_violation_callback([](auto) {});
        } catch (HalException &) {}
        cam.cd().add_callback([&duration](const EventCD *begin, const EventCD *end) {
            duration = std::max(duration, std::prev(end)->t);
        });
        cam.start();
        bool seek_done = false;
        while (cam.is_running()) {
            if (!seek_done && cam.offline_streaming_control().is_ready()) {
                cam.offline_streaming_control().seek(cam.offline_streaming_control().get_seek_end_time());
                duration  = std::max(duration, cam.offline_streaming_control().get_seek_end_time());
                seek_done = true;
            }
            std::this_thread::yield();
        }
        return duration;
    }

    bool seek_impl(timestamp t) {
        timestamp ts;
        if (i_events_stream_->seek(t, ts) == I_EventsStream::SeekStatus::Success) {
            i_events_stream_decoder_->reset_last_timestamp(ts);
            reader_.notify_seek(ts);
            raw_data_cur_ptr_ = raw_data_end_ptr_; // force a wait_next_buffer at next read_impl
            return true;
        }
        return false;
    }

    std::unordered_map<std::string, std::string> get_metadata_map_impl() const {
        auto hw_id = device_.get_facility<I_HW_Identification>();
        if (hw_id) {
            auto m = hw_id->get_header().get_header_map();
            return {m.begin(), m.end()};
        };
        return {};
    }

    EventFileReader::Private &get_parent_pimpl() {
        return reader_.EventFileReader::get_pimpl();
    }

    const EventFileReader::Private &get_parent_pimpl() const {
        return reader_.EventFileReader::get_pimpl();
    }

    bool started_               = false;
    mutable timestamp duration_ = -1;
    RAWEventFileReader &reader_;
    CallbackManager<RawDataBufferReadCallback, size_t> raw_data_cb_mgr_;
    Device &device_;
    I_EventsStream *i_events_stream_                = nullptr;
    I_EventsStreamDecoder *i_events_stream_decoder_ = nullptr;
    I_Decoder *i_decoder_                           = nullptr;
    DataTransfer::BufferPtr data_buffer_;
    const std::uint8_t *raw_data_cur_ptr_ = nullptr, *raw_data_end_ptr_ = nullptr;
};

RAWEventFileReader::RAWEventFileReader(Device &device, const std::filesystem::path &path) :
    EventFileReader(path), pimpl_(new Private(*this, device)) {}

RAWEventFileReader::~RAWEventFileReader() {}

size_t RAWEventFileReader::add_raw_read_callback(const RawDataBufferReadCallback &cb) {
    return pimpl_->add_raw_read_callback(cb);
}

bool RAWEventFileReader::has_raw_read_callbacks() const {
    return pimpl_->has_raw_read_callbacks();
}

void RAWEventFileReader::remove_raw_callback(size_t id) {
    pimpl_->remove_raw_callback(id);
}

void RAWEventFileReader::notify_raw_data_buffer(const std::uint8_t *begin, const std::uint8_t *end) {
    return pimpl_->notify_raw_data_buffer(begin, end);
}

void RAWEventFileReader::start() {
    return pimpl_->start();
}

void RAWEventFileReader::stop() {
    return pimpl_->stop();
}

bool RAWEventFileReader::read_impl() {
    return pimpl_->read_impl();
}

bool RAWEventFileReader::seekable() const {
    return pimpl_->seekable();
}

bool RAWEventFileReader::seek_impl(timestamp t) {
    return pimpl_->seek_impl(t);
}

bool RAWEventFileReader::get_seek_range_impl(timestamp &min_t, timestamp &max_t) const {
    return pimpl_->get_seek_range_impl(min_t, max_t);
}

timestamp RAWEventFileReader::get_duration_impl() const {
    return pimpl_->get_duration_impl();
}

std::unordered_map<std::string, std::string> RAWEventFileReader::get_metadata_map_impl() const {
    return pimpl_->get_metadata_map_impl();
}

} // namespace Metavision
