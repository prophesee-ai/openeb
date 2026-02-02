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
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "metavision/hal/decoders/evt2/evt2_encoder.h"
#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/internal/event_file_writer_internal.h"
#include "metavision/sdk/stream/raw_evt2_event_file_writer.h"

namespace Metavision {

RAWEvt2EventFileWriter::RAWEvt2EventFileWriter(int stream_width, int stream_height, const std::filesystem::path &path,
                                               bool enable_trigger_support,
                                               const std::unordered_map<std::string, std::string> &metadata_map,
                                               timestamp max_events_add_latency) :
    EventFileWriter(path.string()),
    exttrigger_support_enabled_(enable_trigger_support),
    max_events_add_latency_(max_events_add_latency <= 0 ? std::numeric_limits<timestamp>::max() :
                                                          max_events_add_latency),
    encoder_(std::make_unique<Evt2Encoder>()) {
    get_pimpl().set_max_event_trigger_buffer_size(1);
    if (!path.empty()) {
        open_impl(path);
    }
    for (auto &p : metadata_map) {
        add_metadata_impl(p.first, p.second);
    }
    std::stringstream ss;
    ss << "EVT2;width=" << stream_width << ";height=" << stream_height;
    add_metadata_impl("format", ss.str());
    add_metadata_impl("camera_integrator_name", "MetavisionSDK");
    add_metadata_impl("plugin_integrator_name", "MetavisionSDK");
    header_.add_date();
}

RAWEvt2EventFileWriter::~RAWEvt2EventFileWriter() {
    if (is_open_impl()) {
        close();
    }
}

void RAWEvt2EventFileWriter::open_impl(const std::filesystem::path &path) {
    if (is_open_impl()) {
        close_impl();
    }
    ofs_ = std::ofstream(path, std::ios::binary);
    if (!ofs_.is_open()) {
        std::stringstream ss;
        ss << "[RAWEvt2EventFileWriter] Unable to open " << path << " for writing";
        MV_SDK_LOG_ERROR() << ss.str();
        throw std::runtime_error(ss.str());
    }
}

void RAWEvt2EventFileWriter::close_impl() {
    if (!header_written_) {
        ofs_ << header_;
        header_written_ = true;
    }
    encode_buffered_events(true);
    ofs_.close();
}

bool RAWEvt2EventFileWriter::is_open_impl() const {
    return ofs_.is_open();
}

void RAWEvt2EventFileWriter::add_metadata_impl(const std::string &key, const std::string &value) {
    if (header_written_) {
        MV_SDK_LOG_ERROR() << "[RAWEvt2EventFileWriter] Unable to modify metadata in RAW once data has been added";
        throw std::runtime_error("Unable to modify metadata in RAW once data has been added");
    }
    header_.set_field(key, value);
}

void RAWEvt2EventFileWriter::add_metadata_map_from_camera_impl(const Camera &camera) {
    MV_SDK_LOG_ERROR() << "[RAWEvt2EventFileWriter] add_metadata_map_from_camera is not supported!";
    throw std::runtime_error("add_metadata_map_from_camera is not supported!");
}

void RAWEvt2EventFileWriter::remove_metadata_impl(const std::string &key) {
    if (header_written_) {
        MV_SDK_LOG_ERROR() << "[RAWEvt2EventFileWriter] Unable to modify metadata in RAW once data has been added";
        throw std::runtime_error("Unable to modify metadata in RAW once data has been added");
    }
    header_.remove_field(key);
}

bool RAWEvt2EventFileWriter::add_events_impl(const EventCD *begin, const EventCD *end) {
    if (!is_open_impl()) {
        MV_SDK_LOG_ERROR() << "[RAWEvt2EventFileWriter] File writer is not open, ignoring input events";
        return false;
    }
    if (begin != end) {
        events_cd_.insert(events_cd_.end(), begin, end);
        ts_last_cd_ = std::prev(end)->t;
        encode_buffered_events(false);
    }
    return true;
}

bool RAWEvt2EventFileWriter::add_events_impl(const EventExtTrigger *begin, const EventExtTrigger *end) {
    if (!exttrigger_support_enabled_) {
        MV_SDK_LOG_ERROR()
            << "[RAWEvt2EventFileWriter] External trigger support is not enabled, ignoring input trigger events";
        return false;
    }
    if (!is_open_impl()) {
        MV_SDK_LOG_ERROR() << "[RAWEvt2EventFileWriter] File writer is not open, ignoring input events";
        return false;
    }
    if (begin != end) {
        events_trigger_.insert(events_trigger_.end(), begin, end);
        ts_last_trigger_ = std::prev(end)->t;
        encode_buffered_events(false);
    }
    return true;
}

void RAWEvt2EventFileWriter::flush_impl() {
    if (!is_open_impl()) {
        return;
    }
    std::atomic<bool> done{false};
    get_pimpl().writer_thread_.add_task([&done, this]() {
        encode_buffered_events(false);
        ofs_.flush();
        done = true;
    });
    while (!done) {
        std::this_thread::yield();
    }
}

void RAWEvt2EventFileWriter::encode_buffered_events(bool flush_all_queued_events) {
    if (!header_written_) {
        ofs_ << header_;
        header_written_ = true;
    }
    if (exttrigger_support_enabled_) {
        merge_encode_buffered_events(flush_all_queued_events);
    } else {
        for (const auto &ev : events_cd_) {
            encoder_->encode_event_cd(ofs_, ev);
        }
        events_cd_.clear();
    }
}

void RAWEvt2EventFileWriter::merge_encode_buffered_events(bool flush_all_queued_events) {
    timestamp ts_encode_up_to;
    if (flush_all_queued_events) {
        ts_encode_up_to = std::numeric_limits<timestamp>::max();
    } else if (max_events_add_latency_ == std::numeric_limits<timestamp>::max()) {
        ts_encode_up_to = std::min(ts_last_cd_, ts_last_trigger_);
    } else {
        const timestamp ts_last = std::max(ts_last_cd_, ts_last_trigger_);
        ts_encode_up_to         = ts_last < std::numeric_limits<timestamp>::min() + max_events_add_latency_ ?
                                      std::numeric_limits<timestamp>::min() :
                                      ts_last - max_events_add_latency_;
    }
    auto it_cd          = events_cd_.begin(),
         it_cd_end      = std::lower_bound(events_cd_.begin(), events_cd_.end(), ts_encode_up_to,
                                           [](const EventCD &ev, timestamp ts) { return ev.t < ts; });
    auto it_trigger     = events_trigger_.begin(),
         it_trigger_end = std::lower_bound(events_trigger_.begin(), events_trigger_.end(), ts_encode_up_to,
                                           [](const EventExtTrigger &ev, timestamp ts) { return ev.t < ts; });

    while (it_cd != it_cd_end || it_trigger != it_trigger_end) {
        if (it_cd != it_cd_end && it_trigger != it_trigger_end) {
            if (it_cd->t < it_trigger->t) {
                encoder_->encode_event_cd(ofs_, *it_cd);
                ++it_cd;
            } else {
                encoder_->encode_event_trigger(ofs_, *it_trigger);
                ++it_trigger;
            }
        } else if (it_cd != it_cd_end) {
            encoder_->encode_event_cd(ofs_, *it_cd);
            ++it_cd;
        } else {
            encoder_->encode_event_trigger(ofs_, *it_trigger);
            ++it_trigger;
        }
    }
    events_cd_.erase(events_cd_.begin(), it_cd_end);
    events_trigger_.erase(events_trigger_.begin(), it_trigger_end);
}

} // namespace Metavision
