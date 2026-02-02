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

#ifndef METAVISION_SDK_STREAM_EVENT_FILE_READER_INTERNAL_H
#define METAVISION_SDK_STREAM_EVENT_FILE_READER_INTERNAL_H

#include <filesystem>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>
#include "metavision/sdk/core/utils/callback_manager.h"
#include "metavision/sdk/core/utils/index_manager.h"
#include "metavision/sdk/stream/event_file_reader.h"

namespace Metavision {

class EventFileReader::Private {
public:
    Private(EventFileReader &reader, const std::filesystem::path &path);

    const std::filesystem::path &get_path() const;

    size_t add_read_callback(const EventsBufferReadCallback<EventCD> &cb);
    size_t add_read_callback(const EventsBufferReadCallback<EventExtTrigger> &cb);
    size_t add_read_callback(const EventsBufferReadCallback<EventERCCounter> &cb);
    size_t add_read_callback(const EventsBufferReadCallback<EventMonitoring> &cb);
    size_t add_read_callback(const EventFrameReadCallback<RawEventFrameHisto> &cb);
    size_t add_read_callback(const EventFrameReadCallback<RawEventFrameDiff> &cb);
    size_t add_read_callback(const EventFrameReadCallback<PointCloud> &cb);
    bool has_read_callbacks() const;

    size_t add_seek_callback(const SeekCompletionCallback &cb);
    bool has_seek_callbacks() const;

    void remove_callback(size_t id);

    void notify_events_buffer(const EventCD *begin, const EventCD *end);
    void notify_events_buffer(const EventExtTrigger *begin, const EventExtTrigger *end);
    void notify_events_buffer(const EventERCCounter *begin, const EventERCCounter *end);
    void notify_events_buffer(const EventMonitoring *begin, const EventMonitoring *end);
    void notify_event_frame(const RawEventFrameHisto &h);
    void notify_event_frame(const RawEventFrameDiff &d);
    void notify_event_frame(const PointCloud &pc);
    void notify_seek(timestamp t);

    bool read();
    bool get_seek_range(timestamp &min_t, timestamp &max_t) const;
    timestamp get_duration() const;
    bool seek(timestamp t);

    std::unordered_map<std::string, std::string> get_metadata_map() const;

    EventFileReader &reader_;
    std::atomic<bool> seeking_;
    mutable std::mutex mutex_;
    mutable std::condition_variable cond_;
    IndexManager cb_id_mgr_;
    CallbackManager<EventsBufferReadCallback<EventCD>, size_t> cd_buffer_cb_mgr_;
    CallbackManager<EventsBufferReadCallback<EventExtTrigger>, size_t> ext_trigger_buffer_cb_mgr_;
    CallbackManager<EventsBufferReadCallback<EventERCCounter>, size_t> erc_counter_buffer_cb_mgr_;
    CallbackManager<EventsBufferReadCallback<EventMonitoring>, size_t> monitoring_buffer_cb_mgr_;
    CallbackManager<EventFrameReadCallback<RawEventFrameHisto>, size_t> histogram_cb_mgr_;
    CallbackManager<EventFrameReadCallback<RawEventFrameDiff>, size_t> diff_cb_mgr_;
    CallbackManager<EventFrameReadCallback<PointCloud>, size_t> pointcloud_cb_mgr_;
    CallbackManager<SeekCompletionCallback, size_t> seek_cb_mgr_;
    std::filesystem::path path_;
    mutable timestamp min_t_, max_t_, duration_;                        // cached
    mutable std::unordered_map<std::string, std::string> metadata_map_; // cached
};

} // namespace Metavision

#endif /* METAVISION_SDK_STREAM_EVENT_FILE_READER_INTERNAL_H */
