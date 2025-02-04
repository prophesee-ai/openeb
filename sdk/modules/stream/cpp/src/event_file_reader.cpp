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

#include "metavision/sdk/stream/internal/callback_tag_ids.h"
#include "metavision/sdk/stream/internal/event_file_reader_internal.h"

namespace Metavision {

EventFileReader::Private::Private(EventFileReader &reader, const std::filesystem::path &path) :
    reader_(reader),
    seeking_(false),
    cd_buffer_cb_mgr_(cb_id_mgr_, CallbackTagIds::READ_CALLBACK_TAG_ID),
    ext_trigger_buffer_cb_mgr_(cb_id_mgr_, CallbackTagIds::READ_CALLBACK_TAG_ID),
    erc_counter_buffer_cb_mgr_(cb_id_mgr_, CallbackTagIds::READ_CALLBACK_TAG_ID),
    monitoring_buffer_cb_mgr_(cb_id_mgr_, CallbackTagIds::READ_CALLBACK_TAG_ID),
    histogram_cb_mgr_(cb_id_mgr_, CallbackTagIds::READ_CALLBACK_TAG_ID),
    diff_cb_mgr_(cb_id_mgr_, CallbackTagIds::READ_CALLBACK_TAG_ID),
    pointcloud_cb_mgr_(cb_id_mgr_, CallbackTagIds::READ_CALLBACK_TAG_ID),
    seek_cb_mgr_(cb_id_mgr_, CallbackTagIds::SEEK_CALLBACK_TAG_ID),
    path_(path),
    min_t_(-1),
    max_t_(-1),
    duration_(-1) {}

const std::filesystem::path &EventFileReader::Private::get_path() const {
    return path_;
}

size_t EventFileReader::Private::add_read_callback(const EventsBufferReadCallback<EventCD> &cb) {
    return cd_buffer_cb_mgr_.add_callback(cb);
}

size_t EventFileReader::Private::add_read_callback(const EventsBufferReadCallback<EventExtTrigger> &cb) {
    return ext_trigger_buffer_cb_mgr_.add_callback(cb);
}

size_t EventFileReader::Private::add_read_callback(const EventsBufferReadCallback<EventERCCounter> &cb) {
    return erc_counter_buffer_cb_mgr_.add_callback(cb);
}

size_t EventFileReader::Private::add_read_callback(const EventFrameReadCallback<RawEventFrameHisto> &cb) {
    return histogram_cb_mgr_.add_callback(cb);
}

size_t EventFileReader::Private::add_read_callback(const EventFrameReadCallback<RawEventFrameDiff> &cb) {
    return diff_cb_mgr_.add_callback(cb);
}

size_t EventFileReader::Private::add_read_callback(const EventFrameReadCallback<PointCloud> &cb) {
    return pointcloud_cb_mgr_.add_callback(cb);
}

bool EventFileReader::Private::has_read_callbacks() const {
    return cb_id_mgr_.counter_map_.tag_count(CallbackTagIds::READ_CALLBACK_TAG_ID) > 0;
}

void EventFileReader::Private::notify_events_buffer(const EventCD *begin, const EventCD *end) {
    auto cbs = cd_buffer_cb_mgr_.get_cbs();
    for (auto &cb : cbs) {
        cb(begin, end);
    }
}

void EventFileReader::Private::notify_events_buffer(const EventExtTrigger *begin, const EventExtTrigger *end) {
    auto cbs = ext_trigger_buffer_cb_mgr_.get_cbs();
    for (auto &cb : cbs) {
        cb(begin, end);
    }
}

void EventFileReader::Private::notify_events_buffer(const EventERCCounter *begin, const EventERCCounter *end) {
    auto cbs = erc_counter_buffer_cb_mgr_.get_cbs();
    for (auto &cb : cbs) {
        cb(begin, end);
    }
}

void EventFileReader::Private::notify_events_buffer(const EventMonitoring *begin, const EventMonitoring *end) {
    auto cbs = monitoring_buffer_cb_mgr_.get_cbs();
    for (auto &cb : cbs) {
        cb(begin, end);
    }
}

void EventFileReader::Private::notify_event_frame(const RawEventFrameHisto &h) {
    auto cbs = histogram_cb_mgr_.get_cbs();
    for (auto &cb : cbs) {
        cb(h);
    }
}

void EventFileReader::Private::notify_event_frame(const RawEventFrameDiff &d) {
    auto cbs = diff_cb_mgr_.get_cbs();
    for (auto &cb : cbs) {
        cb(d);
    }
}

void EventFileReader::Private::notify_event_frame(const PointCloud &pc) {
    auto cbs = pointcloud_cb_mgr_.get_cbs();
    for (auto &cb : cbs) {
        cb(pc);
    }
}

size_t EventFileReader::Private::add_seek_callback(const SeekCompletionCallback &cb) {
    return seek_cb_mgr_.add_callback(cb);
}

bool EventFileReader::Private::has_seek_callbacks() const {
    return cb_id_mgr_.counter_map_.tag_count(CallbackTagIds::SEEK_CALLBACK_TAG_ID) > 0;
}

void EventFileReader::Private::notify_seek(timestamp t) {
    auto cbs = seek_cb_mgr_.get_cbs();
    for (auto &cb : cbs) {
        cb(t);
    }
}

void EventFileReader::Private::remove_callback(size_t id) {
    if (cd_buffer_cb_mgr_.remove_callback(id)) {
        return;
    }
    if (ext_trigger_buffer_cb_mgr_.remove_callback(id)) {
        return;
    }
    if (erc_counter_buffer_cb_mgr_.remove_callback(id)) {
        return;
    }
    if (seek_cb_mgr_.remove_callback(id)) {
        return;
    }
}

bool EventFileReader::Private::read() {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return !seeking_; });
    return reader_.read_impl();
}

bool EventFileReader::Private::get_seek_range(timestamp &min_t, timestamp &max_t) const {
    bool ret = true;
    if (min_t_ < 0 || max_t_ < 0) {
        std::unique_lock<std::mutex> lock(mutex_);
        ret = reader_.get_seek_range_impl(min_t_, max_t_);
    }
    min_t = min_t_;
    max_t = max_t_;
    return ret;
}

timestamp EventFileReader::Private::get_duration() const {
    if (duration_ < 0) {
        std::unique_lock<std::mutex> lock(mutex_);
        duration_ = reader_.get_duration_impl();
    }
    return duration_;
}

bool EventFileReader::Private::seek(timestamp t) {
    bool ret = false;
    seeking_ = true;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        ret      = reader_.seek_impl(t);
        seeking_ = false;
    }
    cond_.notify_all();
    return ret;
}

std::unordered_map<std::string, std::string> EventFileReader::Private::get_metadata_map() const {
    if (metadata_map_.empty()) {
        std::unique_lock<std::mutex> lock(mutex_);
        metadata_map_ = reader_.get_metadata_map_impl();
    }
    return metadata_map_;
}

EventFileReader::EventFileReader(const std::filesystem::path &path) : pimpl_(new Private(*this, path)) {}

EventFileReader::~EventFileReader() {}

const std::filesystem::path &EventFileReader::get_path() const {
    return pimpl_->get_path();
}

size_t EventFileReader::add_read_callback(const EventsBufferReadCallback<EventCD> &cb) {
    return pimpl_->add_read_callback(cb);
}

size_t EventFileReader::add_read_callback(const EventsBufferReadCallback<EventExtTrigger> &cb) {
    return pimpl_->add_read_callback(cb);
}

size_t EventFileReader::add_read_callback(const EventsBufferReadCallback<EventERCCounter> &cb) {
    return pimpl_->add_read_callback(cb);
}

size_t EventFileReader::add_read_callback(const EventFrameReadCallback<RawEventFrameHisto> &cb) {
    return pimpl_->add_read_callback(cb);
}

size_t EventFileReader::add_read_callback(const EventFrameReadCallback<RawEventFrameDiff> &cb) {
    return pimpl_->add_read_callback(cb);
}

size_t EventFileReader::add_read_callback(const EventFrameReadCallback<PointCloud> &cb) {
    return pimpl_->add_read_callback(cb);
}

bool EventFileReader::has_read_callbacks() const {
    return pimpl_->has_read_callbacks();
}

size_t EventFileReader::add_seek_callback(const SeekCompletionCallback &cb) {
    return pimpl_->add_seek_callback(cb);
}

bool EventFileReader::has_seek_callbacks() const {
    return pimpl_->has_seek_callbacks();
}

void EventFileReader::remove_callback(size_t id) {
    return pimpl_->remove_callback(id);
}

void EventFileReader::notify_events_buffer(const EventCD *begin, const EventCD *end) {
    return pimpl_->notify_events_buffer(begin, end);
}

void EventFileReader::notify_events_buffer(const EventExtTrigger *begin, const EventExtTrigger *end) {
    return pimpl_->notify_events_buffer(begin, end);
}

void EventFileReader::notify_events_buffer(const EventERCCounter *begin, const EventERCCounter *end) {
    return pimpl_->notify_events_buffer(begin, end);
}

void EventFileReader::notify_events_buffer(const EventMonitoring *begin, const EventMonitoring *end) {
    return pimpl_->notify_events_buffer(begin, end);
}

void EventFileReader::notify_event_frame(const RawEventFrameHisto &h) {
    return pimpl_->notify_event_frame(h);
}

void EventFileReader::notify_event_frame(const RawEventFrameDiff &d) {
    return pimpl_->notify_event_frame(d);
}

void EventFileReader::notify_event_frame(const PointCloud &p) {
    return pimpl_->notify_event_frame(p);
}

void EventFileReader::notify_seek(timestamp t) {
    return pimpl_->notify_seek(t);
}

bool EventFileReader::read() {
    return pimpl_->read();
}

bool EventFileReader::get_seek_range(timestamp &min_t, timestamp &max_t) const {
    return pimpl_->get_seek_range(min_t, max_t);
}

timestamp EventFileReader::get_duration() const {
    return pimpl_->get_duration();
}

bool EventFileReader::seek(timestamp t) {
    return pimpl_->seek(t);
}

std::unordered_map<std::string, std::string> EventFileReader::get_metadata_map() const {
    return pimpl_->get_metadata_map();
}

std::unordered_map<std::string, std::string> EventFileReader::get_metadata_map_impl() const {
    return {};
}

EventFileReader::Private &EventFileReader::get_pimpl() {
    return *pimpl_;
}

} // namespace Metavision
