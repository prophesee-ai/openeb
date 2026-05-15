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

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/internal/event_file_writer_internal.h"

namespace Metavision {

EventFileWriter::Private::Private(EventFileWriter &writer, const std::filesystem::path &path) :
    writer_(writer), last_cd_ts_(-1), last_ext_trigger_ts_(-1), path_(path) {
    writer_thread_.start();
    cd_buffer_ptr_          = cd_buffer_pool_.acquire();
    ext_trigger_buffer_ptr_ = ext_trigger_buffer_pool_.acquire();
}

void EventFileWriter::Private::set_max_event_cd_buffer_size(size_t size) {
    max_event_cd_buffer_size_ = size;
}

void EventFileWriter::Private::set_max_event_trigger_buffer_size(size_t size) {
    max_event_trigger_buffer_size_ = size;
}

void EventFileWriter::Private::open(const std::filesystem::path &path) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (writer_.is_open_impl()) {
        return;
    }
    path_ = path;
    writer_thread_.start();
    writer_.open_impl(path);
}

void EventFileWriter::Private::close() {
    flush();

    std::unique_lock<std::mutex> lock(mutex_);
    writer_thread_.stop();
    writer_.close_impl();
}

bool EventFileWriter::Private::is_open() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return writer_.is_open_impl();
}

void EventFileWriter::Private::flush() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (cd_buffer_ptr_ && !cd_buffer_ptr_->empty()) {
        std::atomic<bool> done{false};
        writer_thread_.add_task([&done, this] {
            writer_.add_events_impl(cd_buffer_ptr_->data(), cd_buffer_ptr_->data() + cd_buffer_ptr_->size());
            done = true;
        });
        while (!done) {
            std::this_thread::yield();
        }
        cd_buffer_ptr_->clear();
    }
    if (ext_trigger_buffer_ptr_ && !ext_trigger_buffer_ptr_->empty()) {
        std::atomic<bool> done{false};
        writer_thread_.add_task([&done, this] {
            writer_.add_events_impl(ext_trigger_buffer_ptr_->data(),
                                    ext_trigger_buffer_ptr_->data() + ext_trigger_buffer_ptr_->size());
            done = true;
        });
        while (!done) {
            std::this_thread::yield();
        }
        ext_trigger_buffer_ptr_->clear();
    }
    writer_.flush_impl();
}

const std::filesystem::path &EventFileWriter::Private::get_path() const {
    return path_;
}

bool EventFileWriter::Private::add_events(const EventCD *begin, const EventCD *end) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!writer_.is_open_impl()) {
        return false;
    }
    if (begin == end) {
        return false;
    }
    if (std::prev(end)->t < begin->t) {
        throw std::runtime_error("Invalid event buffer, the events are not chronologically ordered");
    }
    if (begin->t < last_cd_ts_) {
        throw std::runtime_error("Invalid event buffer, the events are older than last added buffer of events");
    }
    last_cd_ts_ = std::prev(end)->t;

    cd_buffer_ptr_->insert(cd_buffer_ptr_->end(), begin, end);
    if (cd_buffer_ptr_->size() > max_event_cd_buffer_size_) {
        writer_thread_.add_task(
            [this, buf = cd_buffer_ptr_]() { writer_.add_events_impl(buf->data(), buf->data() + buf->size()); });
        cd_buffer_ptr_ = cd_buffer_pool_.acquire();
        cd_buffer_ptr_->clear();
    }

    return true;
}

bool EventFileWriter::Private::add_events(const EventExtTrigger *begin, const EventExtTrigger *end) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!writer_.is_open_impl()) {
        return false;
    }
    if (begin == end) {
        return false;
    }
    if (std::prev(end)->t < begin->t) {
        throw std::runtime_error("Invalid event buffer, the events are not chronologically ordered");
    }
    if (begin->t < last_ext_trigger_ts_) {
        throw std::runtime_error("Invalid event buffer, the events are older than last added buffer of events");
    }
    last_ext_trigger_ts_ = std::prev(end)->t;

    ext_trigger_buffer_ptr_->insert(ext_trigger_buffer_ptr_->end(), begin, end);
    if (ext_trigger_buffer_ptr_->size() > max_event_trigger_buffer_size_) {
        writer_thread_.add_task([this, buf = ext_trigger_buffer_ptr_]() {
            writer_.add_events_impl(buf->data(), buf->data() + buf->size());
        });
        ext_trigger_buffer_ptr_ = ext_trigger_buffer_pool_.acquire();
        ext_trigger_buffer_ptr_->clear();
    }

    return true;
}

void EventFileWriter::Private::add_metadata(const std::string &key, const std::string &value) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!writer_.is_open_impl()) {
        return;
    }
    writer_thread_.add_task([this, key, value] { writer_.add_metadata_impl(key, value); });
}

void EventFileWriter::Private::add_metadata_map_from_camera(const Camera &camera) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!writer_.is_open_impl()) {
        return;
    }

    const unsigned short width                                = camera.geometry().get_width();
    const unsigned short height                               = camera.geometry().get_height();
    auto &config                                              = camera.get_camera_configuration();
    std::unordered_map<std::string, std::string> metadata_map = {
        {"plugin_integrator_name", config.integrator},
        {"camera_integrator_name", config.integrator},
        {"format", config.data_encoding_format},
        {"serial_number", config.serial_number},
        {"generation", std::to_string(camera.generation().version_major()) + "." +
                           std::to_string(camera.generation().version_minor())},
        {"geometry", std::to_string(width) + "x" + std::to_string(height)}};

    std::atomic<bool> done{false};
    writer_thread_.add_task([this, metadata_map, &camera, &done] {
        for (auto &p : metadata_map) {
            writer_.add_metadata_impl(p.first, p.second);
        }
        writer_.add_metadata_map_from_camera_impl(camera);
        done = true;
    });
    // we add a task to synchronize accesses to the writer_ on the writer thread, but we actually
    // don't want the task to be run asynchroneously because the camera (ref) variable will be invalid
    // outside this scope, so we wait for the task to be completed before exiting the function
    while (!done) {
        std::this_thread::yield();
    }
}

void EventFileWriter::Private::remove_metadata(const std::string &key) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!writer_.is_open_impl()) {
        return;
    }
    writer_thread_.add_task([this, key] { writer_.remove_metadata_impl(key); });
}

EventFileWriter::EventFileWriter(const std::filesystem::path &path) : pimpl_(new Private(*this, path)) {}

EventFileWriter::~EventFileWriter() {}

void EventFileWriter::open(const std::filesystem::path &path) {
    pimpl_->open(path);
}

void EventFileWriter::close() {
    pimpl_->close();
}

bool EventFileWriter::is_open() const {
    return pimpl_->is_open();
}

void EventFileWriter::flush() {
    pimpl_->flush();
}

const std::filesystem::path &EventFileWriter::get_path() const {
    return pimpl_->get_path();
}

bool EventFileWriter::add_events(const EventCD *begin, const EventCD *end) {
    return pimpl_->add_events(begin, end);
}

bool EventFileWriter::add_events(const EventExtTrigger *begin, const EventExtTrigger *end) {
    return pimpl_->add_events(begin, end);
}

void EventFileWriter::add_metadata(const std::string &key, const std::string &value) {
    pimpl_->add_metadata(key, value);
}

void EventFileWriter::add_metadata_map_from_camera(const Camera &camera) {
    pimpl_->add_metadata_map_from_camera(camera);
}

void EventFileWriter::remove_metadata(const std::string &key) {
    pimpl_->remove_metadata(key);
}

EventFileWriter::Private &EventFileWriter::get_pimpl() {
    return *pimpl_;
}

} // namespace Metavision
