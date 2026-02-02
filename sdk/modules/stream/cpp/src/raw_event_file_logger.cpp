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
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/utils/raw_file_header.h"
#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/internal/event_file_writer_internal.h"
#include "metavision/sdk/stream/raw_event_file_logger.h"

namespace Metavision {

class RAWEventFileLogger::Private {
public:
    Private(RAWEventFileLogger &writer, const std::filesystem::path &path,
            const std::unordered_map<std::string, std::string> &metadata_map) :
        writer_(writer), header_written_(false) {
        if (!path.empty()) {
            open_impl(path);
            for (auto &p : metadata_map) {
                add_metadata_impl(p.first, p.second);
            }
        }
    }

    EventFileWriter::Private &get_parent_pimpl() {
        return static_cast<EventFileWriter &>(writer_).get_pimpl();
    }

    void open_impl(const std::filesystem::path &path) {
        ofs_ = std::ofstream(path, std::ios::binary);
        if (!ofs_.is_open()) {
            throw std::runtime_error("Unable to open " + path.string() + " for writing");
        }
        raw_data_buffer_ptr_ = raw_data_buffer_pool_.acquire();
    }

    void close_impl() {
        if (!header_written_) {
            ofs_ << header_;
            header_written_ = true;
        }
        ofs_.close();
    }

    bool is_open_impl() const {
        return ofs_.is_open();
    }

    void flush_impl() {
        if (raw_data_buffer_ptr_ && !raw_data_buffer_ptr_->empty()) {
            std::atomic<bool> done{false};
            get_parent_pimpl().writer_thread_.add_task([&done, this] {
                add_raw_data_impl(raw_data_buffer_ptr_->data(), raw_data_buffer_ptr_->size());
                done = true;
            });
            while (!done) {
                std::this_thread::yield();
            }
            raw_data_buffer_ptr_->clear();
        }
        ofs_.flush();
    }

    void add_metadata_impl(const std::string &key, const std::string &value) {
        if (header_written_) {
            throw std::runtime_error("Unable to modify metadata in RAW once data has been added");
        }
        header_.set_field(key, value);
    }

    void add_metadata_map_from_camera_impl(const Camera &camera) {
        if (header_written_) {
            throw std::runtime_error("Unable to modify metadata in RAW once data has been added");
        }
        RawFileHeader header;
        try {
            auto hw_id = camera.get_device().get_facility<I_HW_Identification>();
            if (hw_id) {
                header = hw_id->get_header();
            }
        } catch (...) {}
        for (auto &p : header.get_header_map()) {
            header_.set_field(p.first, p.second);
        }
        header_.add_date();
    }

    void remove_metadata_impl(const std::string &key) {
        if (header_written_) {
            throw std::runtime_error("Unable to modify metadata in RAW once data has been added");
        }
        header_.remove_field(key);
    }

    void add_raw_data_impl(const std::uint8_t *ptr, size_t size) {
        if (!header_written_) {
            ofs_ << header_;
            header_written_ = true;
        }
        ofs_.write(reinterpret_cast<const char *>(ptr), size);
    }

    bool add_raw_data(const std::uint8_t *ptr, size_t size) {
        std::unique_lock<std::mutex> lock(get_parent_pimpl().mutex_);
        if (!is_open_impl()) {
            return false;
        }

        raw_data_buffer_ptr_->insert(raw_data_buffer_ptr_->end(), ptr, ptr + size);
        if (raw_data_buffer_ptr_->size() > kMaxRawDataBufferSize) {
            writer_.get_pimpl().writer_thread_.add_task(
                [this, buf = raw_data_buffer_ptr_] { add_raw_data_impl(buf->data(), buf->size()); });
            raw_data_buffer_ptr_ = raw_data_buffer_pool_.acquire();
            raw_data_buffer_ptr_->clear();
        }

        return true;
    }

    RAWEventFileLogger &writer_;
    RawFileHeader header_;
    std::ofstream ofs_;
    bool header_written_;

    static constexpr size_t kMaxRawDataBufferSize = 1048576;
    using RawDataBufferPool                       = SharedObjectPool<std::vector<std::uint8_t>>;
    using RawDataBufferPtr                        = RawDataBufferPool::ptr_type;
    RawDataBufferPool raw_data_buffer_pool_;
    RawDataBufferPtr raw_data_buffer_ptr_;
};

RAWEventFileLogger::RAWEventFileLogger(const std::filesystem::path &path,
                                       const std::unordered_map<std::string, std::string> &metadata_map) :
    EventFileWriter(path), pimpl_(new Private(*this, path, metadata_map)) {}

RAWEventFileLogger::~RAWEventFileLogger() {
    close();
}

void RAWEventFileLogger::open_impl(const std::filesystem::path &path) {
    pimpl_->open_impl(path);
}

void RAWEventFileLogger::close_impl() {
    pimpl_->close_impl();
}

bool RAWEventFileLogger::is_open_impl() const {
    return pimpl_->is_open_impl();
}

void RAWEventFileLogger::add_metadata_impl(const std::string &key, const std::string &value) {
    pimpl_->add_metadata_impl(key, value);
}

void RAWEventFileLogger::add_metadata_map_from_camera_impl(const Camera &camera) {
    pimpl_->add_metadata_map_from_camera_impl(camera);
}

void RAWEventFileLogger::remove_metadata_impl(const std::string &key) {
    pimpl_->remove_metadata_impl(key);
}

bool RAWEventFileLogger::add_events_impl(const EventCD *begin, const EventCD *end) {
    throw std::runtime_error("RAW does not support writing decoded events");
}

bool RAWEventFileLogger::add_events_impl(const EventExtTrigger *begin, const EventExtTrigger *end) {
    throw std::runtime_error("RAW does not support writing decoded events");
}

bool RAWEventFileLogger::add_raw_data(const std::uint8_t *ptr, size_t size) {
    return pimpl_->add_raw_data(ptr, size);
}

void RAWEventFileLogger::flush_impl() {
    return pimpl_->flush_impl();
}

} // namespace Metavision
