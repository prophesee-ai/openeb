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

#include <fstream>
#include <memory>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/sdk/base/utils/log.h"
#include "metavision/sdk/base/utils/generic_header.h"
#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/internal/event_file_reader_internal.h"
#include "metavision/sdk/stream/dat_event_file_reader.h"

namespace Metavision {

class DATEventFileReader::Private {
public:
    Private(DATEventFileReader &reader) : reader_(reader), data_stream_(reader.get_path()) {
        metadata_["generation"] = "3.0";

        if (std::filesystem::exists(reader.get_path())) {
            setup_dat_stream(reader.get_path());
        } else {
            bool found_target_files = false;

            if (std::filesystem::exists(reader.get_path().string() + "_cd.dat")) {
                setup_dat_stream(reader.get_path().string() + "_cd.dat");
                found_target_files = true;
            }
            if (std::filesystem::exists(reader.get_path().string() + "_td.dat")) {
                setup_dat_stream(reader.get_path().string() + "_td.dat");
                found_target_files = true;
            }
            if (std::filesystem::exists(reader.get_path().string() + "_trigger.dat")) {
                setup_dat_stream(reader.get_path().string() + "_trigger.dat");
                found_target_files = true;
            }

            if (!found_target_files) {
                throw std::runtime_error("Could not find dat files for \"" + reader.get_path().string() + "\"");
            }
        }
    }

    bool read_impl() {
        bool cd_done       = true;
        bool ext_trig_done = true;

        if (cd_stream_) {
            cd_done = !cd_stream_->read_events();
        }
        if (ext_trig_stream_) {
            ext_trig_done = !ext_trig_stream_->read_events();
        }

        return !cd_done || !ext_trig_done;
    }

    bool get_seek_range_impl(timestamp &min_t, timestamp &max_t) const {
        return false;
    }

    timestamp get_duration_impl() const {
        timestamp duration = -1;

        if (cd_stream_) {
            duration = cd_stream_->get_last_evt_ts();
        }
        if (ext_trig_stream_) {
            duration = std::max(duration, ext_trig_stream_->get_last_evt_ts());
        }

        return duration;
    }

    bool seek_impl(timestamp t) {
        return false;
    }

    bool seekable() {
        return false;
    }

    std::unordered_map<std::string, std::string> get_metadata_map_impl() const {
        return metadata_;
    }

    EventFileReader::Private &get_parent_pimpl() {
        return reader_.EventFileReader::get_pimpl();
    }

    const EventFileReader::Private &get_parent_pimpl() const {
        return reader_.EventFileReader::get_pimpl();
    }

private:
    template<typename Event_Type>
    class DATStream {
    public:
        DATStream(DATEventFileReader &reader, std::ifstream &&ifs, const std::filesystem::path &path) :
            reader_(reader), data_stream_(std::move(ifs)), path_(path) {}

        bool read_events() {
            std::vector<Event_Type> evt_buff;
            do {
                if (pos_ptr_ == end_ptr_) {
                    if (data_stream_.eof()) {
                        break;
                    }
                    data_stream_.read(reinterpret_cast<char *>(raw_events_), kEvtChunk * kEvtSize);
                    pos_ptr_ = raw_events_;
                    end_ptr_ = raw_events_ + data_stream_.gcount() / kEvtSize;
                }

                auto next_ptr =
                    std::lower_bound(pos_ptr_, end_ptr_, next_time_, [](uint64_t &ev, const timestamp &val) {
                        return Event_Type::read_event(&ev).t < val;
                    });

                for (; pos_ptr_ < next_ptr; ++pos_ptr_) {
                    evt_buff.push_back(Event_Type::read_event(pos_ptr_));
                }
            } while (pos_ptr_ == end_ptr_);

            if (!evt_buff.empty() && reader_.has_read_callbacks()) {
                reader_.notify_events_buffer(evt_buff.data(), evt_buff.data() + evt_buff.size());
            }

            next_time_ += kTimeStepUs;

            return !data_stream_.eof() || pos_ptr_ != end_ptr_;
        }

        timestamp get_last_evt_ts() const {
            std::ifstream data(path_, std::ios::binary);

            if (!data) {
                return -1;
            }

            timestamp last_ts = -1;

            if (data.seekg(-8, std::ios::end)) {
                uint64_t ev;
                if (data.read(reinterpret_cast<char *>(&ev), sizeof(ev))) {
                    last_ts = Event_Type::read_event(&ev).t;
                }
            }

            return last_ts;
        }

    private:
        static constexpr size_t kEvtChunk      = 512;
        static constexpr size_t kEvtSize       = sizeof(uint64_t);
        static constexpr timestamp kTimeStepUs = 1000;
        timestamp next_time_                   = kTimeStepUs;
        DATEventFileReader &reader_;
        std::ifstream data_stream_;
        uint64_t raw_events_[kEvtChunk];
        uint64_t *pos_ptr_ = nullptr;
        uint64_t *end_ptr_ = nullptr;
        const std::filesystem::path path_;
    };

    void setup_dat_stream(const std::filesystem::path &path) {
        std::ifstream data(path, std::ios::binary);

        if (!data.is_open()) {
            MV_LOG_ERROR() << "Unable to open DAT file";
            return;
        }

        GenericHeader header(data);

        auto it = metadata_.find("geometry");
        if (it == metadata_.end()) {
            if (header.get_field("Height") != "" && header.get_field("Width") != "") {
                metadata_["geometry"] = header.get_field("Width") + "x" + header.get_field("Height");
            }
        } else {
            if (header.get_field("Height") != "" && header.get_field("Width") != "") {
                if (metadata_["geometry"] != header.get_field("Width") + "x" + header.get_field("Height")) {
                    MV_LOG_ERROR() << "Inconsistent geometries in DAT files.";
                }
            }
        }

        const uint8_t ev_type = data.get();
        const uint8_t ev_size = data.get();

        if (ev_size != 0x8)
            MV_LOG_ERROR() << "Invalid data size: " << ev_size << std::endl;

        if (ev_type == get_event_id<EventCD>() || ev_type == get_event_id<Event2d>()) {
            cd_stream_ = std::make_unique<DATStream<EventCD>>(reader_, std::move(data), path);
        } else if (ev_type == get_event_id<EventExtTrigger>()) {
            ext_trig_stream_ = std::make_unique<DATStream<EventExtTrigger>>(reader_, std::move(data), path);
        } else {
            MV_LOG_ERROR() << "Invalid event type: " << ev_type << std::endl;
        }
    }

    std::ifstream data_stream_;
    std::unordered_map<std::string, std::string> metadata_;
    DATEventFileReader &reader_;
    std::unique_ptr<DATStream<EventCD>> cd_stream_;
    std::unique_ptr<DATStream<EventExtTrigger>> ext_trig_stream_;
};

DATEventFileReader::DATEventFileReader(const std::filesystem::path &path) :
    EventFileReader(path), pimpl_(new Private(*this)) {}

DATEventFileReader::~DATEventFileReader() {}

bool DATEventFileReader::read_impl() {
    return pimpl_->read_impl();
}

bool DATEventFileReader::seek_impl(timestamp t) {
    return pimpl_->seek_impl(t);
}

bool DATEventFileReader::seekable() const {
    return pimpl_->seekable();
}

bool DATEventFileReader::get_seek_range_impl(timestamp &min_t, timestamp &max_t) const {
    return pimpl_->get_seek_range_impl(min_t, max_t);
}

timestamp DATEventFileReader::get_duration_impl() const {
    return pimpl_->get_duration_impl();
}

std::unordered_map<std::string, std::string> DATEventFileReader::get_metadata_map_impl() const {
    return pimpl_->get_metadata_map_impl();
}

} // namespace Metavision
