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
#include <limits>
#include <sstream>
#ifdef HAS_HDF5
#include <H5Cpp.h>
#include <hdf5_ecf/ecf_codec.h>
#endif
#include "metavision/sdk/stream/hdf5_event_file_reader.h"

namespace Metavision {
#ifdef HAS_HDF5
struct Index {
    Index(size_t id = 0, std::int64_t ts = 0) : id(id), ts(ts) {}
    size_t id;
    std::int64_t ts;
};

template<typename T, typename Node>
T readAttr(Node &node, const std::string &key) {
    std::string value;
    const auto &attr = node.openAttribute(key);
    H5::StrType meta_type(0, H5T_VARIABLE);
    attr.read(meta_type, value);
    std::istringstream iss(value);
    T t;
    iss >> t;
    return t;
}

template<class EventType>
class EventsReader {
public:
    using DecodingCallbackType = std::function<size_t(const std::uint8_t *, const std::uint8_t *, EventType *)>;

    EventsReader() : chunk_size_(0), pos_(0), offset_(0), index_(0), num_events_(0), timeshift_(0) {}

    EventsReader(H5::DataSet dset, const DecodingCallbackType &decoding_cb = DecodingCallbackType(),
                 const timestamp &timeshift = 0) :
        EventsReader() {
        dset_        = dset;
        decoding_cb_ = decoding_cb;
        timeshift_   = timeshift;

        hsize_t dims[1];
        dset_.getSpace().getSimpleExtentDims(dims);
        num_events_ = dims[0];
        auto plist  = dset_.getCreatePlist();
        plist.getChunk(1, dims);
        chunk_size_ = dims[0];
        pos_        = chunk_size_;
        events_.resize(chunk_size_);
        read_next_chunk();
    }

    size_t count() const {
        return num_events_;
    }

    size_t index() const {
        return index_;
    }

    timestamp get_first_ts() {
        EventType event;
        timestamp ts = -1;
        auto idx     = index_;
        if (set_index(0)) {
            if (this->operator()(event)) {
                ts = event.t;
            }
        }
        set_index(idx);
        return ts;
    }

    timestamp get_last_ts() {
        EventType event;
        timestamp ts = -1;
        auto idx     = index_;
        if (set_index(num_events_ - 1)) {
            if (this->operator()(event)) {
                ts = event.t;
            }
        }
        set_index(idx);
        return ts;
    }

    timestamp time() const {
        return (index_ < num_events_ ? events_[pos_].t : -1);
    }

    bool set_index(size_t index) {
        size_t offset = offset_;
        offset_       = (index / chunk_size_) * chunk_size_;
        if (!read_next_chunk()) {
            offset_ = offset;
            return false;
        }
        pos_   = index % chunk_size_;
        index_ = index;
        return true;
    }

    bool done() const {
        return pos_ >= events_.size() && offset_ >= num_events_;
    }

    bool operator()(EventType &ev) {
        if (pos_ >= events_.size()) {
            if (!read_next_chunk()) {
                return false;
            }
        }
        ev = events_[pos_];
        ++index_;
        ++pos_;
        return true;
    }

    size_t operator()(timestamp t, EventType *&ptr) {
        if (pos_ >= events_.size()) {
            if (!read_next_chunk()) {
                return 0;
            }
        }
        ptr        = events_.data() + pos_;
        auto end   = std::lower_bound(ptr, events_.data() + events_.size(), t,
                                    [](const auto &ev, const timestamp &t) { return ev.t < t; });
        size_t ret = std::distance(ptr, end);
        index_ += ret;
        pos_ += ret;
        return ret;
    }

private:
    bool read_next_chunk() {
        if (offset_ >= num_events_) {
            return false;
        }
        std::uint32_t filters = 0;
        hsize_t offset[1]     = {offset_};
        hsize_t compressed_size;
        H5Dget_chunk_storage_size(dset_.getId(), offset, &compressed_size);
        if (decoding_cb_) {
            inbuf_.resize(compressed_size);
            if (H5Dread_chunk(dset_.getId(), H5P_DEFAULT, offset, &filters, inbuf_.data()) < 0) {
                return false;
            }
            // Make sure we have enough space to decode events, we will resize to correct size after decoding
            events_.resize(chunk_size_);
            size_t num_bytes = decoding_cb_(inbuf_.data(), inbuf_.data() + compressed_size, events_.data());
            events_.resize(std::min(num_bytes / sizeof(Metavision::EventCD), num_events_ - offset_));
        } else {
            events_.resize(std::min(chunk_size_, num_events_ - offset_));
            if (H5Dread_chunk(dset_.getId(), H5P_DEFAULT, offset, &filters, events_.data()) < 0) {
                return false;
            }
        }
        if (timeshift_ > 0) {
            for (auto &ev : events_) {
                ev.t -= timeshift_;
            }
        }
        pos_ = 0;
        offset_ += chunk_size_;
        return true;
    }

    H5::DataSet dset_;
    size_t chunk_size_;
    size_t pos_, offset_, index_, num_events_;
    std::vector<std::uint8_t> inbuf_;
    std::vector<EventType> events_;
    DecodingCallbackType decoding_cb_;
    timestamp timeshift_;
};

class IndexesReader {
public:
    IndexesReader() : chunk_size_(0), pos_(0), offset_(0), index_(0), num_indexes_(0), ts_offset_(0) {}

    IndexesReader(H5::DataSet dset) : IndexesReader() {
        dset_ = dset;
        hsize_t dims[1];
        dset_.getSpace().getSimpleExtentDims(dims);
        num_indexes_ = dims[0];
        auto plist   = dset_.getCreatePlist();
        plist.getChunk(1, dims);
        chunk_size_ = dims[0];
        pos_        = chunk_size_;
        indexes_.resize(chunk_size_);

        std::string key = "offset";
        if (dset_.attrExists(key)) {
            ts_offset_ = readAttr<timestamp>(dset_, key);
        }
    }

    size_t count() const {
        return num_indexes_;
    }

    size_t index() const {
        return index_;
    }

    std::int64_t get_first_ts() {
        Index index;
        timestamp ts = -1;
        auto idx     = index_;
        if (set_index(0)) {
            while (this->operator()(index) && index.ts < 0) {}
            ts = index.ts;
        }
        set_index(idx);
        return ts;
    }

    std::int64_t get_last_ts() {
        Index index;
        timestamp ts = -1;
        auto idx     = index_;
        if (set_index(num_indexes_ - 1)) {
            if (this->operator()(index)) {
                ts = index.ts;
            }
        }
        set_index(idx);
        return ts;
    }

    bool done() const {
        return pos_ >= indexes_.size() && offset_ >= num_indexes_;
    }

    bool operator()(Index &index) {
        if (pos_ >= indexes_.size()) {
            if (!read_next_chunk()) {
                return false;
            }
        }
        index = indexes_[pos_];
        ++pos_;
        ++index_;
        return true;
    }

    bool seek(const timestamp &t) {
        size_t index = (t + ts_offset_) / kIndexesPeriodUs;
        if (t < 0 || index >= num_indexes_) {
            return false;
        }
        return set_index(index);
    }

private:
    bool set_index(size_t index) {
        size_t offset = offset_;
        offset_       = (index / chunk_size_) * chunk_size_;
        if (!read_next_chunk()) {
            offset_ = offset;
            return false;
        }
        pos_   = index % chunk_size_;
        index_ = index;
        return true;
    }

    bool read_next_chunk() {
        if (offset_ >= num_indexes_) {
            return false;
        }
        hsize_t offset[1] = {offset_};
        std::uint32_t filters;
        if (H5Dread_chunk(dset_.getId(), H5P_DEFAULT, offset, &filters, indexes_.data()) < 0) {
            return false;
        }
        const size_t num_indexes = std::min(num_indexes_ - offset_, chunk_size_);
        indexes_.resize(num_indexes);
        for (auto &index : indexes_) {
            index.ts += ts_offset_;
        }
        pos_ = 0;
        offset_ += chunk_size_;
        return true;
    }

    static constexpr std::uint32_t kIndexesPeriodUs = 2000;
    H5::DataSet dset_;
    size_t chunk_size_;
    size_t pos_, offset_, index_, num_indexes_;
    timestamp ts_offset_;
    std::vector<Index> indexes_;
};
#endif

class HDF5EventFileReader::Private {
public:
    Private(HDF5EventFileReader &reader, const std::filesystem::path &path, bool time_shift) :
        timeshift_(0), reader_(reader) {
#ifdef HAS_HDF5
        file_ = H5::H5File(path.string(), H5F_ACC_RDONLY);

        auto root = file_.openGroup("/");
        if (time_shift) {
            std::string key = "time_shift";
            if (root.attrExists(key)) {
                timeshift_ = readAttr<timestamp>(root, key);
            }
        }

        auto cd_events_dset = file_.openDataSet("/CD/events");
        cd_events_reader_   = EventsReader<EventCD>(
            cd_events_dset,
            [this](const std::uint8_t *begin, const std::uint8_t *end, EventCD *ptr) {
                return cd_events_decoder_(begin, end, reinterpret_cast<ECF::EventCD *>(ptr));
            },
            timeshift_);

        auto cd_indexes_dset = file_.openDataSet("/CD/indexes");
        cd_indexes_reader_   = IndexesReader(cd_indexes_dset);

        auto ext_trigger_events_dset = file_.openDataSet("/EXT_TRIGGER/events");
        ext_trigger_events_reader_   = EventsReader<EventExtTrigger>(ext_trigger_events_dset, {}, timeshift_);

        auto ext_trigger_indexes_dset = file_.openDataSet("/EXT_TRIGGER/indexes");
        ext_trigger_indexes_reader_   = IndexesReader(ext_trigger_indexes_dset);
#else
        throw std::runtime_error("HDF5 is not available");
#endif
    }

    bool read_impl() {
#ifdef HAS_HDF5
        const bool cd_events_done          = cd_events_reader_.done();
        const bool ext_trigger_events_done = ext_trigger_events_reader_.done();
        if (cd_events_done && ext_trigger_events_done) {
            return false;
        }

        next_time_ += kTimeStepUs;

        auto cd_f = [this]() {
            EventCD *ptr;
            size_t num;
            while ((num = cd_events_reader_(next_time_, ptr)) > 0) {
                if (reader_.has_read_callbacks()) {
                    reader_.notify_events_buffer(ptr, ptr + num);
                }
            }
        };

        auto ext_trigger_f = [this]() {
            EventExtTrigger *ptr;
            size_t num;
            while ((num = ext_trigger_events_reader_(next_time_, ptr)) > 0) {
                if (reader_.has_read_callbacks()) {
                    reader_.notify_events_buffer(ptr, ptr + num);
                }
            }
        };

        if (!cd_events_done && !ext_trigger_events_done) {
            if (cd_events_reader_.time() < ext_trigger_events_reader_.time()) {
                cd_f();
                ext_trigger_f();
            } else {
                ext_trigger_f();
                cd_f();
            }
        } else if (!cd_events_done) {
            cd_f();
        } else {
            ext_trigger_f();
        }
        return true;
#else
        return false;
#endif
    }

    bool seekable() const {
        return true;
    }

    bool get_seek_range_impl(timestamp &min_t, timestamp &max_t) const {
        min_t = -1;
        max_t = -1;
#ifdef HAS_HDF5
        timestamp first_cd_ts      = cd_indexes_reader_.get_first_ts();
        timestamp first_trigger_ts = ext_trigger_indexes_reader_.get_first_ts();
        min_t                      = (first_cd_ts >= 0 ? first_cd_ts : min_t);
        min_t                      = (first_trigger_ts >= 0 ? std::min(min_t, first_trigger_ts) : min_t);

        timestamp last_cd_ts      = cd_indexes_reader_.get_last_ts();
        timestamp last_trigger_ts = ext_trigger_indexes_reader_.get_last_ts();
        max_t                     = std::max(last_cd_ts, last_trigger_ts);
        return true;
#else
        return false;
#endif
    }

    timestamp get_duration_impl() const {
        timestamp duration = -1;
#ifdef HAS_HDF5
        auto root       = file_.openGroup("/");
        std::string key = "duration";
        if (root.attrExists(key)) {
            duration = readAttr<timestamp>(root, key);
        } else {
            duration = std::max(cd_events_reader_.get_last_ts(), ext_trigger_events_reader_.get_last_ts());
        }
#endif
        return duration;
    }

    bool seek_impl(timestamp t) {
#ifdef HAS_HDF5
        Index ind(std::numeric_limits<size_t>::max());
        timestamp ts = -1;

        if (cd_indexes_reader_.seek(t)) {
            if (cd_indexes_reader_(ind)) {
                if (cd_events_reader_.set_index(ind.id)) {
                    ts = ind.ts;
                }
            }
        }

        if (ext_trigger_indexes_reader_.seek(t)) {
            if (ext_trigger_indexes_reader_(ind)) {
                if (ext_trigger_events_reader_.set_index(ind.id)) {
                    ts = std::min<timestamp>(ts, ind.ts);
                }
            }
        }

        if (ind.id == std::numeric_limits<size_t>::max()) {
            return false;
        }

        next_time_ = (ts / kTimeStepUs) * kTimeStepUs;
        reader_.notify_seek(ts);

        return true;
#else
        return false;
#endif
    }

    std::unordered_map<std::string, std::string> get_metadata_map_impl() const {
        std::unordered_map<std::string, std::string> metadata_map;
#ifdef HAS_HDF5
        auto root = file_.openGroup("/");
        root.iterateAttrs(
            [](H5::H5Object &object, H5std_string name, void *data) {
                std::string value;
                const auto &attr = object.openAttribute(name);
                H5::StrType meta_type(0, H5T_VARIABLE);
                attr.read(meta_type, value);
                reinterpret_cast<std::unordered_map<std::string, std::string> *>(data)->insert(
                    std::make_pair(name, value));
            },
            nullptr, &metadata_map);
#endif
        return metadata_map;
    }

    static constexpr timestamp kTimeStepUs = 1000;
    timestamp next_time_                   = kTimeStepUs;
    timestamp timeshift_;
#ifdef HAS_HDF5
    H5::H5File file_;
    mutable ECF::Decoder cd_events_decoder_;
    mutable EventsReader<EventCD> cd_events_reader_;
    mutable IndexesReader cd_indexes_reader_;
    mutable EventsReader<EventExtTrigger> ext_trigger_events_reader_;
    mutable IndexesReader ext_trigger_indexes_reader_;
#endif
    HDF5EventFileReader &reader_;
};

HDF5EventFileReader::HDF5EventFileReader(const std::filesystem::path &path, bool time_shift) :
    EventFileReader(path), pimpl_(new Private(*this, path, time_shift)) {}

HDF5EventFileReader::~HDF5EventFileReader() {}

bool HDF5EventFileReader::seekable() const {
    return pimpl_->seekable();
}

bool HDF5EventFileReader::read_impl() {
    return pimpl_->read_impl();
}

bool HDF5EventFileReader::seek_impl(timestamp t) {
    return pimpl_->seek_impl(t);
}

bool HDF5EventFileReader::get_seek_range_impl(timestamp &min_t, timestamp &max_t) const {
    return pimpl_->get_seek_range_impl(min_t, max_t);
}

timestamp HDF5EventFileReader::get_duration_impl() const {
    return pimpl_->get_duration_impl();
}

std::unordered_map<std::string, std::string> HDF5EventFileReader::get_metadata_map_impl() const {
    return pimpl_->get_metadata_map_impl();
}

} // namespace Metavision
