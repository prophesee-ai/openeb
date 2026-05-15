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

#ifdef HAS_HDF5
#include <H5Cpp.h>
#include <hdf5_ecf/ecf_codec.h>
#include <hdf5_ecf/ecf_h5filter.h>
#endif
#include <string>
#include <unordered_map>

#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/hdf5_event_file_writer.h"

namespace Metavision {
#ifdef HAS_HDF5
struct Index {
    Index(size_t id = 0, std::int64_t ts = 0) : id(id), ts(ts) {}
    size_t id;
    std::int64_t ts;
};

template<typename Node>
void writeAttr(Node &node, const std::string &key, const std::string &value) {
    H5::StrType meta_type(0, H5T_VARIABLE);
    H5::DataSpace meta_space(H5S_SCALAR);
    H5::Attribute attr;
    if (node.attrExists(key)) {
        attr = node.openAttribute(key);
    } else {
        attr = node.createAttribute(key, meta_type, meta_space);
    }
    attr.write(meta_type, value);
}

template<typename EventType>
class EventsWriter {
public:
    using EncodingCallbackType = std::function<size_t(const EventType *, const EventType *, std::uint8_t *)>;

    EventsWriter() : pos_(0), offset_(0), chunk_size_(0) {}

    EventsWriter(H5::DataSet dset, size_t chunk_size, size_t max_outbuf_size,
                 const EncodingCallbackType &encoding_cb = EncodingCallbackType()) :
        EventsWriter() {
        dset_       = dset;
        chunk_size_ = chunk_size;
        events_.resize(chunk_size);
        encoding_cb_ = encoding_cb;
        outbuf_.resize(max_outbuf_size);
    }

    ~EventsWriter() {
        try {
            close();
        } catch (std::runtime_error &) {
            std::cerr << "Error writing HDF5 file" << std::endl;
            dset_.close();
        }
    }

    void close() {
        if (pos_ > 0) {
            if (!sync()) {
                throw std::runtime_error("Error writing HDF5 file");
            }
            pos_ = 0;
        }
        dset_.close();
    }

    bool sync() {
        hsize_t offset[1] = {offset_};
        size_t num_bytes_in_chunk;

        if (encoding_cb_) {
            hsize_t dims[1] = {offset_ + pos_};
            dset_.extend(dims);

            num_bytes_in_chunk = encoding_cb_(events_.data(), events_.data() + pos_, outbuf_.data());
            if (H5Dwrite_chunk(dset_.getId(), H5P_DEFAULT, 0, offset, num_bytes_in_chunk, outbuf_.data()) < 0) {
                return false;
            }
        } else {
            hsize_t dims[1] = {offset_ + pos_};
            dset_.extend(dims);

            num_bytes_in_chunk = chunk_size_ * sizeof(EventType);
            if (H5Dwrite_chunk(dset_.getId(), H5P_DEFAULT, 0, offset, num_bytes_in_chunk, events_.data()) < 0) {
                return false;
            }
        }
        return true;
    }

    bool operator()(const EventType *begin, const EventType *end) {
        size_t num_events_to_consume = std::distance(begin, end);
        size_t num_events_to_copy    = std::min(chunk_size_ - pos_, num_events_to_consume);
        auto *ev_ptr                 = begin;
        while (num_events_to_consume > 0) {
            std::copy(ev_ptr, ev_ptr + num_events_to_copy, events_.data() + pos_);
            pos_ += num_events_to_copy;
            if (pos_ == chunk_size_) {
                if (!sync()) {
                    return false;
                }
                offset_ += pos_;
                pos_ = 0;
            }
            num_events_to_consume -= num_events_to_copy;
            ev_ptr += num_events_to_copy;
            num_events_to_copy = std::min(chunk_size_, num_events_to_consume);
        }
        return true;
    }

    H5::DataSet dset_;
    size_t pos_, offset_, chunk_size_;
    EncodingCallbackType encoding_cb_;
    std::vector<EventType> events_;
    std::vector<std::uint8_t> outbuf_;
};

template<typename EventType>
class IndexesWriter {
public:
    IndexesWriter() :
        pos_(0),
        offset_(0),
        count_(0),
        chunk_size_(0),
        index_ts_(-1),
        last_index_ts_(-1),
        ts_offset_(0),
        ev_id_(0),
        index_id_(0),
        last_index_id_(0) {}

    IndexesWriter(H5::DataSet dset, size_t chunk_size) : IndexesWriter() {
        dset_       = dset;
        chunk_size_ = chunk_size;
        indexes_.resize(chunk_size_);
    }

    ~IndexesWriter() {
        try {
            close();
        } catch (std::runtime_error &) {
            std::cerr << "Error writing HDF5 file" << std::endl;
            dset_.close();
        }
    }

    void close() {
        if (count_ > 0) {
            indexes_[pos_].id = index_id_;
            indexes_[pos_].ts = index_ts_;
            ++count_;
            ++pos_;
            if (!sync()) {
                throw std::runtime_error("Error writing HDF5 file");
            }
            count_ = 0;
        }
        dset_.close();
    }

    bool sync() {
        hsize_t dims[1] = {offset_ + pos_};
        dset_.extend(dims);
        hsize_t offset[1]         = {offset_};
        size_t num_bytes_in_chunk = chunk_size_ * sizeof(Index);

        auto error_code = H5Dwrite_chunk(dset_.getId(), H5P_DEFAULT, 0, offset, num_bytes_in_chunk, indexes_.data());
        if (error_code < 0) {
            return false;
        }
        return true;
    }

    bool operator()(const EventType *begin, const EventType *end) {
        if (count_ == 0) {
            ts_offset_ = -begin->t;
            writeAttr(dset_, "offset", std::to_string(ts_offset_));
        }
        for (auto *ev = begin; ev != end;) {
            timestamp t    = ev->t + ts_offset_;
            auto index_pos = static_cast<std::size_t>(t) / kIndexesPeriodUs + 1;
            if (index_pos > count_) {
                if (ev_id_ != index_id_) {
                    last_index_id_ = index_id_;
                    last_index_ts_ = index_ts_;
                }
                indexes_[pos_].id = last_index_id_;
                indexes_[pos_].ts = last_index_ts_;
                ++count_;
                ++pos_;
                if (pos_ == chunk_size_) {
                    if (!sync()) {
                        return false;
                    }
                    offset_ += pos_;
                    pos_ = 0;
                }
                index_id_ = ev_id_;
                index_ts_ = t;
            } else {
                ++ev_id_;
                ++ev;
            }
        }
        return true;
    }

    static constexpr std::uint32_t kIndexesPeriodUs = 2000;
    H5::DataSet dset_;
    size_t offset_, pos_, count_, chunk_size_;
    std::vector<Index> indexes_;
    timestamp index_ts_, last_index_ts_;
    timestamp ts_offset_;
    size_t ev_id_, index_id_, last_index_id_;
};
#endif

class HDF5EventFileWriter::Private {
public:
    Private(HDF5EventFileWriter &writer, const std::filesystem::path &path,
            const std::unordered_map<std::string, std::string> &metadata_map) :
        writer_(writer) {
        if (!path.empty()) {
            open_impl(path);
            for (auto &p : metadata_map) {
                add_metadata_impl(p.first, p.second);
            }
        }
    }

    void open_impl(const std::filesystem::path &path) {
#ifdef HAS_HDF5
        hsize_t dims[1] = {0}, maxdims[1] = {H5S_UNLIMITED};
        hsize_t chunk_dims[1] = {kChunkSize};

        H5::DataSpace cd_event_ds(1, dims, maxdims);
        H5::CompType cd_event_dt(sizeof(Metavision::EventCD));
        cd_event_dt.insertMember("x", HOFFSET(Metavision::EventCD, x), H5::PredType::NATIVE_USHORT);
        cd_event_dt.insertMember("y", HOFFSET(Metavision::EventCD, y), H5::PredType::NATIVE_USHORT);
        cd_event_dt.insertMember("p", HOFFSET(Metavision::EventCD, p), H5::PredType::NATIVE_SHORT);
        cd_event_dt.insertMember("t", HOFFSET(Metavision::EventCD, t), H5::PredType::NATIVE_LLONG);
        H5::DSetCreatPropList cd_event_ds_prop;
        cd_event_ds_prop.setChunk(1, chunk_dims);
        cd_event_ds_prop.setFilter(H5Z_FILTER_ECF, H5Z_FLAG_OPTIONAL, 0, nullptr);

        H5::DataSpace cd_index_ds(1, dims, maxdims);
        H5::CompType cd_index_dt(sizeof(Index));
        cd_index_dt.insertMember("id", HOFFSET(Index, id), H5::PredType::NATIVE_ULLONG);
        cd_index_dt.insertMember("ts", HOFFSET(Index, ts), H5::PredType::NATIVE_LLONG);
        H5::DSetCreatPropList cd_index_ds_prop;
        cd_index_ds_prop.setChunk(1, chunk_dims);

        file_ = H5::H5File(path.string(), H5F_ACC_TRUNC);
        file_.createGroup("/CD");
        H5::DataSet cd_events_dset  = file_.createDataSet("/CD/events", cd_event_dt, cd_event_ds, cd_event_ds_prop);
        H5::DataSet cd_indexes_dset = file_.createDataSet("/CD/indexes", cd_index_dt, cd_index_ds, cd_index_ds_prop);

        cd_events_writer_ = EventsWriter<Metavision::EventCD>(
            cd_events_dset, kChunkSize, encoder_.getCompressedSize(),
            [this](const Metavision::EventCD *begin, const Metavision::EventCD *end, std::uint8_t *ptr) {
                return encoder_(reinterpret_cast<const ECF::EventCD *>(begin),
                                reinterpret_cast<const ECF::EventCD *>(end), ptr);
            });
        cd_indexes_writer_ = IndexesWriter<Metavision::EventCD>(cd_indexes_dset, kChunkSize);

        H5::DataSpace ext_trigger_event_ds(1, dims, maxdims);
        H5::CompType ext_trigger_event_dt(sizeof(Metavision::EventExtTrigger));
        ext_trigger_event_dt.insertMember("p", HOFFSET(Metavision::EventExtTrigger, p), H5::PredType::NATIVE_SHORT);
        ext_trigger_event_dt.insertMember("t", HOFFSET(Metavision::EventExtTrigger, t), H5::PredType::NATIVE_LLONG);
        ext_trigger_event_dt.insertMember("id", HOFFSET(Metavision::EventExtTrigger, id), H5::PredType::NATIVE_SHORT);
        H5::DSetCreatPropList ext_trigger_event_ds_prop;
        ext_trigger_event_ds_prop.setChunk(1, chunk_dims);

        H5::CompType ext_trigger_index_dt(sizeof(Index));
        ext_trigger_index_dt.insertMember("id", HOFFSET(Index, id), H5::PredType::NATIVE_ULLONG);
        ext_trigger_index_dt.insertMember("ts", HOFFSET(Index, ts), H5::PredType::NATIVE_LLONG);
        H5::DataSpace ext_trigger_index_ds(1, dims, maxdims);
        H5::DSetCreatPropList ext_trigger_index_ds_prop;
        ext_trigger_index_ds_prop.setChunk(1, chunk_dims);

        file_.createGroup("/EXT_TRIGGER");
        H5::DataSet ext_trigger_events_dset  = file_.createDataSet("/EXT_TRIGGER/events", ext_trigger_event_dt,
                                                                  ext_trigger_event_ds, ext_trigger_event_ds_prop);
        H5::DataSet ext_trigger_indexes_dset = file_.createDataSet("/EXT_TRIGGER/indexes", ext_trigger_index_dt,
                                                                   ext_trigger_index_ds, ext_trigger_index_ds_prop);

        ext_trigger_events_writer_ = EventsWriter<Metavision::EventExtTrigger>(
            ext_trigger_events_dset, kChunkSize, kChunkSize * sizeof(Metavision::EventExtTrigger));
        ext_trigger_indexes_writer_ = IndexesWriter<Metavision::EventExtTrigger>(ext_trigger_indexes_dset, kChunkSize);

        add_metadata_impl("version", "1.0");
#else
        throw std::runtime_error("HDF5 is not available");
#endif
    }

    void close_impl() {
#ifdef HAS_HDF5
        ext_trigger_indexes_writer_.close();
        ext_trigger_events_writer_.close();
        cd_indexes_writer_.close();
        cd_events_writer_.close();
        file_.close();
#endif
    }

    bool is_open_impl() const {
#ifdef HAS_HDF5
        return file_.getId() >= 0;
#endif
        return false;
    }

    void flush_impl() {
#ifdef HAS_HDF5
        if (is_open_impl()) {
            file_.flush(H5F_SCOPE_GLOBAL);
        }
#endif
    }

    void add_metadata_impl(const std::string &key, const std::string &value) {
#ifdef HAS_HDF5
        H5::Group root = file_.openGroup("/");
        writeAttr(root, key, value);
#endif
    }

    void add_metadata_map_from_camera_impl(const Camera &camera) {
        auto metadata_map = camera.get_metadata_map();
        // remove fields that make no sense for HDF5
        metadata_map.erase("evt");
        metadata_map.erase("plugin_name");
        for (auto &p : metadata_map) {
            add_metadata_impl(p.first, p.second);
        }
    }

    void remove_metadata_impl(const std::string &key) {
#ifdef HAS_HDF5
        H5::Group root = file_.openGroup("/");
        if (root.attrExists(key)) {
            root.removeAttr(key);
        }
#endif
    }

    bool add_events_impl(const EventCD *begin, const EventCD *end) {
#ifdef HAS_HDF5
        if (!cd_indexes_writer_(begin, end)) {
            return false;
        }
        if (!cd_events_writer_(begin, end)) {
            return false;
        }
        return true;
#else
        return false;
#endif
    }

    bool add_events_impl(const EventExtTrigger *begin, const EventExtTrigger *end) {
#ifdef HAS_HDF5
        if (!ext_trigger_indexes_writer_(begin, end)) {
            return false;
        }
        if (!ext_trigger_events_writer_(begin, end)) {
            return false;
        }
        return true;
#else
        return false;
#endif
    }

    static constexpr size_t kChunkSize = 16384;
#ifdef HAS_HDF5
    H5::H5File file_;
    ECF::Encoder encoder_;
    EventsWriter<Metavision::EventCD> cd_events_writer_;
    IndexesWriter<Metavision::EventCD> cd_indexes_writer_;
    EventsWriter<Metavision::EventExtTrigger> ext_trigger_events_writer_;
    IndexesWriter<Metavision::EventExtTrigger> ext_trigger_indexes_writer_;
#endif
    HDF5EventFileWriter &writer_;
};

HDF5EventFileWriter::HDF5EventFileWriter(const std::filesystem::path &path,
                                         const std::unordered_map<std::string, std::string> &metadata_map) :
    EventFileWriter(path), pimpl_(new Private(*this, path, metadata_map)) {}

HDF5EventFileWriter::~HDF5EventFileWriter() {
    close();
}

void HDF5EventFileWriter::open_impl(const std::filesystem::path &path) {
    return pimpl_->open_impl(path);
}

void HDF5EventFileWriter::close_impl() {
    if (pimpl_) {
        pimpl_->close_impl();
    }
}

bool HDF5EventFileWriter::is_open_impl() const {
    return pimpl_ && pimpl_->is_open_impl();
}

void HDF5EventFileWriter::flush_impl() {
    return pimpl_->flush_impl();
}

void HDF5EventFileWriter::add_metadata_impl(const std::string &key, const std::string &value) {
    pimpl_->add_metadata_impl(key, value);
}

void HDF5EventFileWriter::add_metadata_map_from_camera_impl(const Camera &camera) {
    pimpl_->add_metadata_map_from_camera_impl(camera);
}

void HDF5EventFileWriter::remove_metadata_impl(const std::string &key) {
    pimpl_->remove_metadata_impl(key);
}

bool HDF5EventFileWriter::add_events_impl(const EventCD *begin, const EventCD *end) {
    return pimpl_->add_events_impl(begin, end);
}

bool HDF5EventFileWriter::add_events_impl(const EventExtTrigger *begin, const EventExtTrigger *end) {
    return pimpl_->add_events_impl(begin, end);
}

} // namespace Metavision
