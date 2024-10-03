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

#include <chrono>
#include <algorithm>
#include <random>
#include <functional>

#include "metavision/hal/device/device.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_hal_software_info.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "metavision/hal/utils/file_raw_data_producer.h"
#include "metavision/hal/utils/hal_connection_exception.h"
#include "metavision/hal/utils/hal_error_code.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

namespace {
template<typename Container>
bool contains(const Container &container, const typename Container::value_type &value) {
    return (std::find(container.begin(), container.end(), value) != container.end());
}

union BookmarkOrMagicNumber {
    I_EventsStream::Bookmark bookmark;
    std::array<std::uint8_t, sizeof(I_EventsStream::Bookmark)> array;

    BookmarkOrMagicNumber() {}
    BookmarkOrMagicNumber(const I_EventsStream::Bookmark &b) : bookmark(b) {}

    static BookmarkOrMagicNumber magic_number() {
        BookmarkOrMagicNumber m;
        std::mt19937 mt(0x6d76); // MV = 0x6d 0x76
        std::generate(m.array.begin(), m.array.end(), std::ref(mt));
        return m;
    }
};
static constexpr size_t BookmarkPackedSize = sizeof(I_EventsStream::Bookmark::timestamp_) +
                                             sizeof(I_EventsStream::Bookmark::byte_offset_) +
                                             sizeof(I_EventsStream::Bookmark::cd_event_count_);

static const std::string platform_key           = "platform";
static const std::string hal_version_key        = "hal_version";
static const std::string hal_plugin_version_key = "hal_plugin_version";
static const std::string size_key               = "size";
static const std::string bookmark_period_key    = "bookmark_period_us";
static const std::string index_version_key      = "index_version";
static const std::string ts_shift_key           = "ts_shift_us";

static const std::string index_version          = "2.0";
static const uint32_t bookmark_period_us        = 2000;
static const std::string bookmark_period_us_str = std::to_string(bookmark_period_us);

const std::string &get_raw_file_index_extension_suffix() {
    static const std::string extension = ".tmp_index";
    return extension;
}

const std::filesystem::path get_raw_file_index_path(const std::filesystem::path &raw_file_path) {
    std::filesystem::path index_path = raw_file_path;
    index_path.replace_extension(index_path.extension().string() + get_raw_file_index_extension_suffix());
    return index_path;
}

bool serialize_bookmark(I_EventsStream::Bookmark &bookmark, std::ofstream &output_index_file) {
    if (!output_index_file.write(reinterpret_cast<char *>(&bookmark.timestamp_), sizeof(bookmark.timestamp_))) {
        return false;
    }
    if (!output_index_file.write(reinterpret_cast<char *>(&bookmark.byte_offset_), sizeof(bookmark.byte_offset_))) {
        return false;
    }
    if (!output_index_file.write(reinterpret_cast<char *>(&bookmark.cd_event_count_),
                                 sizeof(bookmark.cd_event_count_))) {
        return false;
    }
    return true;
}

bool deserialize_bookmark(I_EventsStream::Bookmark &bookmark, std::ifstream &input_index_file) {
    if (!input_index_file.read(reinterpret_cast<char *>(&bookmark.timestamp_), sizeof(bookmark.timestamp_))) {
        return false;
    }
    if (!input_index_file.read(reinterpret_cast<char *>(&bookmark.byte_offset_), sizeof(bookmark.byte_offset_))) {
        return false;
    }
    if (!input_index_file.read(reinterpret_cast<char *>(&bookmark.cd_event_count_), sizeof(bookmark.cd_event_count_))) {
        return false;
    }
    return true;
}

bool add_bookmarks(size_t last_bookmark_index, size_t bookmark_index, I_EventsStream::Bookmark &bookmark,
                   I_EventsStream::Index &index, std::ofstream &output_index_file) {
    for (; last_bookmark_index < bookmark_index; ++last_bookmark_index) {
        index.bookmarks_.push_back(bookmark);
        if (output_index_file) {
            if (!serialize_bookmark(bookmark, output_index_file)) {
                return false;
            }
        }
        // reset event count
        bookmark.cd_event_count_ = 0;
    }
    return true;
}

bool add_magic_number(std::ofstream &output_index_file) {
    BookmarkOrMagicNumber m = BookmarkOrMagicNumber::magic_number();
    if (output_index_file) {
        if (!serialize_bookmark(m.bookmark, output_index_file)) {
            return false;
        }
    }
    return true;
}

bool is_bookmark_magic_number(const I_EventsStream::Bookmark &bookmark) {
    BookmarkOrMagicNumber b(bookmark);
    BookmarkOrMagicNumber m = BookmarkOrMagicNumber::magic_number();
    return std::equal(b.array.begin(), b.array.begin() + BookmarkPackedSize, m.array.begin());
}

bool check_magic_number_presence(std::ifstream &input_index_file) {
    const auto start_pos = input_index_file.tellg();
    // Seek to last bookmark pos
    input_index_file.clear();
    if (!input_index_file.seekg(-static_cast<std::streamoff>(BookmarkPackedSize), std::ios::end)) {
        return false;
    }

    I_EventsStream::Bookmark bookmark;
    if (!deserialize_bookmark(bookmark, input_index_file)) {
        return false;
    }

    // Reset pos
    input_index_file.clear();
    input_index_file.seekg(start_pos);

    // Check that the last bookmark is indeed the magic number
    return is_bookmark_magic_number(bookmark);
}

bool build_and_try_writing_bookmarks(Device &device, I_EventsStream::Index &index,
                                     const std::filesystem::path &raw_file_path, GenericHeader &index_file_header,
                                     const std::filesystem::path &output_index_file_path,
                                     const std::atomic<bool> &abort) {
    // Opens the output index file
    std::ofstream output_index_file(output_index_file_path, std::ios::binary);
    if (!output_index_file) {
        MV_HAL_LOG_WARNING() << "Failed to write index file" << output_index_file_path << "for input RAW file"
                             << raw_file_path;
        MV_HAL_LOG_WARNING() << "Make sure the folder which contains the RAW file is writeable to avoid building "
                                "the index from scratch again next time";
    }

    I_EventsStream::Bookmark bookmark;

    // Grabs the facilities
    auto file_events_stream = device.get_facility<I_EventsStream>();
    auto decoder            = device.get_facility<I_EventsStreamDecoder>();
    auto hw_identification  = device.get_facility<I_HW_Identification>();

    // Gets a raw events size in bytes to be able to decode event per event
    const long raw_event_size_bytes = decoder->get_raw_event_size_bytes();

    // Retrieve byte offset information of the RAW file
    std::ifstream raw_file(raw_file_path, std::ios::binary);
    if (!raw_file) {
        MV_HAL_LOG_ERROR() << "Could not build index for the file. Failed to open RAW file at" << raw_file_path;
        return false;
    }

    GenericHeader raw_file_header(raw_file);
    size_t current_byte_offset = raw_file.tellg();
    raw_file.close();

    // Gets the decoder default timestamp so that we know when a valid timestamp has been decoded
    timestamp prev_ts = decoder->get_last_timestamp(), last_ts = -1;
    bool ts_shift_computed     = false;
    size_t last_bookmark_index = 0;
    size_t last_byte_offset    = current_byte_offset;
    size_t last_event_count    = 0;

    // Sets a cd callback to compute the event counts
    auto cd_decoder = device.get_facility<I_EventDecoder<EventCD>>();
    cd_decoder->add_event_buffer_callback(
        [&bookmark](auto begin, auto end) { bookmark.cd_event_count_ += std::distance(begin, end); });

    // start the streaming
    file_events_stream->start();

    // reads the file and writes the bookmarks
    auto then = std::chrono::steady_clock::now();
    while (file_events_stream->wait_next_buffer() > 0 && !abort) {
        auto now = std::chrono::steady_clock::now();
        if (then + std::chrono::seconds(1) <= now) {
            then = now;
            MV_HAL_LOG_TRACE() << "Still building index for" << raw_file_path << "...";
        }

        auto buffer = file_events_stream->get_latest_raw_data();
        // Decode the buffer events per events
        for (size_t idx = 0; idx < buffer.size();
             idx += raw_event_size_bytes, current_byte_offset += raw_event_size_bytes) {
            // Decode single event
            decoder->decode(buffer.data() + idx, buffer.data() + idx + raw_event_size_bytes);

            // Wait for timestamp shift to be computed before logging any bookmark
            if (!ts_shift_computed) {
                timestamp ts_shift_us;
                if (!decoder->get_timestamp_shift(ts_shift_us)) {
                    continue;
                }
                index.ts_shift_us_ = ts_shift_us;
                if (output_index_file) {
                    index_file_header.set_field(ts_shift_key, std::to_string(ts_shift_us));
                    output_index_file << index_file_header;
                }
                ts_shift_computed = true;
            }

            const auto new_ts = decoder->get_last_timestamp();
            // Same ts: nothing to do
            if (prev_ts == new_ts) {
                continue;
            }
            prev_ts = new_ts;

            const size_t bookmark_index = new_ts / bookmark_period_us + 1;
            if (bookmark_index > last_bookmark_index) {
                size_t event_count       = bookmark.cd_event_count_;
                bookmark.cd_event_count_ = last_event_count;
                bookmark.timestamp_      = last_ts;
                bookmark.byte_offset_    = last_byte_offset;
                if (!add_bookmarks(last_bookmark_index, bookmark_index, bookmark, index, output_index_file)) {
                    MV_HAL_LOG_ERROR() << "Could not write index to the file" << raw_file_path;
                    return false;
                }
                last_byte_offset    = current_byte_offset;
                last_ts             = new_ts;
                last_event_count    = event_count;
                last_bookmark_index = bookmark_index;
            }
        }
    }

    bookmark.cd_event_count_ = last_event_count;
    bookmark.timestamp_      = last_ts;
    bookmark.byte_offset_    = last_byte_offset;
    if (!add_bookmarks(last_bookmark_index, last_bookmark_index + 1, bookmark, index, output_index_file)) {
        MV_HAL_LOG_ERROR() << "Could not write index to the file" << raw_file_path;
        return false;
    }

    if (!add_magic_number(output_index_file)) {
        MV_HAL_LOG_ERROR() << "Could not write index to the file" << raw_file_path;
        return false;
    }

    return !abort;
}

I_EventsStream::Bookmarks load_bookmarks(std::ifstream &input_index_file, const std::atomic<bool> &abort) {
    // Loads the index from the file
    I_EventsStream::Bookmarks bookmarks;
    I_EventsStream::Bookmark bookmark;

    // The current implementation loads the whole index file in memory, which inevitably puts a limit on the maximum
    // size of a RAW file we support seeking. A better option would be to load the index on demand.
    while (input_index_file.good() && !abort) {
        if (deserialize_bookmark(bookmark, input_index_file)) {
            bookmarks.push_back(bookmark);
        }
    }
    if (!abort) {
        // if we reach here, a magic number should be present since we check for its presence before trying to load the
        // bookmarks but to be on the safe side ...
        if (bookmarks.empty() || !is_bookmark_magic_number(bookmarks.back())) {
            MV_HAL_LOG_ERROR() << "Unexpected error with index for RAW file, magic number expected but not found.";
            bookmarks = I_EventsStream::Bookmarks();
        } else {
            // this is not a real bookmark, let's remove it
            bookmarks.pop_back();
        }
    }

    return bookmarks;
}

I_EventsStream::Index build_index(Device &device, const std::filesystem::path &raw_file_path,
                                  const std::atomic<bool> &abort) {
    I_EventsStream::Index index;

    // ------------------------------
    // Retrieve RAW file info
    std::ifstream raw_file(raw_file_path, std::ios::binary);
    if (!raw_file) {
        // RAW file can't be opened. Should not happen.
        return index;
    }

    GenericHeader raw_file_header(raw_file);
    raw_file.clear();
    raw_file.seekg(0, std::ios::end);
    const auto data_end_pos = std::to_string(raw_file.tellg());
    raw_file.close();

#if defined _WIN32
    const std::string platform = "Windows";
#elif defined __APPLE__
    const std::string platform = "Darwin";
#else
    const std::string platform = "Linux";
#endif

    // ------------------------------
    // Checks the validity of the index file for the input RAW file

    // Opens the index file
    const std::filesystem::path raw_file_index_path(get_raw_file_index_path(raw_file_path));
    bool do_build_index = !std::filesystem::exists(raw_file_index_path);

    if (!do_build_index) {
        std::ifstream index_file(raw_file_index_path, std::ios::binary);

        // Index file exists
        // Now quick check the content to assert if the indexed file actually indexes the input RAW file

        GenericHeader index_file_header(index_file);
        const auto platform_in_file              = index_file_header.get_field(platform_key);
        const auto hal_version_in_file           = index_file_header.get_field(hal_version_key);
        const auto hal_plugin_version_in_file    = index_file_header.get_field(hal_plugin_version_key);
        const auto size_of_the_indexed_file      = index_file_header.get_field(size_key);
        const auto index_bookmark_period_in_file = index_file_header.get_field(bookmark_period_key);
        const auto index_version_in_file         = index_file_header.get_field(index_version_key);

        // Compare platform
        do_build_index = do_build_index || (platform_in_file != platform);

        // Compare HAL version
        do_build_index =
            do_build_index ||
            (hal_version_in_file != device.get_facility<I_HALSoftwareInfo>()->get_software_info().get_version());

        // Compare HAL plugin version
        do_build_index =
            do_build_index || (hal_plugin_version_in_file !=
                               device.get_facility<I_PluginSoftwareInfo>()->get_software_info().get_version());

        // Compare expected size vs actual size
        do_build_index = do_build_index || (size_of_the_indexed_file != data_end_pos);

        // Compare bookmark period in the index vs the one requested (for the moment constant)
        do_build_index = do_build_index || (index_bookmark_period_in_file != bookmark_period_us_str);

        // Compare bookmark version in the index vs the current one
        do_build_index = do_build_index || (index_version_in_file != index_version);

        // Checks that the timestamp shift is present ...
        do_build_index = do_build_index || index_file_header.get_field(ts_shift_key).empty();
        {
            // ... and is indeed a valid integer
            long long ts_shift;
            std::istringstream iss(index_file_header.get_field(ts_shift_key));
            if (!(iss >> ts_shift)) {
                do_build_index = true;
            }
        }

        // Compares header of the indexed RAW file and the input one
        auto raw_file_header_map   = raw_file_header.get_header_map();
        auto index_file_header_map = index_file_header.get_header_map();
        for (auto it = raw_file_header_map.begin(), it_end = raw_file_header_map.end(); it != it_end && !do_build_index;
             ++it) {
            auto found = index_file_header_map.find(it->first);
            if (found == index_file_header_map.end()) {
                do_build_index = true;
                break;
            }
            do_build_index = do_build_index || (found->second != it->second);
        }

        // Make sure that magic number is present
        do_build_index = do_build_index || !check_magic_number_presence(index_file);
    };

    // ------------------------------
    // If necessary, compute and write the index
    if (do_build_index) {
        // Builds and write the index
        MV_HAL_LOG_TRACE() << "Building index for input RAW file" << raw_file_path;

        // Write index file's header
        GenericHeader index_file_header(raw_file_header);
        index_file_header.set_field(platform_key, platform);
        index_file_header.set_field(hal_version_key,
                                    device.get_facility<I_HALSoftwareInfo>()->get_software_info().get_version());
        index_file_header.set_field(hal_plugin_version_key,
                                    device.get_facility<I_PluginSoftwareInfo>()->get_software_info().get_version());
        index_file_header.set_field(size_key, data_end_pos);
        index_file_header.set_field(bookmark_period_key, bookmark_period_us_str);
        index_file_header.set_field(index_version_key, index_version);

        if (!build_and_try_writing_bookmarks(device, index, raw_file_path, index_file_header, raw_file_index_path,
                                             abort)) {
            if (!abort) {
                MV_HAL_LOG_WARNING() << "Failed to build index for input RAW file" << raw_file_path;
            } else {
                if (std::filesystem::exists(raw_file_index_path)) {
                    // remove incomplete index file
                    std::filesystem::remove(raw_file_index_path);
                }
                MV_HAL_LOG_TRACE() << "Indexing for input RAW file" << raw_file_path
                                   << "has been aborted, removing incomplete index file";
            }
            index.status_ = I_EventsStream::IndexStatus::Bad;
            return index;
        }

        index.bookmark_period_ = bookmark_period_us;
        // index.ts_shift_us_ and index.bookmarks_ are filled by build_and_try_writing_bookmarks
        index.status_ = I_EventsStream::IndexStatus::Good;
        MV_HAL_LOG_TRACE() << "Index for input RAW file" << raw_file_path << "built";
    } else {
        std::ifstream index_file(raw_file_index_path, std::ios::binary);
        if (!index_file) {
            MV_HAL_LOG_ERROR() << "Failed to open index for RAW file at" << raw_file_index_path;
            index.status_ = I_EventsStream::IndexStatus::Bad;
            return index;
        }

        // ------------------------------
        // Build the index from the file
        GenericHeader index_file_header(index_file);
        index.bookmark_period_ = std::atol(index_file_header.get_field(bookmark_period_key).c_str());
        index.ts_shift_us_     = std::atoll(index_file_header.get_field(ts_shift_key).c_str());
        index.bookmarks_       = load_bookmarks(index_file, abort);
        if (index.bookmarks_.empty()) {
            MV_HAL_LOG_ERROR() << "Failed to open index for RAW file at" << raw_file_index_path;
            index.status_ = I_EventsStream::IndexStatus::Bad;
            return index;
        }
        index.status_ = I_EventsStream::IndexStatus::Good;
        MV_HAL_LOG_TRACE() << "Index for input RAW file" << raw_file_path << "loaded";
    }

    return index;
}

} // namespace

I_EventsStream::I_EventsStream(std::unique_ptr<DataTransfer::RawDataProducer> data_producer,
                               const std::shared_ptr<I_HW_Identification> &hw_identification,
                               const std::shared_ptr<I_EventsStreamDecoder> &decoder,
                               const std::shared_ptr<DeviceControl> &device_control) :
    data_transfer_(std::move(data_producer)),
    hw_identification_(hw_identification),
    decoder_(decoder),
    seeking_(false),
    device_control_(device_control),
    // this is not the most elegant way of figuring out if we are transferring data from an offline
    // recording where we should not release the buffers when the streaming is stopped, but this keeps
    // binary backward compatibility with previously released interfaces of DataTransfer
    stop_should_release_buffers_(std::dynamic_pointer_cast<FileRawDataProducer>(data_transfer_.get_data_producer()) ==
                                 nullptr),
    stop_(true) {
    if (!hw_identification_) {
        throw(HalException(HalErrorCode::FailedInitialization, "HW identification facility is null."));
    }
    data_transfer_.add_new_buffer_callback([this](const DataTransfer::BufferPtr &buffer) {
        std::unique_lock<std::mutex> lock(new_buffer_safety_);
        if (seeking_) {
            available_buffers_.push(buffer.clone());
        } else {
            if (!stop_) {
                auto buff_ptr = buffer.data();
                if (data_transfer_buffer_ptrs_.count(buff_ptr) == 0) {
                    data_transfer_buffer_ptrs_.insert(buff_ptr);
                }
                available_buffers_.push(buffer);
                new_buffer_cond_.notify_all();
            } else {
                if (!stop_should_release_buffers_) {
                    // streaming is stopped, this buffer comes from the data transfer and we should not release
                    // transferred buffers, so we need to copy this buffer in a temporary buffer pool to make sure the
                    // data transfer buffer pool is empty when streaming is resumed
                    auto tmp_buffer = buffer.clone();
                    available_buffers_.push(tmp_buffer);
                }
            }
        }
    });

    data_transfer_.add_status_changed_callback([this](DataTransfer::Status status) {
        if (status == DataTransfer::Status::Stopped) {
            bool should_notify = false;
            {
                std::unique_lock<std::mutex> lock(new_buffer_safety_);
                if (!seeking_) {
                    stop_         = true;
                    should_notify = true;
                }
            }
            if (should_notify) {
                new_buffer_cond_.notify_all();
            }
        } else {
            while (seeking_) {
                // wait for any seek operation to complete before resuming the actual data transfer
                std::this_thread::yield();
            }
        }
    });

    data_transfer_.add_transfer_error_callback([this](std::exception_ptr eptr) {
        std::unique_lock<std::mutex> lock(new_buffer_safety_);
        try {
            std::rethrow_exception(eptr);
        } catch (const HalConnectionException &) {
            // Only propagate connection exceptions
            data_transfer_connection_error_ = std::current_exception();
        } catch (...) {}
    });
}

I_EventsStream::~I_EventsStream() {
    if (index_build_thread_.joinable()) {
        abort_index_building_ = true;
        index_build_thread_.join();
    }
    try {
        stop();
    } catch (const std::exception &e) { MV_LOG_ERROR() << "I_EventsStream::stop() raised an exception : " << e.what(); }
    data_transfer_connection_error_ = nullptr;
}

void I_EventsStream::release_data_transfer_buffers() {
    std::queue<DataTransfer::BufferPtr> tmp_queue;
    std::swap(tmp_queue, available_buffers_);
    while (!tmp_queue.empty()) {
        auto buffer = tmp_queue.front();
        if (data_transfer_buffer_ptrs_.count(buffer.data())) {
            // we only copy buffers that are coming from the data transfer pool
            // those are the ones we need to release
            auto tmp_buffer = buffer.clone();
            available_buffers_.push(tmp_buffer);
        } else {
            available_buffers_.push(buffer);
        }
        tmp_queue.pop();
    }
}

void I_EventsStream::start() {
    std::lock_guard<std::mutex> lock(start_stop_safety_);
    {
        std::lock_guard<std::mutex> lock(new_buffer_safety_);
        stop_ = false;
    }
    data_transfer_.start();
    start_device();
}

void I_EventsStream::stop() {
    std::lock_guard<std::mutex> lock(start_stop_safety_);
    {
        std::lock_guard<std::mutex> lock(new_buffer_safety_);
        if (stop_should_release_buffers_) {
            available_buffers_ = {};
            returned_buffer_.reset();
        } else {
            // we need to make sure the data transfer pool is full before we resume transferring
            // we do this by releasing the buffers coming from the data transfer pool, copying their data
            // in buffers allocated in a temporary buffer pool
            release_data_transfer_buffers();
        }

        stop_ = true;
        new_buffer_cond_.notify_all();
        stop_log_raw_data();
    }
    stop_device();
    data_transfer_.stop();
}

void I_EventsStream::start_device() {
    if (device_control_ != nullptr) {
        device_control_->start();
    }
}

void I_EventsStream::stop_device() {
    if (device_control_ != nullptr) {
        device_control_->stop();
    }
}

short I_EventsStream::poll_buffer() {
    std::lock_guard<std::mutex> lock(new_buffer_safety_);

    if (stop_ && data_transfer_connection_error_) {
        std::rethrow_exception(data_transfer_connection_error_);
    }

    if (!available_buffers_.empty()) {
        return 1;
    }

    return stop_ ? -1 : 0;
}

short I_EventsStream::wait_next_buffer() {
    std::unique_lock<std::mutex> lock(new_buffer_safety_);
    new_buffer_cond_.wait(lock, [this]() { return !available_buffers_.empty() || stop_; });

    if (stop_ && data_transfer_connection_error_) {
        std::rethrow_exception(data_transfer_connection_error_);
    }

    return available_buffers_.empty() ? -1 : 1;
}

DataTransfer::BufferPtr I_EventsStream::get_latest_raw_data() {
    DataTransfer::BufferPtr res;
    {
        std::lock_guard<std::mutex> lock(new_buffer_safety_);

        if (stop_ && data_transfer_connection_error_) {
            std::rethrow_exception(data_transfer_connection_error_);
        }

        if (available_buffers_.empty()) {
            // If no new buffer available yet
            return {};
        }

        // Reset potential reference from last call
        returned_buffer_.reset();
        res = available_buffers_.front();
        available_buffers_.pop();
    }

    {
        std::lock_guard<std::mutex> log_lock(log_raw_safety_);
        if (log_raw_data_) {
            log_raw_data_->write(reinterpret_cast<const char *>(res.data()), res.size() * sizeof(RawData));
        }
    }
    return res;
}

I_EventsStream::SeekStatus I_EventsStream::seek(timestamp target_ts_us, timestamp &reached_ts_us) {
    std::lock_guard<std::mutex> lock(index_safety_);

    switch (index_.status_) {
    case I_EventsStream::IndexStatus::Bad:
        return SeekStatus::SeekCapabilityNotAvailable;
    case I_EventsStream::IndexStatus::Building:
    case I_EventsStream::IndexStatus::NotBuilt:
        return SeekStatus::IndexNotAvailableYet;
    default:
        break;
    }

    auto file_raw_data_producer = std::dynamic_pointer_cast<FileRawDataProducer>(data_transfer_.get_data_producer());
    if (!file_raw_data_producer || !decoder_) {
        // should never happen, we check that seeking is possible before indexing...
        return SeekStatus::SeekCapabilityNotAvailable;
    }

    if (!decoder_->is_time_shifting_enabled()) {
        target_ts_us -= index_.ts_shift_us_;
    }

    size_t bookmark_index = target_ts_us / index_.bookmark_period_;
    if (target_ts_us < 0 || bookmark_index >= index_.bookmarks_.size()) {
        return SeekStatus::InputTimestampNotReachable;
    }
    // do not seek before first valid timestamp
    while (index_.bookmarks_[bookmark_index].timestamp_ < 0) {
        bookmark_index++;
    }

    // after this point, we make sure next received buffers are released and stored in the
    // temporary buffer pool so that we can stop the data transfer
    // we also ignore data transfer status changes, if we successfully seek, then status change
    // will be invalid
    seeking_ = true;

    {
        // release all buffers from the data transfer pool to unblock any pending transfer to avoid a deadlock
        // when trying to suspend the data transfer
        std::lock_guard<std::mutex> lock(new_buffer_safety_);
        release_data_transfer_buffers();
    }

    data_transfer_.suspend();
    auto seek_succeed = file_raw_data_producer->seek(index_.bookmarks_[bookmark_index].byte_offset_);

    SeekStatus seek_status;
    if (seek_succeed) {
        reached_ts_us = index_.bookmarks_[bookmark_index].timestamp_ +
                        (decoder_->is_time_shifting_enabled() ? 0 : index_.ts_shift_us_);
        seek_status = SeekStatus::Success;

        {
            std::unique_lock<std::mutex> lock(new_buffer_safety_);
            // thrash the buffers, but not resetting the returned_buffer_, it may still be valid and in use
            available_buffers_ = {};
            // even if we were at the end of the file, after we seek, we won't be
            stop_ = false;
        }

        if (data_transfer_.stopped()) {
            // if the data transfer was stopped because the end of file was reached before seeking, restart it
            //
            // Note : even if we start the data transfer now (while seeking_ = true), there is no risk of missing a EOF
            // (which would be signaled by a Stopped status change ... but ignored because a seek is ongoing) since the
            // data transfer won't resume reading (and thus reach a potential EOF) before seeking_ = false (c.f wait
            // loop in status change callback)
            data_transfer_.start();
        }
    } else {
        seek_status = SeekStatus::Failed;

        if (data_transfer_.stopped()) {
            // if the data transfer was stopped while we were seeking, and the seek failed,
            // update our status
            {
                std::unique_lock<std::mutex> lock(new_buffer_safety_);
                stop_ = true;
            }
            new_buffer_cond_.notify_all();
        }
    }

    seeking_ = false;
    data_transfer_.resume();

    return seek_status;
}

I_EventsStream::IndexStatus I_EventsStream::get_seek_range(timestamp &data_start_ts, timestamp &data_end_ts) const {
    std::lock_guard<std::mutex> lock(index_safety_);
    switch (index_.status_) {
    case I_EventsStream::IndexStatus::Bad:
    case I_EventsStream::IndexStatus::Building:
    case I_EventsStream::IndexStatus::NotBuilt:
        return index_.status_;
    default:
        break;
    }

    for (const auto &bookmark : index_.bookmarks_) {
        if (bookmark.timestamp_ >= 0) {
            data_start_ts = bookmark.timestamp_;
            break;
        }
    }
    data_end_ts = index_.bookmarks_.back().timestamp_;

    if (!decoder_->is_time_shifting_enabled()) {
        data_start_ts += index_.ts_shift_us_;
        data_end_ts += index_.ts_shift_us_;
    }
    return IndexStatus::Good;
}

void I_EventsStream::index(std::unique_ptr<Device> device_for_indexing) {
    std::lock_guard<std::mutex> lock(index_safety_);
    if (decoder_ && !decoder_->is_decoded_event_stream_indexable()) {
        index_.status_ = I_EventsStream::IndexStatus::Bad;
        return;
    }

    if (index_.status_ == IndexStatus::Good) {
        return;
    }

    if (get_underlying_file().empty()) {
        MV_HAL_LOG_ERROR() << "Can not build index for the stream input (no valid RAW file name found).";
        index_.status_ = IndexStatus::Bad;
        return;
    }

    auto indexing_fes = device_for_indexing->get_facility<Metavision::I_EventsStream>();
    if (!indexing_fes ||
        !std::dynamic_pointer_cast<FileRawDataProducer>(indexing_fes->data_transfer_.get_data_producer()) ||
        !indexing_fes->decoder_) {
        MV_HAL_LOG_ERROR() << "Can not build index for the stream input: invalid indexing device.";
        return;
    }

    if (indexing_fes->get_underlying_file() != get_underlying_file()) {
        MV_HAL_LOG_ERROR() << "Can not build index for the stream input: indexing device is built from another RAW "
                              "file as source. The file to index is"
                           << get_underlying_file() << "whereas the input indexing device has been built from"
                           << indexing_fes->get_underlying_file();
    }

    index_.status_ = IndexStatus::Building;

    // Start the indexing thread
    index_build_thread_ = std::thread([device_for_indexing = std::move(device_for_indexing), this]() {
        try {
            auto index = index_impl(*device_for_indexing);

            std::lock_guard<std::mutex> lock(index_safety_);
            std::swap(index_, index);
        } catch (const std::exception &e) {
            MV_HAL_LOG_WARNING() << "Unhandled error while building index:";
            MV_HAL_LOG_WARNING() << e.what();
            MV_HAL_LOG_WARNING() << "Seek feature might not work properly.";
        }
    });

    // Wait for the thread to be running
    while (!index_build_thread_.joinable()) {}
}

I_EventsStream::Index I_EventsStream::index_impl(Device &device) {
    abort_index_building_ = false;
    auto index            = build_index(device, get_underlying_file(), abort_index_building_);
    decoder_->reset_timestamp_shift(index.ts_shift_us_);
    return index;
}

void I_EventsStream::stop_log_raw_data() {
    std::lock_guard<std::mutex> guard(log_raw_safety_);
    log_raw_data_.reset(nullptr);
}

bool I_EventsStream::log_raw_data(const std::string &f) {
    if (f == underlying_file_) {
        return false;
    }

    auto header = hw_identification_->get_header();
    header.add_date();

    std::lock_guard<std::mutex> guard(log_raw_safety_);
    log_raw_data_.reset(new std::ofstream(f, std::ios::binary));
    if (!log_raw_data_->is_open()) {
        log_raw_data_ = nullptr;
        return false;
    }

    (*log_raw_data_) << header;
    return true;
}

void I_EventsStream::set_underlying_file(const std::filesystem::path &file) {
    underlying_file_ = file;
}

const std::filesystem::path &I_EventsStream::get_underlying_file() const {
    return underlying_file_;
}

} // namespace Metavision
