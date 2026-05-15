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

#include "metavision/hal/utils/data_transfer.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/file_raw_data_producer.h"

namespace Metavision {

FileRawDataProducer::FileRawDataProducer(std::unique_ptr<std::istream> stream, uint32_t raw_event_size_bytes,
                                         const RawFileConfig &config, DataTransfer::DefaultBufferPool pool) :
    buffer_pool_(pool), seeking_(false), stream_to_read_(std::move(stream)) {
    // Constraint on the configuration
    if (raw_event_size_bytes == 0) {
        throw HalException(HalErrorCode::InvalidArgument, "RAW event byte size must be greater than 0.");
    }

    if (config.n_events_to_read_ == 0) {
        throw HalException(HalErrorCode::InvalidArgument, "Invalid RAW file reader configuration. The number of RAW "
                                                          "events to read per read iteration must be greater than 0.");
    }

    read_bytes_size_ = config.n_events_to_read_ * raw_event_size_bytes;
    data_start_pos_  = stream_to_read_->tellg();

    stream_to_read_->seekg(0, std::ios::end);
    data_end_pos_ = stream_to_read_->tellg();

    stream_to_read_->clear();
    stream_to_read_->seekg(data_start_pos_);

    seek_buffer_ = buffer_pool_.acquire();
}

bool FileRawDataProducer::seek(const std::streampos &target_position) {
    // If input position is outside the seek range in the stream
    if (target_position < data_start_pos_ || target_position > data_end_pos_) {
        return false;
    }

    return seek_impl(target_position);
}

void FileRawDataProducer::get_seek_range(std::streampos &data_start_pos, std::streampos &data_end_pos) const {
    data_start_pos = data_start_pos_;
    data_end_pos   = data_end_pos_;
}

void FileRawDataProducer::start_impl() {}

void FileRawDataProducer::run_impl(const DataTransfer &data_transfer) {
    while (!data_transfer.should_stop()) {
        {
            std::unique_lock<std::mutex> lock(seek_mutex_);
            seek_cond_.wait(lock, [this] { return !seeking_; });

            if (!seek_buffer_) {
                // make sure we always have one buffer of slack to unblock any pending transfer
                // while trying to perform an operation that could invalidate the stream
                seek_buffer_ = buffer_pool_.acquire();
            }
        }

        {
            std::lock_guard<std::mutex> lock(stream_mutex_);

            auto data_read = buffer_pool_.acquire();
            data_read->resize(read_bytes_size_); // Does not reallocate if enough memory already allocated.

            stream_to_read_->read(reinterpret_cast<char *>(data_read->data()), read_bytes_size_);

            // get size of what have been read (in bytes)
            auto count = stream_to_read_->gcount();

            // gets status of the stream
            auto good = stream_to_read_->good();

            if (count > 0) {
                // If something has been read, transfer the data
                data_read->resize(count);
                data_transfer.transfer_data(data_read);
            } else if (!good) {
                // Otherwise stop
                break;
            }
        }
    }
}

bool FileRawDataProducer::seek_impl(const std::streampos &target_position) {
    // we need to do free at least one buffer in the pool to unblock an ongoing transfer (if any)
    // otherwise, we could deadlock waiting for the mutex to be freed after the transfer completes
    {
        std::unique_lock<std::mutex> lock(seek_mutex_);
        seeking_ = true;
        seek_buffer_.reset();
    }

    bool ret;
    {
        std::lock_guard<std::mutex> lock(stream_mutex_);

        // Keep current pos in memory if seek to a bad position
        std::streampos current_pos = stream_to_read_->tellg();
        stream_to_read_->clear(); // If status is failed, seek may not work thus clear is necessary
        stream_to_read_->seekg(target_position);

        stream_to_read_->get();
        if (stream_to_read_->good()) {
            // valid seek
            stream_to_read_->unget();
            ret = true;
        } else {
            // Invalid seek -> return to initial position
            stream_to_read_->clear(); // If status is failed, seek may not work thus clear is necessary
            stream_to_read_->seekg(current_pos);
            ret = false;
        }
    }

    {
        std::lock_guard<std::mutex> lock(seek_mutex_);
        seeking_ = false;
    }
    seek_cond_.notify_all();

    return ret;
}

} // namespace Metavision
