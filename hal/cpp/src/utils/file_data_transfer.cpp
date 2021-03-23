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

#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/file_data_transfer.h"

namespace Metavision {

FileDataTransfer::FileDataTransfer(std::unique_ptr<std::istream> stream, uint32_t raw_event_size_bytes,
                                   const RawFileConfig &config) :
    DataTransfer(raw_event_size_bytes, BufferPool::make_bounded(std::max(2u, config.n_read_buffers_))),
    stream_to_read_(std::move(stream)) {
    // Constraint on the configuration
    if (raw_event_size_bytes == 0) {
        throw HalException(HalErrorCode::InvalidArgument, "RAW event byte size must be greater than 0.");
    }

    if (config.n_events_to_read_ == 0) {
        throw HalException(HalErrorCode::InvalidArgument, "Invalid RAW file reader configuration. The number of RAW "
                                                          "events to read per read iteration must be greater than 0.");
    }

    read_bytes_size_ = config.n_events_to_read_ * get_raw_event_size_bytes();
}

FileDataTransfer::~FileDataTransfer() {
    stop();
}

void FileDataTransfer::start_impl(BufferPtr buffer) {
    data_read_ = buffer;
}

void FileDataTransfer::run_impl() {
    while (!should_stop()) {
        data_read_->resize(read_bytes_size_); // Does not reallocate if enough memory already allocated.
        stream_to_read_->read(reinterpret_cast<char *>(data_read_->data()), read_bytes_size_);

        // get size of what have been read (in bytes)
        auto read = stream_to_read_->gcount();
        if (read > 0) {
            // If something has been read: transfer the data
            data_read_->resize(read);
            auto next_data_read = transfer_data(data_read_);
            data_read_          = next_data_read;
        } else {
            // Otherwise investigate if something is wrong
            if (!stream_to_read_->good()) {
                break;
            }
        }
    }
}

} // namespace Metavision
