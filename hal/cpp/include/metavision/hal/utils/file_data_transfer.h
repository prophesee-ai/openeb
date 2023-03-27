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

#ifndef METAVISION_HAL_FILE_DATA_TRANSFER_H
#define METAVISION_HAL_FILE_DATA_TRANSFER_H

#include <fstream>
#include <memory>
#include <atomic>
#include <mutex>

#include "metavision/hal/utils/data_transfer.h"
#include "metavision/hal/utils/raw_file_config.h"

namespace Metavision {

/// @brief Standard stream reader
class FileDataTransfer : public DataTransfer {
public:
    /// @brief Reads the input standard @a stream batch by batch according to the input configuration
    /// @param stream The stream to read from
    /// @param raw_event_size_bytes The size of a RAW event in bytes
    /// @param config The configuration to use to read the stream
    FileDataTransfer(std::unique_ptr<std::istream> stream, uint32_t raw_event_size_bytes, const RawFileConfig &config);

    /// @brief Stops ongoing transfers
    ~FileDataTransfer();

    /// @brief Seeks the target position in the file
    /// @param target_position The target position of the cursor to seek in the file
    bool seek(const std::streampos &target_position);

    /// @brief Gets the range of available positions when using @ref seek
    /// @param data_start_pos The offset position of the first data in the file
    /// @param data_end_pos The offset position of the next position after the last data in the file
    void get_seek_range(std::streampos &data_start_pos, std::streampos &data_end_pos) const;

private:
    void start_impl(BufferPtr buffer) override final;
    void run_impl() override final;

    virtual bool seek_impl(const std::streampos &target_position);

    /// Buffer
    BufferPtr data_read_;

    /// Bytes batch size to read from stream at each read iteration
    uint32_t read_bytes_size_{0};

    std::mutex seek_mutex_;
    std::condition_variable seek_cond_;
    std::atomic<bool> seeking_;
    BufferPtr seek_buffer_; // extra slack buffer we can use to unblock any pending acquire from the transfer buffer
                            // pool before doing a seek

    std::mutex stream_mutex_;
    std::condition_variable stream_cond_;
    std::streampos data_start_pos_, data_end_pos_;
    std::unique_ptr<std::istream> stream_to_read_;
};
} // namespace Metavision

#endif // METAVISION_HAL_FILE_DATA_TRANSFER_H
