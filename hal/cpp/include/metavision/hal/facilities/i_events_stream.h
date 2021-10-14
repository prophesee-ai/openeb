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

#ifndef METAVISION_HAL_I_EVENTS_STREAM_H
#define METAVISION_HAL_I_EVENTS_STREAM_H

#include <string>
#include <fstream>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <queue>

#include "metavision/hal/facilities/i_registrable_facility.h"
#include "metavision/hal/utils/data_transfer.h"

namespace Metavision {

class I_HW_Identification;

/// @brief Class for getting buffers from cameras or files.
class I_EventsStream : public I_RegistrableFacility<I_EventsStream> {
public:
    using RawData = DataTransfer::Data;

    /// @brief Constructor
    /// @param data_transfer Data transfer class owned by the events stream and used to transfer data
    /// @param hw_identification Hardware identification associated to this events stream
    I_EventsStream(std::unique_ptr<DataTransfer> data_transfer,
                   const std::shared_ptr<I_HW_Identification> &hw_identification);

    /// @brief Destructor
    ~I_EventsStream();

    /// @brief Starts streaming events
    void start();

    /// @brief Stops streaming events
    void stop();

    /// @brief Returns a value that informs if some events are available in the buffer from the camera or the file
    /// @return Value that informs if some events are available in the buffer
    ///         -  1 if there are events available
    ///         -  0 if no events are available
    ///         - -1 if an error occurred or no more events will ever be available (like when reaching end of file)
    short poll_buffer();

    /// @brief Returns a value that informs if some events are available in the buffer from the camera
    /// and blocks waiting until more events are available
    /// @return Value that informs if some events are available in the buffer
    ///         -  1 if there are events available
    ///         - -1 if an error occurred or no more events will ever be available (like when reaching end of file)
    short wait_next_buffer();

    /// @brief Gets latest raw data from the event buffer
    ///
    /// Gets raw data from the event buffer received since the last time this function was called.
    /// @param n_rawbytes Address of a variable in which to put the number of bytes contained in the buffer
    /// @return Pointer to an array of Event structures
    /// @note This function must be called to write the buffer of events in the log file defined in @ref log_raw_data
    RawData *get_latest_raw_data(long &n_rawbytes);

    /// @brief Enables the logging of the stream of events in the input file @a f
    ///
    /// This methods first writes the header retrieved through @ref I_HW_Identification.
    /// Buffers of data are then written each time @ref get_latest_raw_data is called (i.e. in the same thread it is
    /// called).
    /// @param f The file to log into
    /// @return true if the file could be opened for writing, false otherwise or if the file name @a f is the same as
    /// the one read from
    /// @warning The writing of each buffer of event will have to be triggered by calls to @ref get_latest_raw_data
    bool log_raw_data(const std::string &f);

    /// @brief Stops logging RAW data
    ///
    /// Does nothing if no recording has been started
    void stop_log_raw_data();

    /// @brief Sets name of the file read to avoid writing in the same file when calling log_raw_data
    /// @param filename Name of the file from which the events are read
    /// @note This function is directly called when opening a RAW file
    void set_underlying_filename(const std::string &filename);

private:
    std::shared_ptr<I_HW_Identification> hw_identification_;

    // Name of the file read if one
    std::string underlying_filename_;

    std::unique_ptr<std::ofstream> log_raw_data_;
    std::mutex log_raw_safety_;

    std::unique_ptr<DataTransfer> data_transfer_;
    std::mutex new_buffer_safety_;
    std::condition_variable new_buffer_cond_;
    std::queue<DataTransfer::BufferPtr> available_buffers_;
    DataTransfer::BufferPtr returned_buffer_;

    // For some data transfer, we should not release already transferred buffers when streaming is stopped
    // To achieve this, we copy buffers internally in a temporary buffer pool to always leave the data transfer
    // buffer pool full when resuming streaming
    const bool stop_should_release_buffers_;
    DataTransfer::BufferPool tmp_buffer_pool_;
    std::vector<DataTransfer::BufferPtr::element_type *>
        data_transfer_buffer_ptrs_; // for quick check if copying is necessary

    std::mutex start_stop_safety_;
    bool stop_;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_EVENTS_STREAM_H
