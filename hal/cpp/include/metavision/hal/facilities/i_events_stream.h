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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_set>

#include "metavision/hal/facilities/i_camera_synchronization.h"
#include "metavision/hal/facilities/i_registrable_facility.h"
#include "metavision/hal/utils/data_transfer.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/hal/utils/device_control.h"

namespace Metavision {
class Device;
class I_HW_Identification;

class I_EventsStreamDecoder;

/// @brief Class for getting buffers from cameras or files.
class I_EventsStream : public I_RegistrableFacility<I_EventsStream> {
public:
    using RawData = DataTransfer::Data;

    /// @brief Constructor
    /// @param data_transfer Data transfer class owned by the events stream and used to transfer data
    /// @param hw_identification Hardware identification associated to this events stream
    /// @param decoder Decoder associated to this events stream
    /// @param device_control Device control class for starting and stopping
    I_EventsStream(std::unique_ptr<DataTransfer> data_transfer,
                   const std::shared_ptr<I_HW_Identification> &hw_identification,
                   const std::shared_ptr<I_EventsStreamDecoder> &decoder = nullptr,
                   const std::shared_ptr<DeviceControl> &device_control  = std::shared_ptr<DeviceControl>());

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
    ///
    /// @param n_rawbytes Address of a variable in which to put the number of bytes contained in the buffer
    /// @return Pointer to an array of Event structures
    /// @note This function must be called to write the buffer of events in the log file defined in @ref
    /// log_raw_data
    RawData *get_latest_raw_data(long &n_rawbytes);

    /// @brief Enables the logging of the stream of events in the input file @a f
    ///
    /// This methods first writes the header retrieved through @ref I_HW_Identification.
    /// Buffers of data are then written each time @ref get_latest_raw_data is called (i.e. in the same thread it is
    /// called).
    ///
    /// @param f The file to log into
    /// @return true if the file could be opened for writing, false otherwise or if the file name @a f is the same
    /// as the one read from
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

    /// @brief Gets name of the file read to avoid writing in the same file when calling log_raw_data
    const std::string &get_underlying_filename() const;

    /// @brief Structure representing a bookmark that composes the index
    struct Bookmark {
        uint64_t byte_offset_{0};    // The byte offset in the file
        int64_t timestamp_{0};       // timestamp associated
        uint32_t cd_event_count_{0}; // cd events count in the RAW file since the last bookmark
    };

    /// @brief List of bookmarks in the index
    using Bookmarks = std::vector<Bookmark>;

    /// @brief Enumerator stating the different output of a call to @ref seek
    enum class SeekStatus {
        Success,
        Failed,
        InputTimestampNotReachable,
        IndexNotAvailableYet,
        SeekCapabilityNotAvailable
    };

    /// @brief Enumerator stating the status of index
    enum class IndexStatus {
        Good,     ///< Index is loaded and ready to be used
        Bad,      ///< Index failed to be loaded or built. Seek operation can not be done.
        Building, ///< Index is being built. Seek operations are not available yet.
        NotBuilt  ///< Index has not been built: @ref index has not been called yet
    };

    /// @brief The index structure
    struct Index {
        Bookmarks bookmarks_;         ///< the bookmarks that compose the index
        uint32_t bookmark_period_{0}; ///< the minimum period between two successive bookmarks
        timestamp ts_shift_us_{0};    ///< The timeshift to apply to the data

        IndexStatus status_{IndexStatus::NotBuilt}; ///< The index's state
    };

    /// @brief Tries to reach the input @a target_ts_us in the file
    /// If the seek succeeds, the next data read from the file and accessible through @ref get_latest_raw_data will
    /// hold data from the closest bookmark (lower timestamp). The first timestamp decoded then will be @a
    /// reached_ts_us
    /// @param target_ts_us The target timestamp to reach in the RAW file
    /// @param reached_ts_us The reached timestamp if the seek succeeds
    /// @return a status (@ref SeekStatus) holding the result of the seek
    SeekStatus seek(timestamp target_ts_us, timestamp &reached_ts_us);

    /// @brief Gets the range of timestamp reachable through the @ref seek method.
    /// @param event_start_ts_us The timestamp of the first valid data in the file
    /// @param event_end_ts_us The timestamp of the last valid data in the file
    /// @return true If the seek feature is implemented
    IndexStatus get_seek_range(timestamp &event_start_ts_us, timestamp &event_end_ts_us) const;

    /// @brief Builds an index to enable navigation in a RAW file through the @ref seek and @ref get_seek_range
    /// routines.
    ///
    /// To navigate efficiently, an index is created allowing to associate a timestamp with a position in the file.
    /// When this index is loaded in memory, one can get the range of timestamp that can be reached when navigating.
    ///
    /// If the index already exists for the source RAW file, it is just loaded in memory. Otherwise it is built in a
    /// dedicated thread. Until the index is loaded in memory, seek operations are not available. The return value
    /// of @ref get_seek_range can be used to check the current @ref IndexStatus.
    ///
    /// @param device_for_indexing The device to use to index the RAW file.
    /// @warning The input device must have been built with the same RAW file used to initialize this class
    void index(std::unique_ptr<Device> device_for_indexing);

private:
    /// @brief Builds and loads the index in memory
    virtual Index index_impl(Device &device);

    void release_data_transfer_buffers();
    void start_device();
    void stop_device();

    std::shared_ptr<I_HW_Identification> hw_identification_;
    std::shared_ptr<I_EventsStreamDecoder> decoder_;

    // Name of the file read if one
    std::string underlying_filename_;

    std::unique_ptr<std::ofstream> log_raw_data_;
    std::mutex log_raw_safety_;

    std::unique_ptr<DataTransfer> data_transfer_;
    std::shared_ptr<DeviceControl> device_control_;
    std::mutex new_buffer_safety_;
    std::condition_variable new_buffer_cond_;
    std::queue<DataTransfer::BufferPtr> available_buffers_;
    DataTransfer::BufferPtr returned_buffer_;

    // For some data transfer, we should not release already transferred buffers when streaming is stopped
    // To achieve this, we copy buffers internally in a temporary buffer pool to always leave the data transfer
    // buffer pool full when resuming streaming
    const bool stop_should_release_buffers_;
    DataTransfer::BufferPool tmp_buffer_pool_;
    std::unordered_set<DataTransfer::BufferPtr::element_type *>
        data_transfer_buffer_ptrs_; // for quick check if copying is necessary

    std::mutex start_stop_safety_;
    bool stop_;

    std::atomic<bool> seeking_;
    std::thread index_build_thread_;
    Index index_;
    std::atomic<bool> abort_index_building_;
    mutable std::mutex index_safety_;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_EVENTS_STREAM_H
