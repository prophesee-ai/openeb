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

#ifndef METAVISION_SDK_DRIVER_EVENT_FILE_READER_H
#define METAVISION_SDK_DRIVER_EVENT_FILE_READER_H

#include <functional>
#include <memory>
#include <unordered_map>
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_erc_counter.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/sdk/base/events/raw_event_frame_diff.h"
#include "metavision/sdk/base/events/raw_event_frame_histo.h"

namespace Metavision {

/// @brief Abstract class used for reading event based file
///
class EventFileReader {
public:
    /// @brief Builds an EventFileReader
    /// @param path Path to the event based file to read
    EventFileReader(const std::string &path);

    /// @brief Destructs an EventFileReader
    virtual ~EventFileReader();

    /// @brief Gets the path of the event based file
    /// @return Path of the file
    const std::string &get_path() const;

    /// @brief Alias for the "reading callback" used when a buffer of events has been read
    ///
    /// The callback will be called with two pointers which represent the range of events that were just read
    ///
    /// @tparam EventType Type of event read
    template<typename EventType>
    using EventsBufferReadCallback = std::function<void(const EventType *, const EventType *)>;

    /// @brief Alias for the "reading callback" used when an event frame has been read
    ///
    /// The callback will be called with a reference to the event frame that was read
    ///
    /// @tparam EventFrameType Type of event frame read
    template<typename EventFrameType>
    using EventFrameReadCallback = std::function<void(const EventFrameType &)>;

    /// @brief Adds a "reading callback" called when a buffer of events has been read
    /// @param cb Callback to be called
    /// @return Id of the added callback, see @ref remove_callback
    size_t add_read_callback(const EventsBufferReadCallback<EventCD> &cb);

    /// @brief Adds a "reading callback" called when a buffer of events has been read
    /// @overload
    size_t add_read_callback(const EventsBufferReadCallback<EventExtTrigger> &cb);

    /// @brief Adds a "reading callback" called when a buffer of events has been read
    /// @overload
    size_t add_read_callback(const EventsBufferReadCallback<EventERCCounter> &cb);

    /// @brief Adds a "reading callback" called when a histogram has been read
    /// @overload
    size_t add_read_callback(const EventFrameReadCallback<RawEventFrameHisto> &cb);

    /// @brief Adds a "reading callback" called when a diff histogram has been read
    /// @overload
    size_t add_read_callback(const EventFrameReadCallback<RawEventFrameDiff> &cb);

    /// @brief Checks if "reading callbacks" have been added
    /// @return true if callbacks have been added, false otherwise
    bool has_read_callbacks() const;

    /// @brief Reads block of events from the input stream
    /// @return true if some events could be read, false otherwise
    bool read();

    /// @brief Queries whether the event file supports seeking
    /// @return true if seek is supported, false otherwise
    virtual bool seekable() const = 0;

    /// @brief Alias for the "seeking callback" used when a seek operation successfully completed
    ///
    /// The callback will be called with a timestamp representing the actual timestamp at which
    /// the reader has seeked to
    using SeekCompletionCallback = std::function<void(timestamp)>;

    /// @brief Adds a "seeking callback" called when a seek operation successfully completed
    /// @param cb Callback to be called
    /// @return Id of the added callback, see @ref remove_callback
    size_t add_seek_callback(const SeekCompletionCallback &cb);

    /// @brief Checks if "seeking callbacks" have been added
    /// @return true if callbacks have been added, false otherwise
    bool has_seek_callbacks() const;

    /// @brief Gets the allowed seek range timestamps
    /// @param min_t Reference to the minimum timestamp to seek to
    /// @param max_t Reference to the maximum timestamp to seek to
    /// @return true if the seek range could be determined, false otherwise
    bool get_seek_range(timestamp &min_t, timestamp &max_t) const;

    /// @brief Gets the duration of the file
    /// @return Duration or -1 if not available
    timestamp get_duration() const;

    /// @brief Seeks to a specified timestamp
    /// @param t Timestamp to seek to
    /// @return true if the seek was successful, false otherwise
    bool seek(timestamp t);

    /// @brief Removes one callback
    /// @param id Id of the callback to remove (see @ref add_read_callback, @ref add_seek_callback)
    void remove_callback(size_t id);

    /// @brief Gets the metadata map of the file
    /// @return Metadata, if available
    std::unordered_map<std::string, std::string> get_metadata_map() const;

    /// @brief For internal use
    class Private;

    /// @brief For internal use
    Private &get_pimpl();

protected:
    void notify_events_buffer(const EventCD *, const EventCD *);
    void notify_events_buffer(const EventExtTrigger *, const EventExtTrigger *);
    void notify_events_buffer(const EventERCCounter *, const EventERCCounter *);
    void notify_event_frame(const RawEventFrameHisto &);
    void notify_event_frame(const RawEventFrameDiff &);
    void notify_seek(timestamp t);

private:
    virtual bool read_impl()                                                   = 0;
    virtual bool get_seek_range_impl(timestamp &min_t, timestamp &max_t) const = 0;
    virtual bool seek_impl(timestamp t)                                        = 0;
    virtual timestamp get_duration_impl() const                                = 0;
    virtual std::unordered_map<std::string, std::string> get_metadata_map_impl() const;

    std::unique_ptr<Private> pimpl_;
};

} // namespace Metavision

#endif /* METAVISION_SDK_DRIVER_EVENT_FILE_READER_H */
