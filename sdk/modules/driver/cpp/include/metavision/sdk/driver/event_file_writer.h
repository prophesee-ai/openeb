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

#ifndef METAVISION_SDK_DRIVER_EVENT_FILE_WRITER_H
#define METAVISION_SDK_DRIVER_EVENT_FILE_WRITER_H

#include <string>
#include <memory>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"

namespace Metavision {

class Camera;

/// @brief Abstract class used for reading event based file
///
class EventFileWriter {
public:
    /// @brief Builds an EventFileWriter
    /// @param path Path to the event based file to read
    EventFileWriter(const std::string &path = std::string());

    /// @brief Destructs an EventFileWriter
    virtual ~EventFileWriter();

    /// @brief Open the writer with specified path
    /// @param path Path to the file to write
    void open(const std::string &path);

    /// @brief Close the writer
    void close();

    /// @brief Check that the writer is currently opened and ready to write data
    /// @return true if the writer is opened and ready, false otherwise
    bool is_open() const;

    /// @brief Try to flush any data not yet written to disk
    ///
    /// The underlying implementation will do its best to flush any data not yet written
    /// but it's not guaranteed that all data will be written after this call (a library or the OS
    /// might prevent completely flushing that data at this point).
    /// If no more data will be added, closing the writer is the only way to make sure the data
    /// is fully written to disk.
    void flush();

    /// @brief Gets the path of the event based file
    /// @return Path of the file
    const std::string &get_path() const;

    /// @brief Adds metadata to the file
    /// @param key Metatadata key to be added
    /// @param value Metadata value to be added
    /// @warning Calling this function after some data has been added may throw an exception
    ///          in some implementation
    void add_metadata(const std::string &key, const std::string &value);

    /// @brief Adds metadata to the file from a Camera instance
    /// @param camera Camera instance to build metadata from
    /// @warning Calling this function after some data has been added may throw an exception
    ///          in some implementation
    void add_metadata_map_from_camera(const Camera &camera);

    /// @brief Removes metadata from the file
    /// @param key Metadata key to be removed
    void remove_metadata(const std::string &key);

    /// @brief Adds a buffer of events to be written
    /// @param begin Pointer to the beginning of the buffer
    /// @param end Pointer to the end of the buffer
    /// @warning The events must be chronologically ordered inside the buffer, and
    ///          in between calls. An exception will be thrown if the first event of the
    ///          added buffer is older than the timestamp of the last event of the last
    ///          added buffer.
    bool add_events(const EventCD *begin, const EventCD *end);

    /// @brief Adds a buffer of events to be written
    /// @overload
    bool add_events(const EventExtTrigger *begin, const EventExtTrigger *end);

    /// @brief For internal use
    class Private;

    /// @brief For internal use
    Private &get_pimpl();

private:
    virtual void open_impl(const std::string &path) = 0;
    virtual void close_impl()                       = 0;
    virtual bool is_open_impl() const               = 0;

    virtual void add_metadata_impl(const std::string &key, const std::string &value) = 0;
    virtual void add_metadata_map_from_camera_impl(const Camera &camera)             = 0;
    virtual void remove_metadata_impl(const std::string &key)                        = 0;

    virtual bool add_events_impl(const EventCD *begin, const EventCD *end)                 = 0;
    virtual bool add_events_impl(const EventExtTrigger *begin, const EventExtTrigger *end) = 0;
    virtual void flush_impl()                                                              = 0;

    std::unique_ptr<Private> pimpl_;
};

} // namespace Metavision

#endif /* METAVISION_SDK_DRIVER_EVENT_FILE_WRITER_H */
