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

#ifndef METAVISION_SDK_STREAM_RAW_EVT2_EVENT_FILE_WRITER_H
#define METAVISION_SDK_STREAM_RAW_EVT2_EVENT_FILE_WRITER_H

#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include "metavision/hal/utils/raw_file_header.h"
#include "metavision/sdk/stream/event_file_writer.h"

namespace Metavision {

class Evt2Encoder;

/// @brief EventFileWriter specialized class to write events to a RAW EVT2 file
/// @note This class only supports writing CD and Trigger events
class RAWEvt2EventFileWriter : public EventFileWriter {
public:
    /// @brief Constructor
    /// @param stream_width Width of the stream
    /// @param stream_height Height of the stream
    /// @param path Path to the file to write to
    /// @param enable_trigger_support If true, the writer will merge CD and Trigger event streams before writing them to
    /// the file, leading to some encoding latency. If false, CD event will be written as soon as they are added to the
    /// writer, and Trigger events will not be supported.
    /// @param metadata_map Map of metadata to add in the header of the written file
    /// @param max_events_add_latency Maximum latency for the addition of events, in camera time, beyond which the
    /// writer will assume it can safely encode early buffered events. By default, there is no limit, meaning the writer
    /// will buffer CD events until a least one trigger event is received, which may in certain cases lead to memory
    /// issues. Alternatively, a finite positive latency value can be specified, the writer will then assume absence of
    /// triggers if CD events have been buffered for a larger time, hence avoiding memory issues but with the risk of
    /// EVT2 encoding errors if trigger events are actually received with a higher latency.
    RAWEvt2EventFileWriter(int stream_width, int stream_height,
                           const std::filesystem::path &path = std::filesystem::path(),
                           bool enable_trigger_support       = false,
                           const std::unordered_map<std::string, std::string> &metadata_map =
                               std::unordered_map<std::string, std::string>(),
                           timestamp max_events_add_latency = std::numeric_limits<timestamp>::max());

    /// @brief Destructor
    ~RAWEvt2EventFileWriter() override;

private:
    void open_impl(const std::filesystem::path &path) override;
    void close_impl() override;
    bool is_open_impl() const override;
    void flush_impl() override;

    void add_metadata_impl(const std::string &key, const std::string &value) override;
    void add_metadata_map_from_camera_impl(const Camera &camera) override;
    void remove_metadata_impl(const std::string &key) override;

    bool add_events_impl(const EventCD *begin, const EventCD *end) override;
    bool add_events_impl(const EventExtTrigger *begin, const EventExtTrigger *end) override;

    void encode_buffered_events(bool flush_all_queued_events);
    void merge_encode_buffered_events(bool flush_all_queued_events);

    const bool exttrigger_support_enabled_;
    const timestamp max_events_add_latency_;
    RawFileHeader header_;
    bool header_written_ = false;
    std::ofstream ofs_;
    std::deque<EventExtTrigger> events_trigger_;
    std::deque<EventCD> events_cd_;
    timestamp ts_last_cd_      = std::numeric_limits<timestamp>::min(),
              ts_last_trigger_ = std::numeric_limits<timestamp>::min();
    std::unique_ptr<Evt2Encoder> encoder_;
};

} // namespace Metavision

#endif // METAVISION_SDK_STREAM_RAW_EVT2_EVENT_FILE_WRITER_H
