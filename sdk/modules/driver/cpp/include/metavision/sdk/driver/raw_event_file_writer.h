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

#ifndef METAVISION_SDK_DRIVER_RAW_EVENT_FILE_WRITER_H
#define METAVISION_SDK_DRIVER_RAW_EVENT_FILE_WRITER_H

#include <memory>
#include <string>
#include <unordered_map>
#include "metavision/sdk/driver/event_file_writer.h"

namespace Metavision {

class RAWEventFileWriter : public EventFileWriter {
public:
    RAWEventFileWriter(const std::string &path = std::string(),
                        const std::unordered_map<std::string, std::string> &metadata_map =
                            std::unordered_map<std::string, std::string>());
    ~RAWEventFileWriter() override;

    bool add_raw_data(const std::uint8_t *ptr, size_t size);

private:
    void open_impl(const std::string &path) override;
    void close_impl() override;
    bool is_open_impl() const override;
    void flush_impl() override;

    void add_metadata_impl(const std::string &key, const std::string &value) override;
    void add_metadata_map_from_camera_impl(const Camera &camera) override;
    void remove_metadata_impl(const std::string &key) override;

    bool add_events_impl(const EventCD *begin, const EventCD *end) override;
    bool add_events_impl(const EventExtTrigger *begin, const EventExtTrigger *end) override;

    class Private;
    std::unique_ptr<Private> pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_RAW_EVENT_FILE_WRITER_H
