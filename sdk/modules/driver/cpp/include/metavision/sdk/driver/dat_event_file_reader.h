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

#ifndef METAVISION_SDK_DRIVER_DAT_EVENT_FILE_READER_H
#define METAVISION_SDK_DRIVER_DAT_EVENT_FILE_READER_H

#include <memory>
#include "metavision/sdk/driver/event_file_reader.h"

namespace Metavision {

class Device;

class DATEventFileReader : public EventFileReader {
public:
    DATEventFileReader(const std::string &path);
    ~DATEventFileReader();

    bool seekable() const override;

private:
    bool read_impl() override;
    bool seek_impl(timestamp t) override;
    bool get_seek_range_impl(timestamp &min_t, timestamp &max_t) const override;
    timestamp get_duration_impl() const override;
    std::unordered_map<std::string, std::string> get_metadata_map_impl() const override;

    class Private;
    std::unique_ptr<Private> pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_RAW_EVENT_FILE_READER_H
