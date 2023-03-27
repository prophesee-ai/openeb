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

#ifndef METAVISION_SDK_DRIVER_EVENT_FILE_WRITER_INTERNAL_H
#define METAVISION_SDK_DRIVER_EVENT_FILE_WRITER_INTERNAL_H

#include <mutex>
#include "metavision/sdk/base/utils/object_pool.h"
#include "metavision/sdk/core/utils/threaded_process.h"
#include "metavision/sdk/driver/event_file_writer.h"

namespace Metavision {

class EventFileWriter::Private {
public:
    Private(EventFileWriter &writer, const std::string &path);

    const std::string &get_path() const;

    void open(const std::string &path);
    void close();
    bool is_open() const;
    void flush();

    bool add_events(const EventCD *begin, const EventCD *end);
    bool add_events(const EventExtTrigger *begin, const EventExtTrigger *end);

    void add_metadata(const std::string &key, const std::string &value);
    void add_metadata_map_from_camera(const Camera &camera);
    void remove_metadata(const std::string &key);

    mutable std::mutex mutex_;
    EventFileWriter &writer_;
    timestamp last_cd_ts_, last_ext_trigger_ts_;
    std::string path_;

    static constexpr size_t kMaxEventCDBufferSize      = 64384;
    static constexpr size_t kMaxEventTriggerBufferSize = 1028;
    using EventCDBufferPool                            = SharedObjectPool<std::vector<EventCD>>;
    using EventCDBufferPtr                             = EventCDBufferPool::ptr_type;
    using EventExtTriggerBufferPool                    = SharedObjectPool<std::vector<EventExtTrigger>>;
    using EventExtTriggerBufferPtr                     = EventExtTriggerBufferPool::ptr_type;
    EventCDBufferPool cd_buffer_pool_;
    EventCDBufferPtr cd_buffer_ptr_;
    EventExtTriggerBufferPool ext_trigger_buffer_pool_;
    EventExtTriggerBufferPtr ext_trigger_buffer_ptr_;
    ThreadedProcess writer_thread_;
};

} // namespace Metavision

#endif /* METAVISION_SDK_DRIVER_EVENT_FILE_WRITER_INTERNAL_H */