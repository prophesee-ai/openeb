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

#ifndef METAVISION_SDK_STREAM_OFFLINE_STREAMING_CONTROL_INTERNAL_H
#define METAVISION_SDK_STREAM_OFFLINE_STREAMING_CONTROL_INTERNAL_H

#include "metavision/sdk/stream/camera.h"

namespace Metavision {

class EventFileReader;

class OfflineStreamingControl::Private {
public:
    static OfflineStreamingControl *build(EventFileReader &reader);

    bool is_ready() const;
    bool seek(timestamp);
    timestamp get_seek_start_time() const;
    timestamp get_seek_end_time() const;
    timestamp get_duration() const;

private:
    Private(EventFileReader &reader);

    EventFileReader &reader_;
};

} // namespace Metavision

#endif // METAVISION_SDK_STREAM_OFFLINE_STREAMING_CONTROL_INTERNAL_H
