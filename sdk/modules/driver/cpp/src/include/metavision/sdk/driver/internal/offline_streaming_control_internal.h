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

#ifndef METAVISION_SDK_DRIVER_OFFLINE_STREAMING_CONTROL_INTERNAL_H
#define METAVISION_SDK_DRIVER_OFFLINE_STREAMING_CONTROL_INTERNAL_H

#include "metavision/sdk/driver/camera.h"

namespace Metavision {
namespace Future {
class I_EventsStream;
class I_Decoder;
} // namespace Future

class OfflineStreamingControl::Private {
public:
    Private(Camera::Private &ptr);

    bool is_valid() const;
    bool is_ready() const;
    bool seek(timestamp);
    timestamp get_seek_start_time() const;
    timestamp get_seek_end_time() const;
    timestamp get_duration() const;

    Camera::Private &camera_priv_;
    Future::I_EventsStream *i_events_stream_;
    Future::I_Decoder *i_decoder_;
    mutable timestamp start_ts_, end_ts_, duration_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_OFFLINE_STREAMING_CONTROL_INTERNAL_H
