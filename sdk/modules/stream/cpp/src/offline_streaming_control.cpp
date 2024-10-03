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

#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/utils/raw_file_config.h"
#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/internal/camera_internal.h"
#include "metavision/sdk/stream/event_file_reader.h"
#include "metavision/sdk/stream/offline_streaming_control.h"
#include "metavision/sdk/stream/internal/offline_streaming_control_internal.h"
#include <thread>

namespace Metavision {
OfflineStreamingControl *OfflineStreamingControl::Private::build(EventFileReader &reader) {
    return new OfflineStreamingControl(new Private(reader));
}

OfflineStreamingControl::Private::Private(EventFileReader &reader) : reader_(reader) {}

bool OfflineStreamingControl::Private::is_ready() const {
    timestamp start_ts, end_ts;
    return !reader_.seekable() || reader_.get_seek_range(start_ts, end_ts);
}

bool OfflineStreamingControl::Private::seek(timestamp ts) {
    return reader_.seek(ts);
}

timestamp OfflineStreamingControl::Private::get_seek_start_time() const {
    timestamp start_ts, end_ts;
    if (reader_.get_seek_range(start_ts, end_ts)) {
        return start_ts;
    }
    return -1;
}

timestamp OfflineStreamingControl::Private::get_seek_end_time() const {
    timestamp start_ts, end_ts;
    if (reader_.get_seek_range(start_ts, end_ts)) {
        return end_ts;
    }
    return -1;
}

timestamp OfflineStreamingControl::Private::get_duration() const {
    return reader_.get_duration();
}

OfflineStreamingControl::OfflineStreamingControl(Private *pimpl) : pimpl_(pimpl) {}

OfflineStreamingControl::~OfflineStreamingControl() {}

bool OfflineStreamingControl::is_ready() const {
    return pimpl_->is_ready();
}

bool OfflineStreamingControl::seek(timestamp ts) {
    return pimpl_->seek(ts);
}

timestamp OfflineStreamingControl::get_seek_start_time() const {
    return pimpl_->get_seek_start_time();
}

timestamp OfflineStreamingControl::get_seek_end_time() const {
    return pimpl_->get_seek_end_time();
}

timestamp OfflineStreamingControl::get_duration() const {
    return pimpl_->get_duration();
}

OfflineStreamingControl::Private &OfflineStreamingControl::get_pimpl() {
    return *pimpl_;
}
} // namespace Metavision
