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

#include "metavision/hal/facilities/future/i_events_stream.h"
#include "metavision/hal/facilities/future/i_decoder.h"
#include "metavision/hal/utils/future/raw_file_config.h"
#include "metavision/sdk/driver/camera.h"
#include "metavision/sdk/driver/internal/camera_internal.h"
#include "metavision/sdk/driver/offline_streaming_control.h"
#include "metavision/sdk/driver/internal/offline_streaming_control_internal.h"
#include <thread>

namespace Metavision {
OfflineStreamingControl::Private::Private(Camera::Private &cam_priv) :
    camera_priv_(cam_priv), start_ts_(-1), end_ts_(-1), duration_(-1) {
    i_events_stream_ = camera_priv_.i_future_events_stream_;
    i_decoder_       = camera_priv_.i_future_decoder_;
}

bool OfflineStreamingControl::Private::is_valid() const {
    return i_events_stream_ && i_decoder_;
}

bool OfflineStreamingControl::Private::is_ready() const {
    if (!i_events_stream_) {
        return false;
    }
    if (start_ts_ >= 0 && end_ts_ >= 0) {
        return true;
    }
    return i_events_stream_->get_seek_range(start_ts_, end_ts_) == Future::I_EventsStream::IndexStatus::Good;
}

bool OfflineStreamingControl::Private::seek(timestamp ts) {
    if (!i_events_stream_) {
        return false;
    }
    if (start_ts_ < 0 || end_ts_ < 0) {
        if (i_events_stream_->get_seek_range(start_ts_, end_ts_) != Future::I_EventsStream::IndexStatus::Good) {
            return false;
        }
    }

    auto f = [this, ts] {
        timestamp ts_reached;
        if (i_events_stream_->seek(ts, ts_reached) == Future::I_EventsStream::SeekStatus::Success) {
            i_decoder_->reset_timestamp(ts_reached);
            camera_priv_.init_clocks();
            return true;
        }
        return false;
    };

    // if main loop is not running, we can directly seek without risks of the events stream being used
    // by another thread
    if (!camera_priv_.is_running_) {
        return f();
    }

    // otherwise, submit the seek callback to the camera thread and wait for it to be called
    auto cb = camera_priv_.add_events_stream_update_callback(f);
    return cb->wait();
}

timestamp OfflineStreamingControl::Private::get_seek_start_time() const {
    if (!i_events_stream_) {
        return false;
    }
    if (start_ts_ >= 0 && end_ts_ >= 0) {
        return start_ts_;
    }
    if (i_events_stream_->get_seek_range(start_ts_, end_ts_) == Future::I_EventsStream::IndexStatus::Good) {
        return start_ts_;
    }
    return -1;
}

timestamp OfflineStreamingControl::Private::get_seek_end_time() const {
    if (!i_events_stream_) {
        return false;
    }
    if (start_ts_ >= 0 && end_ts_ >= 0) {
        return end_ts_;
    }
    if (i_events_stream_->get_seek_range(start_ts_, end_ts_) == Future::I_EventsStream::IndexStatus::Good) {
        return end_ts_;
    }
    return -1;
}

timestamp OfflineStreamingControl::Private::get_duration() const {
    if (!i_events_stream_) {
        return false;
    }
    if (duration_ >= 0) {
        return duration_;
    }
    Camera cam = Camera::from_file(i_events_stream_->get_underlying_filename(), false, Future::RawFileConfig());
    cam.cd().add_callback([this](const EventCD *begin, const EventCD *end) { duration_ = std::prev(end)->t; });
    cam.offline_streaming_control().seek(get_seek_end_time());
    cam.start();
    while (cam.is_running()) {
        std::this_thread::yield();
    }
    return duration_;
}

template<>
OfflineStreamingControl::OfflineStreamingControl(Camera::Private &cam_priv) : pimpl_(new Private(cam_priv)) {}

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
