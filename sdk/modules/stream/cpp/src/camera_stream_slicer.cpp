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

#include <filesystem>

#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/sdk/stream/camera_stream_slicer.h"
#include "metavision/hal/facilities/i_camera_synchronization.h"

namespace Metavision {
bool Slice::operator==(const Slice &other) const {
    return events == other.events && triggers == other.triggers;
}

CameraStreamSlicer::CameraStreamSlicer(Camera &&camera, const SliceCondition &slice_condition, size_t max_queue_size) :
    queue_(std::make_unique<ConcurrentQueue<Slice>>(max_queue_size)), camera_(std::move(camera)) {
    if (camera_.is_running()) {
        throw std::runtime_error(
            "Camera is already running. Cannot create a CameraStreamSlicer from a running camera.");
    }

    event_buffer_pool_   = SharedObjectPool<std::vector<EventCD>>::make_unbounded();
    trigger_buffer_pool_ = SharedObjectPool<std::vector<EventExtTrigger>>::make_unbounded();
    curt_event_buffer_   = event_buffer_pool_.acquire();
    curt_trigger_buffer_ = trigger_buffer_pool_.acquire();

    slicer_.set_slicing_condition(slice_condition);
}

CameraStreamSlicer::CameraStreamSlicer(CameraStreamSlicer &&slicer) noexcept :
    queue_(std::move(slicer.queue_)),
    event_buffer_pool_(std::move(slicer.event_buffer_pool_)),
    trigger_buffer_pool_(std::move(slicer.trigger_buffer_pool_)),
    curt_event_buffer_(std::move(slicer.curt_event_buffer_)),
    curt_trigger_buffer_(std::move(slicer.curt_trigger_buffer_)),
    slicer_(std::move(slicer.slicer_)),
    camera_(std::move(slicer.camera_)) {}

CameraStreamSlicer &CameraStreamSlicer::operator=(CameraStreamSlicer &&slicer) noexcept {
    queue_               = std::move(slicer.queue_);
    event_buffer_pool_   = std::move(slicer.event_buffer_pool_);
    trigger_buffer_pool_ = std::move(slicer.trigger_buffer_pool_);
    curt_event_buffer_   = std::move(slicer.curt_event_buffer_);
    curt_trigger_buffer_ = std::move(slicer.curt_trigger_buffer_);
    slicer_              = std::move(slicer.slicer_);
    camera_              = std::move(slicer.camera_);

    return *this;
}

CameraStreamSlicer::~CameraStreamSlicer() {
    if (queue_) {
        queue_->close();
        camera_.stop();
    }
}

CameraStreamSlicer::SliceIterator CameraStreamSlicer::begin() {
    init_slicing();

    return SliceIterator(queue_);
}

CameraStreamSlicer::SliceIterator CameraStreamSlicer::end() {
    return SliceIterator();
}

const Camera &CameraStreamSlicer::camera() const {
    return camera_;
}

void CameraStreamSlicer::init_slicing() {
    camera_.cd().add_callback([this](const auto &begin, const auto &end) {
        slicer_.process_events(begin, end, [this](const auto &slice_begin, const auto &slice_end) {
            curt_event_buffer_->insert(curt_event_buffer_->end(), slice_begin, slice_end);
        });
    });

    camera_.ext_trigger().add_callback([this](const auto &begin, const auto &end) {
        curt_trigger_buffer_->insert(curt_trigger_buffer_->end(), begin, end);
    });

    camera_.add_status_change_callback([this](const CameraStatus &status) {
        if (status == CameraStatus::STOPPED) {
            slicer_.flush();
            queue_->close();
        }
    });

    try {
        auto decoder = camera_.get_device().get_facility<I_EventsStreamDecoder>();

        decoder->add_time_callback([this](timestamp t) { slicer_.notify_elapsed_time(t); });
    } catch (const CameraException &e) { MV_LOG_TRACE() << e.what(); }

    slicer_.set_on_new_slice_callback([this](auto status, auto t, auto nevents) {
        const bool new_slice_added = queue_->emplace({status, t, nevents, curt_event_buffer_, curt_trigger_buffer_});
        if (new_slice_added) {
            curt_event_buffer_   = event_buffer_pool_.acquire();
            curt_trigger_buffer_ = trigger_buffer_pool_.acquire();

            curt_event_buffer_->clear();
            curt_trigger_buffer_->clear();
        }
    });

    camera_.start();
}

} // namespace Metavision