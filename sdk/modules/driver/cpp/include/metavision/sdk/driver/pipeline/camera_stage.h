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

#ifndef METAVISION_SDK_DRIVER_CAMERA_STAGE_H
#define METAVISION_SDK_DRIVER_CAMERA_STAGE_H

#include "metavision/sdk/driver/camera.h"
#include "metavision/sdk/core/pipeline/base_stage.h"
#include "metavision/sdk/core/algorithms/shared_cd_events_buffer_producer_algorithm.h"

namespace Metavision {

/// @brief Producing stage to generate @ref EventCD buffers and/or @ref EventExtTrigger from a @ref Camera.
///
/// The CD events buffer producing part uses the @ref SharedEventsBufferProducerAlgorithm allowing to tune further the
/// way you want to build the CD events buffers.
///
/// Default usage:
/// @code {.cpp}
/// int main(void) {
///     // Opens a camera
///     auto cam = Camera::from_first_available();
///
///     // Creates a pipeline object
///     Pipeline p;
///
///     // Add the Camera producing stage to the pipeline
///     auto &cam_stage = p.add_stage(std::make_unique<CameraStage>(std::move(cam)));
///
///     // Runs the pipeline
///     p.run();
///
///     return 0;
/// }
/// @endcode
///
/// Using @ref SharedEventsBufferProducerAlgorithm capabilities:
/// @code {.cpp}
/// int main(void) {
///     auto camera = Camera::from_first_available();
///
///     // Creates a pipeline object
///     Pipeline p;
///
///     SharedEventsBufferProducerParameters parameters;
///     parameters.buffers_time_slice_us_ = 5000;
///
///     // Will produce buffers containing 5 ms of events for the next consuming stage.
///     auto &cam_stage = p.add_stage(std::make_unique<CameraStage>(std::move(cam), parameters));
///
///     // Runs the pipeline
///     p.run();
///
///     return 0;
/// }
/// @endcode
///
/// @sa Pipeline
/// @ref BaseStage
class CameraStage : public BaseStage {
public:
    using CdBufferProducer       = SharedCdEventsBufferProducerAlgorithm;
    using EventTriggerBuffer     = std::vector<EventExtTrigger>;
    using EventTriggerBufferPool = SharedObjectPool<EventTriggerBuffer>;
    using EventTriggerBufferPtr  = EventTriggerBufferPool::ptr_type;

    /// @brief Constructor
    /// @param camera Camera producing the input events
    /// @param buffer_producer_parameters Buffer to use by the CD events buffer producing part
    /// @param enable_cd_events_callback If true, enables the callback of CD events
    CameraStage(Camera &&camera, SharedEventsBufferProducerParameters buffer_producer_parameters,
                bool enable_cd_events_callback = true) :
        ext_trigger_buffer_pool_(EventTriggerBufferPool::make_bounded()) {
        if (enable_cd_events_callback) {
            init_cd_processing(buffer_producer_parameters);
        }

        init();
    }

    /// @brief Constructor
    /// @param camera Camera producing the input events
    /// @param buffer_duration_ms Time slice (in milliseconds) to use in the @ref SharedEventsBufferProducerParameters
    /// used by the CD events buffer producing part
    /// @param enable_cd_events_callback If true, enables the callback of CD events
    CameraStage(Camera &&camera, timestamp buffer_duration_ms = 1, bool enable_cd_events_callback = true) :
        camera_(std::move(camera)), ext_trigger_buffer_pool_(EventTriggerBufferPool::make_bounded()) {
        SharedEventsBufferProducerParameters buffer_producer_parameters;
        buffer_producer_parameters.buffers_events_count_       = 0;
        buffer_producer_parameters.buffers_pool_size_          = 64;
        buffer_producer_parameters.buffers_time_slice_us_      = static_cast<uint32_t>(buffer_duration_ms * 1000);
        buffer_producer_parameters.buffers_preallocation_size_ = 50000;

        if (enable_cd_events_callback) {
            init_cd_processing(buffer_producer_parameters);
        }
        init();
    }

    /// @brief Adds trigger events
    /// @param begin @ref EventExtTrigger pointer to the beginning of the buffer
    /// @param end @ref EventExtTrigger pointer to the end of the buffer
    void add_ext_trigger_events(const EventExtTrigger *begin, const EventExtTrigger *end) {
        cur_ext_trigger_buffer_->insert(std::end(*cur_ext_trigger_buffer_), begin, end);
        produce(cur_ext_trigger_buffer_);
        cur_ext_trigger_buffer_ = ext_trigger_buffer_pool_.acquire();
        cur_ext_trigger_buffer_->clear();
    }

    /// @brief Gets @ref Camera
    /// @return Camera used in the stage
    Camera &camera() {
        return camera_;
    }

private:
    void init_cd_processing(const SharedEventsBufferProducerParameters &params) {
        cd_buffer_pool_.reset(new CdBufferProducer(
            params,
            [this](Metavision::timestamp ts, const CdBufferProducer::SharedEventsBuffer buffer) { produce(buffer); }));
        camera_.cd().add_callback(
            [this](const EventCD *begin, const EventCD *end) { cd_buffer_pool_->process_events(begin, end); });
    }

    void init() {
        cur_ext_trigger_buffer_ = ext_trigger_buffer_pool_.acquire();
        cur_ext_trigger_buffer_->clear();

        camera_.add_status_change_callback([this](const CameraStatus &status) {
            if (status == CameraStatus::STOPPED) {
                cd_buffer_pool_->flush();
                if (!cur_ext_trigger_buffer_->empty())
                    produce(cur_ext_trigger_buffer_);
                complete();
            }
        });

        set_starting_callback([this]() { camera_.start(); });
        set_stopping_callback([this]() { camera_.stop(); });
    }

    Camera camera_;
    std::unique_ptr<CdBufferProducer> cd_buffer_pool_;
    EventTriggerBufferPool ext_trigger_buffer_pool_;
    EventTriggerBufferPtr cur_ext_trigger_buffer_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_CAMERA_STAGE_H
