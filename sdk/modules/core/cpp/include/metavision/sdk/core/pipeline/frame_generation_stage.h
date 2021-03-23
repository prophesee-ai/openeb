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

#ifndef METAVISION_SDK_CORE_FRAME_GENERATION_STAGE_H
#define METAVISION_SDK_CORE_FRAME_GENERATION_STAGE_H

#include <memory>
#include <assert.h>
#include <opencv2/opencv.hpp>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/pipeline/base_stage.h"
#include "metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h"

namespace Metavision {

/// @brief Stage that generates an OpenCV frame out of @ref EventCD events.
class FrameGenerationStage : public BaseStage {
public:
    using FramePool = SharedObjectPool<cv::Mat>;
    using FramePtr  = FramePool::ptr_type;
    using Output    = std::pair<timestamp, FramePtr>;

    /// @brief Constructor
    /// @param width Width of the frame.
    /// @param height Height of the frame.
    /// @param accumulation_time_ms Accumulation time (in ms)
    /// @param fps The fps at which to generate the frames. The time reference used is the one from the input events
    /// @param palette The Prophesee's color palette to use
    /// @note If the fps is zero, @p accumulation_time_ms will be used instead as reference time to generate frames
    FrameGenerationStage(int width, int height, uint32_t accumulation_time_ms = 10, double fps = 0.,
                         const Metavision::ColorPalette &palette = BaseFrameGenerationAlgorithm::default_palette()) :
        // using a queue of 2 frames to not pre-compute too many frames in advance
        // which uses a lot of memory, and makes real-time interactions weird (due to the
        // interaction operating on frames which will be displayed only much later)
        frame_pool_(FramePool::make_bounded(2)) {
        const uint32_t accumulation_time_us = accumulation_time_ms * 1000;
        algo_ = std::make_unique<PeriodicFrameGenerationAlgorithm>(width, height, accumulation_time_us, fps, palette);
        algo_->set_output_callback([this](const timestamp ts, cv::Mat &f) {
            crt_frame_ptr_ = frame_pool_.acquire();
            cv::swap(f, *crt_frame_ptr_);

            produce(std::make_pair(ts, crt_frame_ptr_));
        });

        set_consuming_callback([this](const boost::any &data) { consume_cd_events(data); });
    }

    /// @brief Constructor
    /// @param prev_stage Previous stage.
    /// @param width Width of the frame.
    /// @param height Height of the frame.
    /// @param fps Target of frames to generate per second.
    FrameGenerationStage(BaseStage &prev_stage, int width, int height, int fps) :
        FrameGenerationStage(width, height, fps) {
        set_previous_stage(prev_stage);
    }

private:
    void consume_cd_events(const boost::any &data) {
        try {
            auto buffer = boost::any_cast<EventBufferPtr>(data);
            algo_->process_events(buffer->cbegin(), buffer->cend());
        } catch (boost::bad_any_cast &c) { MV_SDK_LOG_ERROR() << c.what(); }
    }

    // Frames
    FramePtr crt_frame_ptr_;
    FramePool frame_pool_;

    std::unique_ptr<PeriodicFrameGenerationAlgorithm> algo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_FRAME_GENERATION_STAGE_H
