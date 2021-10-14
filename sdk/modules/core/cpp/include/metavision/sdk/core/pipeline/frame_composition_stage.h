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

#ifndef METAVISION_SDK_CORE_FRAME_COMPOSITION_STAGE_H
#define METAVISION_SDK_CORE_FRAME_COMPOSITION_STAGE_H

#include <opencv2/opencv.hpp>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/pipeline/base_stage.h"
#include "metavision/sdk/core/utils/frame_composer.h"

namespace Metavision {

/// @brief Class for composing a frame out of multiple frames.
///
/// It connects stages together to display side by side their image streams at a fixed refresh rate. Each sub-frame
/// coming from the input stages is rendered to its specified coordinates in the composed output frame.
///
/// The FPS given by the user determines at which frequency the full image must be updated (in the data's clock, not in
/// the system's one). However, input stages are not necessarily synchronous and might have different output
/// frequencies. This class acts like a synchronizer and displays at each multiple of dt = 1/FPS only the most recent
/// sub-image for each input stage.
///
/// Nullptrs images can be used as temporal markers, so that input stages can let the class know there are no
/// available data to display for this input at time ts. In that case, the corresponding part in the whole image won't
/// be updated.
class FrameCompositionStage : public BaseStage {
public:
    using FramePool = SharedObjectPool<cv::Mat>;
    using FramePtr  = FramePool::ptr_type;
    using Input     = std::pair<timestamp, FramePtr>;
    using Output    = std::pair<timestamp, FramePtr>;

public:
    FrameCompositionStage(int fps, timestamp max_ts_range = 5000000, const cv::Vec3b &bg_color = cv::Vec3b(0, 0, 0));

    /// @brief Sets up the frame composer to put the frame produced by @p prev_frame_stage at the given location
    /// @param prev_frame_stage Stage producing the frames
    /// @param x X-position of the top-left corner of the image in the composition
    /// @param y Y-position of the top-left corner of the image in the composition
    /// @param width Width of the (possibly scaled) frame inside the composed frame
    /// @param height Height of the (possibly scaled) frame inside the composed frame
    /// @param enable_crop Whether to enable cropping the image to the specified width and height (maintains the center)
    /// @param gray_to_color_options Options used to rescale and/or apply a colormap on the grayscale image
    void add_previous_frame_stage(
        BaseStage &prev_frame_stage, int x, int y, int width, int height, bool enable_crop = false,
        const FrameComposer::GrayToColorOptions &gray_to_color_options = FrameComposer::GrayToColorOptions());

    /// @brief Gets the underlying frame composer algorithm
    /// @return @ref FrameComposer & the frame composer algorithm
    FrameComposer &frame_composer();

private:
    struct SourceInfo {
        // Member that will be set to true when the source has finished producing frames
        bool finished_producing = false;

        // Reference to the last frame received from the source
        FramePtr ptr_to_last_frame_received = nullptr;

        // Queue containing the copied frames
        std::queue<std::pair<timestamp, cv::Mat>> saved_frames_queue_;

        // More recent timestamp for which both:
        //  - an update of the composed frame is required (i.e. multiple of the composer's period), and
        //  - the needed information has been received from the source
        timestamp ts_last_frame_saved = 0;

        // Queue containing the references to the input frames received (the ones that have not been copied)
        std::queue<std::pair<timestamp, FramePtr>> references_to_input_frames_queue_;

        // Returns true if we can copy the next frame received
        bool can_copy(timestamp next_frame_ts, timestamp max_ts_range);

        // Adds frame
        void add_frame(timestamp ts, FrameCompositionStage::FramePtr &ptr, timestamp next_frame_ts,
                       timestamp max_ts_range);

        // Function called when we receive a frame with a timestamp t < source_next_frame_ts
        void update_with_past_ptr(FrameCompositionStage::FramePtr &ptr);

        // Function called when we receive a frame with a timestamp t == source_next_frame_ts
        // Returns true if we can directly update the composed image
        bool update_with_current_ptr(FrameCompositionStage::FramePtr &ptr, timestamp source_next_frame_ts,
                                     timestamp composer_next_frame_ts, timestamp max_ts_range);

        // Function called when we receive a frame with a timestamp t > source_next_frame_ts
        void update_with_future_ptr(FrameCompositionStage::FramePtr &ptr, timestamp ptr_ts,
                                    timestamp source_next_frame_ts, timestamp frame_period,
                                    timestamp composer_next_frame_ts, timestamp max_ts_range);
    };

    /// @brief Processes a new frame coming from a given stage
    ///
    /// @param frame_ref ID referring to a sub-image of the frame composer
    /// @param data Timestamped shared_pointer of a new frame
    void consume_frame(int frame_ref, const boost::any &data);

    void set_stage_status(BaseStage &stage, const boost::any &data);

    /// @brief Updates the composed image stored in the frame composer
    ///
    /// REMARK : this method makes the assumption that we are capable of updating the frame for ts next_frame_ts_, i.e.
    ///          each input source has either provided the update for next_frame_ts_ (or greater), or finished
    ///          producing
    ///
    /// @return true if there still are frames to consume, false otherwise
    bool update_composed_image();

    /// @brief Produces the composed image (which has to be ready - i.e. already updated)
    void produce_composed_frame();

    /// @brief Checks if we received the update for next_frame_ts_ from all sources
    ///
    /// @return true if composer is ready to generate frame for next_frame_ts_
    bool got_all_updates();

    const timestamp frame_period_;
    timestamp next_frame_ts_;
    const timestamp max_ts_range_;
    std::unordered_map<BaseStage *, int> stages_map_; //< Index map using BaseStage pointer as key

    FrameComposer frame_composer_;

    std::unordered_map<int, SourceInfo> sources_info_map_;

    FramePool output_frame_pool_;
    FramePtr produced_frame_ptr_;
};

} // namespace Metavision

#include "detail/frame_composition_stage_impl.h"

#endif // METAVISION_SDK_CORE_FRAME_COMPOSITION_STAGE_H
