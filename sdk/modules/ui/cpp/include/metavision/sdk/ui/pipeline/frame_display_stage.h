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

#ifndef METAVISION_SDK_CORE_FRAME_DISPLAY_STAGE_H
#define METAVISION_SDK_CORE_FRAME_DISPLAY_STAGE_H

#include <chrono>
#include <boost/any.hpp>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/pipeline/base_stage.h"
#include "metavision/sdk/core/pipeline/pipeline.h"
#include "metavision/sdk/ui/utils/window.h"
#include "metavision/sdk/ui/utils/event_loop.h"

namespace Metavision {

/// @brief Stage that displays the input frame in a window
///
/// The window is refreshed every time a new frame is received.
class FrameDisplayStage : public BaseStage {
public:
    using FramePool = SharedObjectPool<cv::Mat>;
    using FramePtr  = FramePool::ptr_type;
    using FrameData = std::pair<timestamp, FramePtr>;

    /// @brief Constructs a new frame display stage
    /// @param title Window's title
    /// @param width Window's initial width
    /// @param height Window's initial height
    /// @param mode Window's rendering mode (i.e. either BGR or GRAY). Cannot be changed afterwards
    /// @param auto_exit Flag indicating if the application automatically closes if the user presses 'Q' or 'ESCAPE'
    /// @warning Must only be called from the main thread
    FrameDisplayStage(const std::string &title, int width, int height,
                      Window::RenderMode mode = Window::RenderMode::BGR, bool auto_exit = true);

    /// @brief Constructs a new frame display stage given an explicit previous stage
    /// @param prev_stage Stage producing the input image for this display stage
    /// @param title Window's title
    /// @param width Window's initial width
    /// @param height Window's initial height
    /// @param mode Window's rendering mode (i.e. either BGR or GRAY). Cannot be changed afterwards
    /// @param auto_exit Flag indicating if the application automatically closes if the user presses 'Q' or 'ESCAPE'
    /// @warning Must only be called from the main thread
    FrameDisplayStage(BaseStage &prev_stage, const std::string &title, int width, int height,
                      Window::RenderMode mode = Window::RenderMode::BGR, bool auto_exit = true);

    /// @brief Destructor
    /// @warning Must only be called from the main thread
    ~FrameDisplayStage();

    /// @brief Sets a callback that is called when the user presses a key
    ///
    /// @note The callback is only called when the window has the focus
    /// @param cb The callback to call
    void set_key_callback(const Window::KeyCallback &cb);

private:
    void init(bool auto_exit);

    // Key pressed callback
    Window window_;
    Window::KeyCallback on_key_cb_;
    static std::uint32_t instance_counter_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_FRAME_DISPLAY_STAGE_H
