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
    FrameDisplayStage(const std::string &title, int width, int height,
                      Window::RenderMode mode = Window::RenderMode::BGR, bool auto_exit = true) :
        window_(title, width, height, mode) {
        init(auto_exit);
    }

    /// @brief Constructs a new frame display stage given an explicit previous stage
    /// @param prev_stage Stage producing the input image for this display stage
    /// @param title Window's title
    /// @param width Window's initial width
    /// @param height Window's initial height
    /// @param mode Window's rendering mode (i.e. either BGR or GRAY). Cannot be changed afterwards
    /// @param auto_exit Flag indicating if the application automatically closes if the user presses 'Q' or 'ESCAPE'
    FrameDisplayStage(BaseStage &prev_stage, const std::string &title, int width, int height,
                      Window::RenderMode mode = Window::RenderMode::BGR, bool auto_exit = true) :
        FrameDisplayStage(title, width, height, mode, auto_exit) {
        set_previous_stage(prev_stage);
    }

    /// @brief Sets a callback that is called when the user presses a key
    ///
    /// @note The callback is only called when the window has the focus
    /// @param cb The callback to call
    void set_key_callback(const Window::KeyCallback &cb) {
        on_key_cb_ = cb;
    }

private:
    void init(bool auto_exit) {
        static bool is_pre_step_cb_set = false;
        if (!is_pre_step_cb_set) {
            set_setup_callback(
                [this]() { pipeline().add_pre_step_callback([]() { EventLoop::poll_and_dispatch(); }); });
            is_pre_step_cb_set = true;
        }

        set_consuming_callback([this](const boost::any &data) {
            try {
                auto res    = boost::any_cast<FrameData>(data);
                timestamp t = res.first;
                FramePtr &f = res.second;
                if (f && !f->empty())
                    window_.show(*f);
            } catch (boost::bad_any_cast &c) { MV_SDK_LOG_ERROR() << c.what(); }
        });

        window_.set_keyboard_callback([this, auto_exit](UIKeyEvent key, int scancode, UIAction action, int mods) {
            on_key_cb_(key, scancode, action, mods);

            if (auto_exit) {
                if (action == UIAction::RELEASE) {
                    if (key == UIKeyEvent::KEY_ESCAPE || key == UIKeyEvent::KEY_Q)
                        this->pipeline().cancel();
                }
            }
        });

        on_key_cb_ = [](UIKeyEvent key, int scancode, UIAction action, int mods) {};
    }

    // Key pressed callback
    Window window_;
    Window::KeyCallback on_key_cb_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_FRAME_DISPLAY_STAGE_H
