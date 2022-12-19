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
#include "metavision/sdk/ui/pipeline/frame_display_stage.h"

namespace Metavision {

std::uint32_t FrameDisplayStage::instance_counter_ = 0;

FrameDisplayStage::FrameDisplayStage(const std::string &title, int width, int height, Window::RenderMode mode,
                                     bool auto_exit) :
    window_(title, width, height, mode) {
    init(auto_exit);
}

FrameDisplayStage::FrameDisplayStage(BaseStage &prev_stage, const std::string &title, int width, int height,
                                     Window::RenderMode mode, bool auto_exit) :
    FrameDisplayStage(title, width, height, mode, auto_exit) {
    set_previous_stage(prev_stage);
}

FrameDisplayStage::~FrameDisplayStage() {
    --instance_counter_;
}

void FrameDisplayStage::set_key_callback(const Window::KeyCallback &cb) {
    on_key_cb_ = cb;
}

void FrameDisplayStage::init(bool auto_exit) {
    if (instance_counter_++ == 0) {
        set_setup_callback([this]() { pipeline().add_pre_step_callback([]() { EventLoop::poll_and_dispatch(); }); });
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

} // namespace Metavision