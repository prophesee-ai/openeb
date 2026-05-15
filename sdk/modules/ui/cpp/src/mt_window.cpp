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

#include "metavision/sdk/ui/utils/mt_window.h"
#include "metavision/sdk/ui/detail/texture_utils.h"

namespace Metavision {

MTWindow::MTWindow(const std::string &title, int width, int height, RenderMode mode) :
    BaseWindow(title, width, height, mode) {
    has_been_updated_ = false;
    rendering_loop_   = std::thread(&MTWindow::rendering_loop, this);
}

MTWindow::~MTWindow() noexcept {
    if (rendering_loop_.joinable()) {
        glfwSetWindowShouldClose(glfw_window_, GLFW_TRUE);

        rendering_loop_.join();
    }
}

void MTWindow::show_async(cv::Mat &image, bool auto_poll) {
    assert((render_mode_ == RenderMode::GRAY && image.channels() == 1) ||
           (render_mode_ == RenderMode::BGR && image.channels() == 3));

    if (auto_poll)
        poll_events();

    std::lock_guard<std::mutex> lock(swap_mtx_);

    // pre-allocate the frame for the next time
    front_.create(image.size(), image.type());

    cv::swap(image, front_);
    has_been_updated_ = true;
}

void MTWindow::rendering_loop() {
    glfwMakeContextCurrent(glfw_window_);

    // Enable the V-Sync
    glfwSwapInterval(1);

    while (!glfwWindowShouldClose(glfw_window_)) {
        upload_texture_if_updated();

        draw_background_texture();
    }

    glfwMakeContextCurrent(nullptr);
}

void MTWindow::upload_texture_if_updated() {
    bool do_upload = false;

    {
        std::lock_guard<std::mutex> lock(swap_mtx_);

        if (has_been_updated_) {
            cv::swap(front_, back_);
            has_been_updated_ = false;
            do_upload         = !back_.empty();
        }
    }

    if (do_upload)
        detail::upload_texture(back_, tex_id_);
}

} // namespace Metavision
