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

#include "metavision/sdk/ui/utils/window.h"
#include "metavision/sdk/ui/detail/texture_utils.h"

namespace Metavision {
Window::Window(const std::string &title, int width, int height, RenderMode mode) :
    BaseWindow(title, width, height, mode) {}

Window::~Window() noexcept {}

void Window::show(const cv::Mat &image, bool auto_poll) {
    assert((render_mode_ == RenderMode::GRAY && image.channels() == 1) ||
           (render_mode_ == RenderMode::BGR && image.channels() == 3));

    if (auto_poll)
        poll_events();

    auto *prev_context = glfwGetCurrentContext();

    glfwMakeContextCurrent(glfw_window_);

    detail::upload_texture(image, tex_id_);

    draw_background_texture();

    glfwMakeContextCurrent(prev_context);
}
} // namespace Metavision