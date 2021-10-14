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

#include <map>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <metavision/sdk/base/utils/sdk_log.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/mt_window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

#include "constants.h"

int main() {
    const auto img = cv::imread(image_path);

    // Convert input image to grayscale for use in window 1
    cv::Mat img1;
    cv::cvtColor(img, img1, cv::COLOR_BGR2GRAY);

    /// [SIMPLE_WINDOW_CREATION_BEGIN]
    Metavision::Window w1("Window GRAY", img.cols, img.rows, Metavision::Window::RenderMode::GRAY);
    Metavision::MTWindow w2("MTWindow BGR", img.cols, img.rows, Metavision::Window::RenderMode::BGR);
    /// [SIMPLE_WINDOW_CREATION_END]

    /// [SIMPLE_WINDOW_W1_SUBSCRIPTION_BEGIN]
    // An example of a Key callback. It will be called only when a key is pressed while the window 1 has the focus.
    w1.set_keyboard_callback([&w1](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
        if (action == Metavision::UIAction::RELEASE) {
            const auto it = key_to_names.find(key);

            if (it != key_to_names.end())
                MV_LOG_INFO("[KEYBOARD EVENT]") << " " << it->second;

            if (key == Metavision::UIKeyEvent::KEY_ESCAPE)
                w1.set_close_flag();
        }
    });

    w1.set_mouse_callback([](Metavision::UIMouseButton button, Metavision::UIAction action, int mods) {
        if (action == Metavision::UIAction::RELEASE) {
            const auto it = button_to_names.find(button);

            if (it != button_to_names.cend())
                MV_LOG_INFO("[MOUSE EVENT]") << " " << it->second;
        }
    });
    /// [SIMPLE_WINDOW_W1_SUBSCRIPTION_END]

    /// [SIMPLE_WINDOW_W2_SUBSCRIPTION_BEGIN]
    // An example of a mouse cursor callback. It will be called whenever the mouse's cursor is on top of the window 2.
    w2.set_cursor_pos_callback(
        [](double x, double y) { MV_LOG_INFO("[MOUSE CURSOR EVENT]") << " x:" << x << "y:" << y; });

    // Another Key callback called when a key is pressed while the window 2 has the focus.
    w2.set_keyboard_callback([&w2](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
        if (action == Metavision::UIAction::RELEASE && key == Metavision::UIKeyEvent::KEY_ESCAPE)
            w2.set_close_flag();
    });
    /// [SIMPLE_WINDOW_W2_SUBSCRIPTION_END]

    /// [SIMPLE_WINDOW_W1_RENDERING_LOOP_BEGIN]
    // The window 1 will be rendered in a separate thread
    auto rendering_thread_w1 = std::thread([&]() {
        // Continue until one of the two windows is asked to close
        while (!w1.should_close() && !w2.should_close()) {
            // This call will:
            // - Poll and process the events received by the window 1. The attached callbacks are processed here in this
            // thread.
            // - Immediately update the displayed image.
            w1.show(img1);

            // Here we make the window 1 have a 40Hz refresh rate
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    });
    /// [SIMPLE_WINDOW_W1_RENDERING_LOOP_END]

    /// [SIMPLE_WINDOW_W2_RENDERING_LOOP_BEGIN]
    // The window 2 will be updated asynchronously from the main thread (i.e. the image will be updated in the internal
    // rendering thread).
    // Continue until one of the two windows is asked to close
    while (!w1.should_close() && !w2.should_close()) {
        // Poll all the events from the system and send them to the corresponding windows' internal queues.
        Metavision::EventLoop::poll_and_dispatch();

        auto img2 = img.clone();

        // This call will:
        // - Poll and process the events received by the window 2. The attached callbacks are processed here in the main
        // thread.
        // - Asynchronously update the displayed image.
        w2.show_async(img2);
    }
    /// [SIMPLE_WINDOW_W2_RENDERING_LOOP_END]

    if (rendering_thread_w1.joinable())
        rendering_thread_w1.join();

    return 0;
}
