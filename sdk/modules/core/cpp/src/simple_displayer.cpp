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

#include "metavision/sdk/core/utils/simple_displayer.h"

namespace Metavision {
SimpleDisplayer::SimpleDisplayer(const std::string window_name, int fps) : window_name_(window_name) {
    wait_time_ms_      = 1000 / fps;
    on_key_pressed_cb_ = [](int) {};
}

/// @brief Quits the display
void SimpleDisplayer::stop() {
    std::lock_guard<std::mutex> lock(img_mutex_);
    should_stop_ = true;

    started_cond_.notify_all();
}

/// @brief Updates current frame by swap
/// @param frame Frame to swap
void SimpleDisplayer::swap_frame(cv::Mat &frame) {
    std::lock_guard<std::mutex> lock(img_mutex_);
    std::swap(frame, middle_img_);
    updated_ = true;

    if (!started_) {
        started_ = true;
        started_cond_.notify_all();
    }
}

/// @brief Updates current frame by copy
/// @param frame Frame to copy
void SimpleDisplayer::copy_frame(const cv::Mat &frame) {
    std::lock_guard<std::mutex> lock(img_mutex_);
    frame.copyTo(middle_img_);
    updated_ = true;

    if (!started_) {
        started_ = true;
        started_cond_.notify_all();
    }
}

/// @brief Callback called when the display is exited
void SimpleDisplayer::set_on_key_pressed_cb(const OnKeyPressedCb &on_key_pressed_cb) {
    on_key_pressed_cb_ = on_key_pressed_cb;
}

/// @brief Runs displayer. Should be called in the main thread
void SimpleDisplayer::run() {
    {
        // Wait until we receive some data
        std::unique_lock<std::mutex> lock(img_mutex_);
        started_cond_.wait(lock, [this]() { return started_ || should_stop_; });
        if (should_stop_)
            return;
    }

    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);

    while (!should_stop_) {
        {
            std::lock_guard<std::mutex> lock(img_mutex_);
            if (updated_) {
                updated_ = false;
                std::swap(front_img_, middle_img_);
            }
        }

        cv::imshow(window_name_, front_img_);

        int key = cv::waitKey(wait_time_ms_);
        on_key_pressed_cb_(key);
    }
}

} // namespace Metavision
