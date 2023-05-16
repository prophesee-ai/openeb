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

#ifndef METAVISION_SDK_CORE_SIMPLE_DISPLAYER_H
#define METAVISION_SDK_CORE_SIMPLE_DISPLAYER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <functional>
#include <mutex>
#include <condition_variable>

namespace Metavision {

/// @brief Class representing a simple displayer
class [[deprecated("'SimpleDisplayer' class is deprecated since v4.1.0 and will be removed in future releases, "
                   "please use 'Window' or 'MTWindow' instead.")]] SimpleDisplayer {
public:
    using OnKeyPressedCb = std::function<void(int)>;

    /// @brief Constructor
    /// @param window_name Name of the window to display
    /// @param fps Frames per second to display
    SimpleDisplayer(const std::string window_name, int fps = 50);

    /// @brief Default destructor
    ~SimpleDisplayer() = default;

    /// @brief Quits the display
    void stop();

    /// @brief Updates the current frame by swap
    /// @param frame Frame to swap
    void swap_frame(cv::Mat & frame);

    /// @brief Updates the current frame by copy
    /// @param frame Frame to copy
    void copy_frame(const cv::Mat &frame);

    /// @brief Sets the callback that is called when a key is pressed
    ///	@param on_key_pressed_cb Function to call
    void set_on_key_pressed_cb(const OnKeyPressedCb &on_key_pressed_cb);

    /// @brief Runs the displayer. Should be called in the main thread
    void run();

private:
    // Start and stop mechanisms
    volatile bool should_stop_ = false;    ///< Bool to stop displayer generation from any thread
    volatile bool started_     = false;    ///< Bool to start the displayer
    std::condition_variable started_cond_; ///< Conditional variable to notify starting

    // Images
    cv::Mat front_img_;    ///< Img to be displayed
    cv::Mat middle_img_;   ///< Intermediate buffer to avoid blocking the thread
    std::mutex img_mutex_; ///< Mutex to swap the buffers
    bool updated_ = false; ///< Check to update the front img
    std::string window_name_;

    // Display
    int wait_time_ms_;

    // Key pressed callback
    OnKeyPressedCb on_key_pressed_cb_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_SIMPLE_DISPLAYER_H
