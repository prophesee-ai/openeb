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

#ifndef METAVISION_SDK_UI_MT_WINDOW_H
#define METAVISION_SDK_UI_MT_WINDOW_H

#include <thread>
#include <opencv2/core.hpp>

#include "metavision/sdk/ui/utils/base_window.h"

namespace Metavision {

/// @brief Window using its own rendering thread to render images
///
/// Images are displayed at a fixed frequency (i.e. the screen's refresh one) by the internal rendering thread.
///
/// @warning The constructor and destructor of this class must only be called from the main thread
class MTWindow : public BaseWindow {
public:
    /// @brief Constructor
    /// @param title The window's title
    /// @param width Width of the window at starting time (can be resized later on) and width of the images that will be
    /// displayed
    /// @param height Height of the window at starting time (can be resized later on) and height of the images that will
    /// be displayed
    /// @param mode The color rendering mode (i.e. either GRAY or BGR). Cannot be modified.
    /// @warning Must only be called from the main thread
    MTWindow(const std::string &title, int width, int height, RenderMode mode);

    /// @brief Destructor
    /// @warning Must only be called from the main thread
    virtual ~MTWindow();

    /// @brief Asynchronously displays an image
    ///
    /// Here asynchronously means that the image is not immediately displayed, but will be done later on by the internal
    /// rendering thread.
    /// This window uses a front/back buffers mechanism to avoid copying images.
    ///
    /// @param image The image to display. The image is passed as a non constant reference in order to be swapped with
    /// the front buffer and thus avoid useless copies.
    /// @param auto_poll If True, events in this window's queue are dequeued and processed. If false,
    /// @ref BaseWindow::poll_events must explicitly be called.
    /// @warning If @p auto_poll is True, the events are processed in this method's calling thread, not in the internal
    /// rendering one.
    void show_async(cv::Mat &image, bool auto_poll = true);

private:
    /// @brief The internal rendering thread
    void rendering_loop();

    /// @brief If the front buffer has been updated, swaps the front and back buffers and uploads the back buffer to the
    /// GPU.
    void upload_texture_if_updated();

    bool has_been_updated_;
    std::mutex swap_mtx_;
    cv::Mat front_;
    cv::Mat back_;

    std::thread rendering_loop_;
};

} // namespace Metavision

#endif // METAVISION_SDK_UI_MT_WINDOW_H
