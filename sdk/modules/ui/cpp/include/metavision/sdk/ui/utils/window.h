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

#ifndef METAVISION_SDK_UI_WINDOW_H
#define METAVISION_SDK_UI_WINDOW_H

#include <opencv2/core.hpp>

#include "metavision/sdk/ui/utils/base_window.h"

namespace Metavision {

/// @brief A window that can be used to display images from any thread
///
/// This window has no internal rendering thread, meaning that the images are displayed at the same frequency as the one
/// of the @ref Window::show method.
///
/// @warning The constructor and destructor of this class must only be called from the main thread
class Window : public BaseWindow {
public:
    /// @brief Constructs a new Window
    /// @param title The window's title
    /// @param width Width of the window at starting time (can be resized later on) and width of the images that will be
    /// displayed
    /// @param height Height of the window at starting time (can be resized later on) and height of the images that will
    /// be displayed
    /// @param mode The color rendering mode (i.e. either GRAY or BGR). Cannot be modified.
    /// @warning Must only be called from the main thread
    Window(const std::string &title, int width, int height, RenderMode mode);

    /// @brief Destructor
    /// @warning Must only be called from the main thread
    virtual ~Window();

    /// @brief Displays an image
    /// @param image The image to display
    /// @param auto_poll If True, events in this window's queue are dequeued and processed. If false,
    /// @ref BaseWindow::poll_events must explicitly be called.
    void show(const cv::Mat &image, bool auto_poll = true);
};

} // namespace Metavision

#endif // METAVISION_SDK_UI_WINDOW_H
