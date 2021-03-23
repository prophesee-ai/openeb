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

#include <string>

#include "metavision/sdk/ui/utils/base_window.h"

namespace Metavision {

class MTWindow;
class Window;

/// @brief Parent class used to implement a context manager for the Window and MTWindow classes
class BaseWindowWrapper {
public:
    /// @brief Constructor that stores the arguments needed to instantiate a new BaseWindow
    /// @param title The window's title
    /// @param width Width of the window at starting time (can be resized later on) and width of the images that will be
    /// displayed
    /// @param height Height of the window at starting time (can be resized later on) and height of the images that will
    /// be displayed
    /// @param mode The color rendering mode (i.e. either GRAY or BGR). Cannot be modified
    BaseWindowWrapper(const std::string &title, int width, int height, BaseWindow::RenderMode mode);

    /// @brief Destructor
    ///
    /// It also make sure the window has been properly closed
    virtual ~BaseWindowWrapper();

    /// @brief Gets a pointer to the current displayed window or a nullptr if the window isn't open
    /// @tparam Window type. Either BaseWindow, Window or MTWindow
    template<typename T>
    T *get();

    /// @brief Gets a const pointer to the current displayed window or a nullptr if the window isn't open
    /// @tparam Window type. Either BaseWindow, Window or MTWindow
    template<typename T>
    const T *get() const;

    /// @brief Destructs the current displayed BaseWindow
    void exit();

protected:
    std::string title_;
    int width_;
    int height_;
    BaseWindow::RenderMode mode_;

    BaseWindow *ptr_; ///< Pointer to the window to display
};

/// @brief Class that implements a context manager for the Window class
class WindowWrapper : public BaseWindowWrapper {
public:
    /// @brief Constructor that stores the arguments needed to instantiate a new Window
    WindowWrapper(const std::string &title, int width, int height, BaseWindow::RenderMode mode);

    /// @brief Instantiates a new Window
    void enter();
};

/// @brief Class that implements a context manager for the MTWindow class
class MTWindowWrapper : public BaseWindowWrapper {
public:
    /// @brief Constructor that stores the arguments needed to instantiate a new MTWindow
    MTWindowWrapper(const std::string &title, int width, int height, BaseWindow::RenderMode mode);

    /// @brief Instantiates a new MTWindow
    void enter();
};

} // namespace Metavision
