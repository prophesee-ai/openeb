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
#include "metavision/sdk/ui/utils/window.h"

#include "window_wrapper.h"

namespace Metavision {

BaseWindowWrapper::BaseWindowWrapper(const std::string &title, int width, int height, BaseWindow::RenderMode mode) :
    title_(title), width_(width), height_(height), mode_(mode) {
    ptr_ = nullptr;
}

BaseWindowWrapper::~BaseWindowWrapper() {
    exit();
}

template<typename T>
T *BaseWindowWrapper::get() {
    return static_cast<T *>(ptr_);
}

template<typename T>
const T *BaseWindowWrapper::get() const {
    return static_cast<T *>(ptr_);
}

// Template instantiation
template BaseWindow *BaseWindowWrapper::get();
template Window *BaseWindowWrapper::get();
template MTWindow *BaseWindowWrapper::get();

template const BaseWindow *BaseWindowWrapper::get() const;
template const Window *BaseWindowWrapper::get() const;
template const MTWindow *BaseWindowWrapper::get() const;

void BaseWindowWrapper::exit() {
    delete ptr_;
    ptr_ = nullptr; // Avoid deleting twice the same ptr
}

WindowWrapper::WindowWrapper(const std::string &title, int width, int height, BaseWindow::RenderMode mode) :
    BaseWindowWrapper(title, width, height, mode) {}

void WindowWrapper::enter() {
    ptr_ = new Window(title_, width_, height_, mode_);
}

MTWindowWrapper::MTWindowWrapper(const std::string &title, int width, int height, BaseWindow::RenderMode mode) :
    BaseWindowWrapper(title, width, height, mode) {}

void MTWindowWrapper::enter() {
    ptr_ = new MTWindow(title_, width_, height_, mode_);
}

} // namespace Metavision
