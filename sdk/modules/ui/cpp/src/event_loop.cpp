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

#include <thread>

#include "metavision/sdk/ui/utils/base_window.h"
#include "metavision/sdk/ui/utils/event_loop.h"

namespace Metavision {
void EventLoop::poll_and_dispatch(std::int64_t sleep_time_ms) {
    BaseWindow::poll_pending_events();

    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_ms));
}
} // namespace Metavision
