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

#ifndef METAVISION_SDK_UI_EVENT_LOOP_H
#define METAVISION_SDK_UI_EVENT_LOOP_H

#include <cstdint>

namespace Metavision {

/// @brief A static class used to dispatch system events to the windows they belong to
class EventLoop {
public:
    /// @brief Polls events from the system and pushes them into the corresponding windows' internal queue
    /// @param sleep_time_ms Amount of time in ms this call will wait after polling and dispatching the events
    /// @warning Must only be called from the main thread
    static void poll_and_dispatch(std::int64_t sleep_time_ms = 0);
};
} // namespace Metavision

#endif // METAVISION_SDK_UI_EVENT_LOOP_H
