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

#ifndef METAVISION_HAL_EVENT_BASE_H
#define METAVISION_HAL_EVENT_BASE_H

namespace Metavision {
class EventBase {
public:
    struct RawEvent {
        unsigned int trail : 28;
        unsigned int type : 4;
    };
    static_assert(sizeof(RawEvent) == 4,
                  "The size of the packed struct EventBase::RawEvent is not the expected one (which is 4 bytes)");
};
} // namespace Metavision

#endif // METAVISION_HAL_EVENT_BASE_H
