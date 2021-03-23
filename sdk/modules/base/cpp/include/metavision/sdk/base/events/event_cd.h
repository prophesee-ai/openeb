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

#ifndef METAVISION_SDK_BASE_EVENT_CD_H
#define METAVISION_SDK_BASE_EVENT_CD_H

// Metavision SDK Base CD event
#include "metavision/sdk/base/events/event2d.h"

namespace Metavision {

/// @brief Class representing basic 2D CD (Contrast Detection) events:
class EventCD : public Event2d {
public:
    /// @brief Default constructor
    EventCD() = default;

    /// @brief Constructor from Event2d
    inline EventCD(const Event2d &ev) : Event2d(ev) {}
    using Event2d::Event2d;
};

} // namespace Metavision

METAVISION_DEFINE_EVENT_TRAIT(Metavision::EventCD, 12, "CD")

#endif // METAVISION_SDK_BASE_EVENT_CD_H
