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

#ifndef METAVISION_HAL_EVT4_EVENT_TYPES_H
#define METAVISION_HAL_EVT4_EVENT_TYPES_H

#include "metavision/hal/decoders/base/base_event_types.h"
#include "metavision/hal/decoders/base/event_base.h"

#include <cstdint>

namespace Metavision {

enum class EVT4EventTypes : EventTypesUnderlying_t {
    OTHER       = static_cast<EventTypesUnderlying_t>(0x6), // To be used for extensions in the event types
    CONTINUED   = static_cast<EventTypesUnderlying_t>(0x7), // Extra data to previous events
    EXT_TRIGGER = static_cast<EventTypesUnderlying_t>(0x9), // External trigger output
    CD_OFF      = static_cast<EventTypesUnderlying_t>(0xA), // CD OFF event, decrease in illumination (polarity '0')
    CD_ON       = static_cast<EventTypesUnderlying_t>(0xB), // CD ON event, increase in illumination (polarity '1')
    CD_VEC_OFF =
        static_cast<EventTypesUnderlying_t>(0xC), // CD Vector OFF event, decrease in illumination (polarity '0')
    CD_VEC_ON = static_cast<EventTypesUnderlying_t>(0xD), // CD Vector ON event, increase in illumination (polarity '1')
    EVT_TIME_HIGH = static_cast<EventTypesUnderlying_t>(0xE), // Timer high bits
    PADDING       = static_cast<EventTypesUnderlying_t>(0xF)  // Padding is all bits set: 0xFFFFFFFF
};

enum class EVT4EventSubTypes : std::uint16_t {
    MASTER_IN_CD_EVENT_COUNT           = static_cast<std::uint16_t>(0x0014),
    MASTER_RATE_CONTROL_CD_EVENT_COUNT = static_cast<std::uint16_t>(0x0016),
    UNUSED                             = static_cast<std::uint16_t>(0xFFFF),
};

struct EVT4Timestamp {
    timestamp time_low : 6;
    timestamp time_high : 28;
    timestamp n_loop : 30;
};

namespace Evt4Raw {

using RawEvent = EventBase::RawEvent;

struct EVT4EventCD {
    std::uint32_t y : 11;
    std::uint32_t x : 11;
    std::uint32_t timestamp : 6;
    std::uint32_t type : 4;
};
static_assert(sizeof(EVT4EventCD) == 4,
              "The size of the packed struct EVT4EventCD is not the expected one (which is 4 bytes)");

struct EVT4EventExtTrigger {
    std::uint32_t value : 1;
    std::uint32_t count : 7;
    std::uint32_t id : 5;
    std::uint32_t unused : 9;
    std::uint32_t timestamp : 6;
    std::uint32_t type : 4;
};
static_assert(sizeof(EVT4EventExtTrigger) == 4,
              "The size of the packed struct EVT4EventExtTrigger is not the expected one (which is 4 bytes)");

struct EVT4EventMonitor {
    std::uint32_t subtype : 16;
    std::uint32_t reserved : 6;
    std::uint32_t timestamp : 6;
    std::uint32_t type : 4;
};
static_assert(sizeof(EVT4EventMonitor) == 4,
              "The size of the packed struct EVT4EventMonitor is not the expected one (which is 4 bytes)");

struct EVT4EventMonitorMasterInCdEventCount {
    std::uint32_t count : 22;
    std::uint32_t unused : 6;
    std::uint32_t type : 4;
};
static_assert(
    sizeof(EVT4EventMonitorMasterInCdEventCount) == 4,
    "The size of the packed struct EVT4EventMonitorMasterInCdEventCount is not the expected one (which is 4 bytes)");

} // namespace Evt4Raw

} // namespace Metavision

#endif // METAVISION_HAL_EVT4_EVENT_TYPES_H
