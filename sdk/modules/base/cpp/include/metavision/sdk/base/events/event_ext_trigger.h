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

#ifndef METAVISION_SDK_BASE_EVENT_EXT_TRIGGER_H
#define METAVISION_SDK_BASE_EVENT_EXT_TRIGGER_H

#include <cstddef>
#include <cstdint>
#include <iostream>

#include "metavision/sdk/base/utils/detail/struct_pack.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/events/detail/event_traits.h"

namespace Metavision {

/// @brief Class representing an external trigger event.
class EventExtTrigger {
public:
    /// @brief Default constructor
    EventExtTrigger() = default;

    /// @param p Polarity of the external trigger event
    /// @param t Timestamp of the external trigger event (in us)
    /// @param id Channel ID of the external trigger event
    inline EventExtTrigger(short p, timestamp t, short id) : p(p), t(t), id(id) {}

    /// Writes EventExtTrigger in buffer
    void write_event(void *buf, timestamp origin) const {
        RawEvent *buffer = static_cast<RawEvent *>(buf);
        buffer->ts       = static_cast<uint32_t>(t - origin);
        buffer->p        = p;
        buffer->id       = id;
        buffer->pad1     = 0;
    }

    /// Reads EventExtTrigger encoded in an old format from buffer
    static EventExtTrigger read_event_v1(void *buf, const timestamp &delta_ts) {
        return EventExtTrigger::read_event(buf, delta_ts);
    }

    /// Reads EventExtTrigger from buffer
    static EventExtTrigger read_event(void *buf, const timestamp &delta_ts = 0) {
        RawEvent *buffer = static_cast<RawEvent *>(buf);
        return EventExtTrigger(buffer->p, buffer->ts + delta_ts, buffer->id);
    }

    /// Returns raw event size
    static size_t get_raw_event_size() {
        return sizeof(RawEvent);
    }

    /// Function shifted returning class EventExtTrigger
    inline EventExtTrigger shifted(timestamp dt) {
        return EventExtTrigger(p, t + dt, id);
    }

    /// Event comparison operator.
    inline bool operator==(const EventExtTrigger &e) const {
        return t == e.t && p == e.p && id == e.id;
    }

    /// Event timestamp comparison operator.
    inline bool operator<(const EventExtTrigger &e) const {
        return t < e.t;
    }

    /// Event timestamp comparison operator.
    inline bool operator<=(const EventExtTrigger &e) const {
        return t <= e.t;
    }

    /// Event timestamp comparison operator.
    inline bool operator>(const EventExtTrigger &e) const {
        return t > e.t;
    }

    /// Event timestamp comparison operator.
    inline bool operator>=(const EventExtTrigger &e) const {
        return t >= e.t;
    }

    // Function operator<< returning int &
    friend std::ostream &operator<<(std::ostream &output, const EventExtTrigger &e) {
        output << "EventExtTrigger: (";
        output << (int)e.p << ", " << e.t << ", " << e.id;
        output << ")";
        return output;
    }

    FORCE_PACK(
        /// The raw event format of an external trigger event
        struct RawEvent {
            uint32_t ts;
            unsigned int p : 4;
            unsigned int pad1 : 22;
            unsigned int id : 6;
        });

    typedef RawEvent RawEventV1;

    /// Polarity representing the change of contrast (1: positive, 0: negative)
    short p;

    /// Timestamp at which the event happened (in us)
    timestamp t;

    /// ID of the external trigger
    short id;
};
} // namespace Metavision

METAVISION_DEFINE_EVENT_TRAIT(Metavision::EventExtTrigger, 14, "Trigger")

#endif // METAVISION_SDK_BASE_EVENT_EXT_TRIGGER_H
