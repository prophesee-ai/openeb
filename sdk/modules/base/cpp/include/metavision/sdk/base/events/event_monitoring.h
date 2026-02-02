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

#ifndef METAVISION_SDK_BASE_EVENT_MONITORING_H
#define METAVISION_SDK_BASE_EVENT_MONITORING_H

#include <cstdint>
#include <ios>
#include <ostream>
#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Class representing monitoring event notifications
class EventMonitoring {
public:
    /// @brief Timestamp at which the event happened (in us)
    timestamp t;
    /// @brief ID of the monitoring event type
    uint32_t type_id;
    /// @brief Data associated to the event
    uint32_t payload;

    /// @brief Default constructor
    EventMonitoring() = default;

    /// @brief Constructor
    /// @param t Timestamp of the event (in us)
    /// @param type_id ID of the monitoring event type
    /// @param payload Data associated to the event
    inline EventMonitoring(timestamp t, uint32_t type_id, uint32_t payload) :
        t(t), type_id(type_id), payload(payload) {}

    bool operator==(const EventMonitoring &e) const {
        return e.t == t && e.type_id == type_id && e.payload == payload;
    }

    friend std::ostream &operator<<(std::ostream &out, const EventMonitoring &e) {
        out << "EventMonitoring: (" << e.t << ", 0x" << std::hex << e.type_id << ", 0x" << e.payload << std::dec << ")";
        return out;
    }
};
} // namespace Metavision

#endif // METAVISION_SDK_BASE_EVENT_MONITORING_H
