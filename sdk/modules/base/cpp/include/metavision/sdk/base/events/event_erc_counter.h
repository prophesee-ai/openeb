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

#ifndef METAVISION_SDK_BASE_EVENT_ERC_COUNTER_H
#define METAVISION_SDK_BASE_EVENT_ERC_COUNTER_H

#include <ostream>
#include "metavision/sdk/base/utils/detail/struct_pack.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/events/detail/event_traits.h"

namespace Metavision {

/// @brief Class representing event notification from Event Rate Controller counting events
class EventERCCounter {
public:
    /// @brief Timestamp at which the event happened (in us)
    timestamp t;
    /// @brief number of events counted
    uint64_t event_count;
    /// @brief states whether the count number represents events that were output by the ERC
    bool is_output;

    /// @brief Default constructor
    EventERCCounter() = default;

    /// @brief Constructor
    /// @param t Timestamp of the event (in us)
    /// @param count number of CD events received
    /// @param erc_output_counter whether count represents number of events output by ERC
    inline EventERCCounter(timestamp t, uint64_t count, bool erc_output_counter) :
        t(t), event_count(count), is_output(erc_output_counter) {}

    bool operator==(const EventERCCounter &e) const {
        return e.t == t && e.event_count == event_count && e.is_output == is_output;
    }

    friend std::ostream &operator<<(std::ostream &out, const EventERCCounter &e) {
        out << "EventERCCounter: (" << e.t << ": " << e.event_count << ")";
        return out;
    }

    FORCE_PACK(
        /// Structure of size 64 bits to represent one event
        struct RawEvent {
            uint32_t ts;
            uint32_t count : 31;
            uint32_t output_counter : 1;
        });
};
} // namespace Metavision

METAVISION_DEFINE_EVENT_TRAIT(Metavision::EventERCCounter, 17, "ERC counter event class")

#endif // METAVISION_SDK_BASE_EVENT_ERC_COUNTER_H
