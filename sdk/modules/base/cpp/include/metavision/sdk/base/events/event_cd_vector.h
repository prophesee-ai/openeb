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

#ifndef METAVISION_SDK_BASE_EVENT_CD_VECTOR_H
#define METAVISION_SDK_BASE_EVENT_CD_VECTOR_H

#include <cstdint>
#include <iostream>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Class representing Vectorized 2D CD (Contrast Detection) events:
/// @details vector_mask represents 32 potentially triggered events in a single lane as a 32bit value.
/// Each set bit represents a triggered event at pos(base_x + vector_mask[i], y)
class EventCDVector {
public:

    /// @brief Default constructor
    EventCDVector() = default;

    /// @brief Constructor from Event2d
    inline EventCDVector(
        uint16_t base_x, uint16_t y,
        bool polarity,
        uint32_t vector_mask,
        timestamp event_timestamp
    ) :
        base_x(base_x), y(y),
        polarity(polarity),
        vector_mask(vector_mask),
        event_timestamp(event_timestamp)
    {}

    inline bool operator==(const EventCDVector &rhs) const {
        return (
            (polarity == rhs.polarity) && 
            (base_x == rhs.base_x) && (y == rhs.y) &&
            (vector_mask == rhs.vector_mask) && 
            (event_timestamp == rhs.event_timestamp)
        );
    }

    /// @brief function operator<< that returns std::ostream &
    friend std::ostream &operator<<(std::ostream &output, const EventCDVector &rhs) {
        output << "EventCDVector: (";

        output 
                << rhs.base_x << ", "
                << rhs.y << ", "
                << (int)rhs.polarity << ", " 
                << rhs.vector_mask << ", " 
                << rhs.event_timestamp;

        output << ")";
        return output;
    }
   
    uint16_t base_x, y;
    bool polarity;
    uint32_t vector_mask;
    timestamp event_timestamp;

};

} // namespace Metavision

// METAVISION_DEFINE_EVENT_TRAIT(Metavision::EventCDVector, 13, "CDVector")

#endif // METAVISION_SDK_BASE_EVENT_CD_VECTOR_H