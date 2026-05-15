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

#ifndef METAVISION_SDK_BASE_EVENT2D_H
#define METAVISION_SDK_BASE_EVENT2D_H

#include <cstdint>
#include <vector>
#include <iterator>
#include <type_traits>
#include <iostream>

#include "metavision/sdk/base/utils/detail/struct_pack.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/events/detail/event_traits.h"

namespace Metavision {

/// @brief Class representing basic 2D events:
///     - Contrast Detection -CD- event
///     - Exposure Measurement -EM- event
class Event2d {
public:
    /// @brief Column position in the sensor at which the event happened
    unsigned short x;

    /// @brief Row position in the sensor at which the event happened
    unsigned short y;

    /// @brief Polarity, whose value depends on the type of the event (CD or EM)
    ///
    /// - In case of CD event: polarity representing the change of contrast
    ///     - 1: a positive contrast change
    ///     - 0: a negative contrast change
    /// - In case of EM event: polarity representing the exposure measurement type
    ///     - 1: EM high i.e. the exposure measurement begins
    ///     - 0: EM low i.e. the exposure measurement ends
    short p;

    /// @brief Timestamp at which the event happened (in us)
    timestamp t;

    /// @brief Default constructor
    Event2d() = default;

    /// @brief Constructor
    /// @param x Column position of the event in the sensor
    /// @param y Row position of the event in the sensor
    /// @param p Polarity specialising the event
    /// @param t Timestamp of the event (in us)
    inline Event2d(unsigned short x, unsigned short y, short p, timestamp t) : x(x), y(y), p(p), t(t) {}

    /// @cond DO_NOT_SHOW_IN_DOC
    /// @brief function shifted that returns class Event2d
    inline Event2d shifted(timestamp dt) {
        return Event2d(x, y, p, t + dt);
    }

    /// @brief function operator< that returns bool
    inline bool operator<(const Event2d &e) const {
        return t < e.t;
    }

    /// @brief function operator<= that returns bool
    inline bool operator<=(const Event2d &e) const {
        return t <= e.t;
    }

    /// @brief function operator> that returns bool
    inline bool operator>(const Event2d &e) const {
        return t > e.t;
    }

    /// @brief function operator>= that returns bool
    inline bool operator>=(const Event2d &e) const {
        return t >= e.t;
    }

    /// @brief function operator== that returns bool
    inline bool operator==(const Event2d &e) const {
        return t == e.t && p == e.p && x == e.x && y == e.y;
    }

    /// @brief function operator<< that returns std::ostream &
    friend std::ostream &operator<<(std::ostream &output, const Event2d &e) {
        output << "Event2d: (";
        output << (int)e.x << ", " << (int)e.y << ", ";
        output << (int)e.p << ", " << e.t;
        output << ")";
        return output;
    }

    /// @brief Reads Event2d (old format) from buffer
    static Event2d read_event_v1(void *buf, const timestamp &delta_ts = 0) {
        RawEventV1 *buffer = static_cast<RawEventV1 *>(buf);
        return Event2d(buffer->x, buffer->y, buffer->p, buffer->ts + delta_ts);
    }

    //// @brief Reads event 2D from buffer
    static Event2d read_event(void *buf, const timestamp &delta_ts = 0) {
        RawEvent *buffer = static_cast<RawEvent *>(buf);
        return Event2d(buffer->x, buffer->y, buffer->p, buffer->ts + delta_ts);
    }

    /// @brief Writes Event2d in buffer
    void write_event(void *buf, timestamp origin) const {
        RawEvent *buffer = static_cast<RawEvent *>(buf);
        buffer->ts       = static_cast<uint32_t>(t - origin);
        buffer->x        = x;
        buffer->y        = y;
        buffer->p        = p;
    }

    FORCE_PACK(
        /// Structure of size 64 bits to represent one event (old format)
        struct RawEventV1 {
            uint32_t ts;
            unsigned int x : 9;
            unsigned int y : 8;
            unsigned int p : 1;
            unsigned int padding : 14;
        });

    FORCE_PACK(
        /// Structure of size 64 bits to represent one event
        struct RawEvent {
            uint32_t ts;
            unsigned int x : 14;
            unsigned int y : 14;
            unsigned int p : 4;
        });

    /// @endcond
};

} // namespace Metavision

METAVISION_DEFINE_EVENT_TRAIT(Metavision::Event2d, 0, "Base event class")

/// @cond DO_NOT_SHOW_IN_DOC
namespace std {
// when calling std::copy with a back_insert_iterator<vector>, some implementations of the STL
// do the right thing and others do not.
// this overload of std::copy is defined to make sure that the most efficient implementation
// is always used
template<typename InputIterator, typename EventType>
typename enable_if<is_base_of<Metavision::Event2d, EventType>::value,
                   std::back_insert_iterator<std::vector<EventType>>>::type
    copy(InputIterator begin, InputIterator end, std::back_insert_iterator<std::vector<EventType>> d_begin,
         std::forward_iterator_tag * =
             static_cast<typename std::iterator_traits<InputIterator>::iterator_category *>(0)) {
    struct container_exposer : public std::back_insert_iterator<std::vector<EventType>> {
        using std::back_insert_iterator<std::vector<EventType>>::container;
    };
    std::vector<EventType> *c = static_cast<container_exposer &>(d_begin).container;
    c->insert(c->end(), begin, end);
    return std::back_inserter(*c);
}
} // namespace std
/// @endcond

#endif // METAVISION_SDK_BASE_EVENT2D_H
