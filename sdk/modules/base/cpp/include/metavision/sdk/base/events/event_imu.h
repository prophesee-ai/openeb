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

#ifndef METAVISION_SDK_BASE_EVENT_IMU_H
#define METAVISION_SDK_BASE_EVENT_IMU_H

#include <iostream>

#include "metavision/sdk/base/utils/detail/struct_pack.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/events/detail/event_traits.h"

namespace Metavision {

/// @brief Class representing an IMU event
/// @note This class is deprecated since version 2.1.0 and will be removed in next releases
class EventIMU {
public:
    /// @brief Accelerometer x, y, and z values [g]
    float ax, ay, az;

    /// @brief Gyroscope x, y, and z values [rad/s]
    float gx, gy, gz;

    /// @brief Timestamp at which the event happened (in us)
    timestamp t;

    /// @brief Default constructor
    EventIMU() = default;

    /// @brief Constructor
    /// @param ax Accelerometer x value (in g)
    /// @param ay Accelerometer y value (in g)
    /// @param az Accelerometer z value (in g)
    /// @param gx Gyroscope x value (in rad/s)
    /// @param gy Gyroscope y value (in rad/s)
    /// @param gz Gyroscope z value (in rad/s)
    /// @param ts Timestamp of the event (in us)
    inline EventIMU(float ax, float ay, float az, float gx, float gy, float gz, timestamp ts) :
        ax(ax), ay(ay), az(az), gx(gx), gy(gy), gz(gz), t(ts) {}

    /// @brief Writes EventIMU in buffer
    void write_event(void *buf, timestamp origin) const {
        RawEvent *buffer = (RawEvent *)buf;
        buffer->ts       = t - origin;
        buffer->ax       = ax;
        buffer->ay       = ay;
        buffer->az       = az;
        buffer->gx       = gx;
        buffer->gy       = gy;
        buffer->gz       = gz;
    }

    /// @brief Reads EventIMU (old format) from buffer
    static EventIMU read_event_v1(void *buf, const timestamp &delta_ts) {
        RawEventV1 *buffer = static_cast<RawEventV1 *>(buf);
        return EventIMU(buffer->ax, buffer->ay, buffer->az, buffer->gx, buffer->gy, buffer->gz, buffer->ts + delta_ts);
    }

    /// @brief Reads event 2D from buffer
    static EventIMU read_event(void *buf, const timestamp &delta_ts) {
        RawEvent *buffer = static_cast<RawEvent *>(buf);
        return EventIMU(buffer->ax, buffer->ay, buffer->az, buffer->gx, buffer->gy, buffer->gz, buffer->ts + delta_ts);
    }

    /// @brief Gets the size of the RawEvent
    static size_t get_raw_event_size() {
        return sizeof(RawEvent);
    }

    FORCE_PACK(
        /// Structure of size 64 bits to represent one event (old format)
        struct RawEventV1 {
            unsigned int ts : 32;
            unsigned int x : 9;
            unsigned int y : 8;
            unsigned int p : 1;
            unsigned int padding : 14;
            float ax;
            float ay;
            float az;
            float gx;
            float gy;
            float gz;
        });

    /// operator<<
    friend std::ostream &operator<<(std::ostream &output, const EventIMU &e) {
        output << "EventIMU: (";
        output << e.ax << ", " << e.ay << ", " << e.az << ", " << e.gx << ", " << e.gy << ", " << e.gz << ", " << e.t
               << ", ";
        output << ")";
        return output;
    }

    FORCE_PACK(
        /// Structure of size 64 bits to represent one event
        struct RawEvent {
            unsigned int ts : 32;
            unsigned int x : 14; // kept for retro-compatibility but empty field
            unsigned int y : 14; // kept for retro-compatibility but empty field
            unsigned int p : 4;  // kept for retro-compatibility but empty field
            float ax;
            float ay;
            float az;
            float gx;
            float gy;
            float gz;
        });
};

} // namespace Metavision

METAVISION_DEFINE_EVENT_TRAIT(Metavision::EventIMU, 15, "IMU")

#endif // METAVISION_SDK_BASE_EVENT_IMU_H
