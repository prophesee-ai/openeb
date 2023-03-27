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

#ifndef METAVISION_HAL_EVT2_EVENT_TYPES_H
#define METAVISION_HAL_EVT2_EVENT_TYPES_H

#include "metavision/hal/decoders/base/base_event_types.h"

namespace Metavision {

constexpr uint8_t EVT2EventsTimeStampBits = 6;

enum class EVT2EventTypes : EventTypesUnderlying_t {
    CD_LOW = static_cast<EventTypesUnderlying_t>(
        BaseEventTypes::CD_LOW), // Left camera TD event, decrease in illumination (polarity '0')
    CD_HIGH = static_cast<EventTypesUnderlying_t>(
        BaseEventTypes::CD_HIGH), // Left camera TD event, increase in illumination (polarity '1')
    EVT_TIME_HIGH = static_cast<EventTypesUnderlying_t>(
        BaseEventTypes::EVT_TIME_HIGH), // Timer high bits, also used to synchronize different event flows in the FPGA.
    EXT_TRIGGER = static_cast<EventTypesUnderlying_t>(BaseEventTypes::EXT_TRIGGER), // External trigger output
    IMU_EVT     = static_cast<EventTypesUnderlying_t>(BaseEventTypes::IMU_EVT), // Inertial Measurement Unit event that
                                                                                // relays accelerometer and gyroscope
                                                                                // information.
    OTHER = static_cast<EventTypesUnderlying_t>(BaseEventTypes::OTHER), // To be used for extensions in the event types
    CONTINUED = static_cast<EventTypesUnderlying_t>(BaseEventTypes::CONTINUED) // Extra data to previous events

};

// Remark: event types
// - GRAY_LEVEL = 0x0B (Gray level event containing pixel location and intensity)
// - OPT_FLOW   = 0x0C (Optical flow event)
// do exist but are not included in the enum because they are not generated

struct EVT2Event2D {
    unsigned int y : 11;
    unsigned int x : 11;
    unsigned int timestamp : 6;
    unsigned int type : 4;
};
static_assert(sizeof(EVT2Event2D) == 4,
              "The size of the packed struct EVT2Event2D is not the expected one (which is 4 bytes)");

struct EVT2EventExtTrigger {
    unsigned int value : 1;
    unsigned int unused2 : 7;
    unsigned int id : 5;
    unsigned int unused1 : 9;
    unsigned int timestamp : 6;
    unsigned int type : 4;
};
static_assert(sizeof(EVT2EventExtTrigger) == 4,
              "The size of the packed struct EVT2EventExtTrigger is not the expected one (which is 4 bytes)");

// IMU event is composed of 6 32-bit blocks (1 IMU_EVT + 5 CONTINUED)

struct EVT2EventIMU {
    // 1st 32-bit block
    unsigned int dmp_1 : 1;    // Is DMP (Digital Motion Processor) active, bit 0
    int ax : 16;               // Accelerometer X value, bits 1 -> 16
    unsigned int unused_1 : 5; // Unused, bits 17 -> 21
    unsigned int ts_1 : 6;     // Least significant bits of the event time base, bits 22 -> 27
    unsigned int type_1 : 4;   // Event type : IMU_EVT ('1101'), bits 28 -> 31

    // 2nd 32-bit block
    unsigned int dmp_2 : 1;    // Is DMP (Digital Motion Processor) active, bit 0
    int ay : 16;               // Accelerometer Y value, bits 1 -> 16
    unsigned int unused_2 : 5; // Unused, bits 17 -> 21
    unsigned int ts_2 : 6;     // Least significant bits of the event time base, bits 22 -> 27
    unsigned int type_2 : 4;   // Event type : CONTINUED ('1111'), bits 28 -> 31

    // 3rd 32-bit block
    unsigned int dmp_3 : 1;    // Is DMP (Digital Motion Processor) active, bit 0
    int az : 16;               // Accelerometer Z value, bits 1 -> 16
    unsigned int unused_3 : 5; // Unused, bits 17 -> 21
    unsigned int ts_3 : 6;     // Least significant bits of the event time base, bits 22 -> 27
    unsigned int type_3 : 4;   // Event type : CONTINUED ('1111'), bits 28 -> 31

    // 4th 32-bit block
    unsigned int dmp_4 : 1;    // Is DMP (Digital Motion Processor) active, bit 0
    int gx : 16;               // Gyroscope X value, bits 1 -> 16
    unsigned int unused_4 : 5; // Unused, bits 17 -> 21
    unsigned int ts_4 : 6;     // Least significant bits of the event time base, bits 22 -> 27
    unsigned int type_4 : 4;   // Event type : CONTINUED ('1111'), bits 28 -> 31

    // 5th 32-bit block
    unsigned int dmp_5 : 1;    // Is DMP (Digital Motion Processor) active, bit 0
    int gy : 16;               // Gyroscope Y value, bits 1 -> 16
    unsigned int unused_5 : 5; // Unused, bits 17 -> 21
    unsigned int ts_5 : 6;     // Least significant bits of the event time base, bits 22 -> 27
    unsigned int type_5 : 4;   // Event type : CONTINUED ('1111'), bits 28 -> 31

    // 6th 32-bit block
    unsigned int dmp_6 : 1;    // Is DMP (Digital Motion Processor) active, bit 0
    int gz : 16;               // Gyroscope Z value, bits 1 -> 16
    unsigned int unused_6 : 5; // Unused, bits 17 -> 21
    unsigned int ts_6 : 6;     // Least significant bits of the event time base, bits 22 -> 27
    unsigned int type_6 : 4;   // Event type : CONTINUED ('1111'), bits 28 -> 31
};
static_assert(sizeof(EVT2EventIMU) == 24,
              "The size of the packed struct EVT2EventIMU is not the expected one (which is 24 bytes)");

struct EVT2EventMonitor {
    unsigned int subtype : 16;
    unsigned int monitorclass : 1;
    unsigned int padding : 5;
    unsigned int timestamp : 6;
    unsigned int type : 4;
};
static_assert(sizeof(EVT2EventMonitor) == 4,
              "The size of the packed struct EVT2EventMonitor is not the expected one (which is 4 bytes)");

struct EVT2EventMonitorTemperature {
    unsigned int temp_10_dot_12_float : 22;
    unsigned int over_temp_alarm : 1;
    unsigned int user_temp_alarm : 1;
    unsigned int source : 4;
    unsigned int type : 4;
};
static_assert(sizeof(EVT2EventMonitorTemperature) == 4,
              "The size of the packed struct EVT2EventMonitorTemperature is not the expected one (which is 4 bytes)");

struct EVT2EventMonitorIdle {
    unsigned int idle_time_us : 26;
    unsigned int unused : 2;
    unsigned int type : 4;
};
static_assert(sizeof(EVT2EventMonitorIdle) == 4,
              "The size of the packed struct EVT2EventMonitorIdle is not the expected one (which is 4 bytes)");

struct EVT2EventMonitorIllumination {
    unsigned int illumination_pulse_duration_us : 26;
    unsigned int unused : 2;
    unsigned int type : 4;
};
static_assert(sizeof(EVT2EventMonitorIllumination) == 4,
              "The size of the packed struct EVT2EventMonitorIllumination is not the expected one (which is 4 bytes)");

struct EVT2EventMonitorEndOfFrame {
    uint32_t frame_size_byte : 28;
    unsigned int type : 4;
};
static_assert(sizeof(EVT2EventMonitorEndOfFrame) == 4,
              "The size of the packed struct EVT2EventMonitorEndOfFrame is not the expected one (which is 4 bytes)");

} // namespace Metavision

#endif // METAVISION_HAL_EVT2_EVENT_TYPES_H
