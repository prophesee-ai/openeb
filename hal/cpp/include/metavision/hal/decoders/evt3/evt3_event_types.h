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

#ifndef METAVISION_HAL_EVT3_EVENT_TYPES_H
#define METAVISION_HAL_EVT3_EVENT_TYPES_H

#include <string>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/hal/decoders/base/base_event_types.h"

namespace Metavision {

namespace Evt3Raw {

union Mask {
    uint32_t valid;
    struct Mask_Vect32 {
        uint32_t valid1 : 12;
        uint32_t valid2 : 12;
        uint32_t valid3 : 8;
    } m;
};

struct Evt3_Event_Type_4bits {
    uint16_t pad : 11;
    uint16_t p : 1;
    uint16_t type : 4;
};

struct Event_PosX {
    uint16_t x : 11;
    uint16_t pol : 1;
    uint16_t type : 4;
};
struct Event_Vect12_12_8 {
    uint16_t valid1 : 12;
    uint16_t type1 : 4;
    uint16_t valid2 : 12;
    uint16_t type2 : 4;
    uint16_t valid3 : 8;
    uint16_t unused3 : 4;
    uint16_t type3 : 4;
};

struct Event_Continue12_12_4 {
    uint16_t valid1 : 12;
    uint16_t type1 : 4;
    uint16_t valid2 : 12;
    uint16_t type2 : 4;
    uint16_t valid3 : 4;
    uint16_t unused3 : 8;
    uint16_t type3 : 4;
    static uint64_t decode(const Event_Continue12_12_4 &e) {
        return static_cast<uint64_t>(e.valid1) | static_cast<uint64_t>(e.valid2) << 12 |
               static_cast<uint64_t>(e.valid3) << 24;
    }
};

struct Event_XBase {
    uint16_t x : 11;
    uint16_t pol : 1;
    uint16_t type : 4;
};

struct Event_Y {
    uint16_t y : 11;
    uint16_t orig : 1;
    uint16_t type : 4;
};

struct Event_ExtTrigger {
    uint16_t pol : 1;
    uint16_t unused : 7;
    uint16_t id : 4;
    uint16_t type : 4;
};
struct RawEvent {
    uint16_t content : 12;
    uint16_t type : 4;
};

struct Event_Time {
    uint16_t time : 12;
    uint16_t type : 4;
    static size_t decode_time_high(const uint16_t *ev, Metavision::timestamp &cur_t) {
        const Event_Time *ev_timehigh = reinterpret_cast<const Event_Time *>(ev);
        cur_t                         = (cur_t & ~(0b111111111111ull << 12)) | (ev_timehigh->time << 12);
        return 1;
    }
};
static_assert(sizeof(Evt3_Event_Type_4bits) == 2,
              "The size of the packed struct Evt3_Event_Type_4bits is not the expected one (which is 2 bytes)");

} // namespace Evt3Raw

enum class Evt3EventTypes_4bits : EventTypesUnderlying_t {
    EVT_ADDR_Y    = 0x0,
    EVT_ADDR_X    = 0x2,
    VECT_BASE_X   = 0x3,
    VECT_12       = 0x4,
    VECT_8        = 0x5,
    EVT_TIME_LOW  = 0x6,
    CONTINUED_4   = 0x7,
    EVT_TIME_HIGH = 0x8,
    EXT_TRIGGER   = 0xA,
    UNUSED_1      = 0xB,
    UNUSED_2      = 0xC,
    IMU           = 0xD,
    OTHERS        = 0xE,
    CONTINUED_12  = 0xF
};

enum class Evt3MasterEventTypes : uint16_t {
    MASTER_IN_CD_EVENT_COUNT           = 0x014,
    MASTER_RATE_CONTROL_CD_EVENT_COUNT = 0x016
};

} // namespace Metavision

#endif // METAVISION_HAL_EVT3_EVENT_TYPES_H
