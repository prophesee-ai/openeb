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

#ifndef METAVISION_HAL_EVT21_EVENT_TYPES_H
#define METAVISION_HAL_EVT21_EVENT_TYPES_H

#include <stdint.h>

#include "metavision/hal/decoders/base/base_event_types.h"

namespace Metavision {

namespace Evt21Raw {

struct Evt21_Event_Type_4bits {
    uint64_t unused2 : 32;
    uint64_t unused1 : 28;
    uint64_t type : 4;
};

struct RawEvent {
    uint64_t content2 : 32;
    uint64_t content1 : 28;
    uint64_t type : 4;
};

struct Event_2D {
    uint64_t valid : 32;
    uint64_t y : 11;
    uint64_t x : 11;
    uint64_t ts : 6;
    uint64_t type : 4;
};

struct Event_TIME_HIGH {
    uint64_t unused : 32;
    uint64_t ts : 28;
    uint64_t type : 4;
};

struct Event_EXT_TRIGGER {
    uint64_t unused3 : 32;
    uint64_t p : 1;
    uint64_t unused2 : 7;
    uint64_t id : 5;
    uint64_t unused1 : 9;
    uint64_t ts : 6;
    uint64_t type : 4;
};

struct Event_OTHERS {
    uint64_t payload : 32;
    uint64_t subtype : 16;
    uint64_t cls : 1;
    uint64_t unused : 5;
    uint64_t ts : 6;
    uint64_t type : 4;
};

static_assert(sizeof(Evt21_Event_Type_4bits) == 8,
              "The size of the packed struct Evt21_Event_Type_4bits is not the expected one (which is 8 bytes)");

} // namespace Evt21Raw

namespace Evt21LegacyRaw {

struct Evt21Legacy_Event_Type_4bits {
    uint64_t unused1 : 28;
    uint64_t type : 4;
    uint64_t unused2 : 32;
};

struct RawEvent {
    uint64_t content1 : 28;
    uint64_t type : 4;
    uint64_t content2 : 32;
};

struct Event_2D {
    uint64_t y : 11;
    uint64_t x : 11;
    uint64_t ts : 6;
    uint64_t type : 4;
    uint64_t valid : 32;
};

struct Event_TIME_HIGH {
    uint64_t ts : 28;
    uint64_t type : 4;
    uint64_t unused : 32;
};

struct Event_EXT_TRIGGER {
    uint64_t p : 1;
    uint64_t unused2 : 7;
    uint64_t id : 5;
    uint64_t unused1 : 9;
    uint64_t ts : 6;
    uint64_t type : 4;
    uint64_t unused3 : 32;
};

struct Event_OTHERS {
    uint64_t subtype : 16;
    uint64_t cls : 1;
    uint64_t unused : 5;
    uint64_t ts : 6;
    uint64_t type : 4;
    uint64_t payload : 32;
};

static_assert(sizeof(Evt21Legacy_Event_Type_4bits) == 8,
              "The size of the packed struct Evt21Legacy_Event_Type_4bits is not the expected one (which is 8 bytes)");

} // namespace Evt21LegacyRaw

enum class Evt21EventTypes_4bits : EventTypesUnderlying_t {
    EVT_NEG       = 0x0,
    EVT_POS       = 0x1,
    EVT_TIME_HIGH = 0x8,
    EXT_TRIGGER   = 0xA,
    OTHERS        = 0xE,
    CONTINUED     = 0xF
};

enum class Evt21EventMasterEventTypes : uint16_t {
    MASTER_IN_CD_EVENT_COUNT           = 0x0014,
    MASTER_RATE_CONTROL_CD_EVENT_COUNT = 0x0016,
};

} // namespace Metavision

#endif // METAVISION_HAL_EVT21_EVENT_TYPES_H
