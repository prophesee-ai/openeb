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

#ifndef METAVISION_HAL_SAMPLE_EVENTS_FORMAT_H
#define METAVISION_HAL_SAMPLE_EVENTS_FORMAT_H

#include <cstdint>
#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/base/events/event_cd.h>

/// This sample encoding format is the following :
/// timestamp : 44 bits
///        x  : 10 bits
///        y  : 9 bits
/// polarity  : 1 bit
///
/// for a total of 64 bits
using SampleEventsFormat = std::uint64_t;

constexpr int TS_BITS = 44;
constexpr int X_BITS  = 10;
constexpr int Y_BITS  = 9;
constexpr int P_BITS  = 1;

// Define the masks used to encode and decode
constexpr uint64_t TS_MASK = 0xFFFFFFFFFFF;
constexpr uint64_t X_MASK  = 0x3FF00000000000;
constexpr uint64_t Y_MASK  = 0x7FC0000000000000;
constexpr uint64_t P_MASK  = 0x8000000000000000;

constexpr uint64_t X_MASK_SHIFTED = (0x3FF); // = X_MASK >> TS_BITS
constexpr uint64_t Y_MASK_SHIFTED = (0x1FF); // = Y_MASK >> (TS_BITS + Y_BITS)
constexpr uint64_t P_MASK_SHIFTED = (0x1);   // P_MASK >> (TS_BITS + Y_BITS + X_MASK)

inline void encode_sample_format(SampleEventsFormat &encoded_ev, unsigned short x, unsigned short y, short p,
                                 Metavision::timestamp t) {
    encoded_ev = (t & TS_MASK) | ((x & X_MASK_SHIFTED) << TS_BITS) | ((y & Y_MASK_SHIFTED) << (TS_BITS + X_BITS)) |
                 ((p & P_MASK_SHIFTED) << (TS_BITS + X_BITS + Y_BITS));
}

inline void decode_sample_format(SampleEventsFormat in, Metavision::EventCD &ev, Metavision::timestamp t_shift = 0) {
    ev.t = (in & TS_MASK) - t_shift;
    ev.x = static_cast<unsigned short>((in & X_MASK) >> TS_BITS);
    ev.y = static_cast<unsigned short>((in & Y_MASK) >> (TS_BITS + X_BITS));
    ev.p = static_cast<short>((in & P_MASK) >> (TS_BITS + X_BITS + Y_BITS));
}

#endif // METAVISION_HAL_SAMPLE_EVENTS_FORMAT_H
