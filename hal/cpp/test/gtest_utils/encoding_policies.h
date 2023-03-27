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

#ifndef METAVISION_HAL_ENCODING_POLICIES_H
#define METAVISION_HAL_ENCODING_POLICIES_H

#include "event_raw_format_traits.h"

namespace Metavision {

//////////////////////////////////////////
template<unsigned int FACTOR>
class TimerHighRedundancyPolicyT {
public:
    static constexpr unsigned int REDUNDANCY_FACTOR = FACTOR;
};

template<unsigned int FACTOR>
constexpr unsigned int TimerHighRedundancyPolicyT<FACTOR>::REDUNDANCY_FACTOR;

using TimerHighRedundancyNone        = TimerHighRedundancyPolicyT<1>;
using TimerHighRedundancyEvt2Default = TimerHighRedundancyPolicyT<4>; // Default for Evt2

} // namespace Metavision

#endif // METAVISION_HAL_ENCODING_POLICIES_H
