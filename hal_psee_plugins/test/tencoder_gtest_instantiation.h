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

#ifndef METAVISION_HAL_TENCODER_GTEST_INSTANTIATION_H
#define METAVISION_HAL_TENCODER_GTEST_INSTANTIATION_H

#include <gtest/gtest.h>

#include "metavision/sdk/base/utils/timestamp.h"
#include "evt2_raw_format.h"
#include "encoding_policies.h"

template<typename Format, typename TimerHighRedundancyPolicyType, Metavision::timestamp T_STEP>
struct GtestsParameters;

typedef ::testing::Types<GtestsParameters<Metavision::Evt2RawFormat, Metavision::TimerHighRedundancyNone, 100>,
                         GtestsParameters<Metavision::Evt2RawFormat, Metavision::TimerHighRedundancyEvt2Default, 100>,
                         GtestsParameters<Metavision::Evt2RawFormat, Metavision::TimerHighRedundancyNone, 100>,
                         GtestsParameters<Metavision::Evt2RawFormat, Metavision::TimerHighRedundancyEvt2Default, 100>,
                         GtestsParameters<Metavision::Evt2RawFormat, Metavision::TimerHighRedundancyNone, 1000>,
                         GtestsParameters<Metavision::Evt2RawFormat, Metavision::TimerHighRedundancyEvt2Default, 1000>,
                         GtestsParameters<Metavision::Evt2RawFormat, Metavision::TimerHighRedundancyNone, 1000>,
                         GtestsParameters<Metavision::Evt2RawFormat, Metavision::TimerHighRedundancyEvt2Default, 1000>,
                         GtestsParameters<Metavision::Evt2RawFormat, Metavision::TimerHighRedundancyNone, 5000>,
                         GtestsParameters<Metavision::Evt2RawFormat, Metavision::TimerHighRedundancyEvt2Default, 5000>,
                         GtestsParameters<Metavision::Evt2RawFormat, Metavision::TimerHighRedundancyNone, 5000>,
                         GtestsParameters<Metavision::Evt2RawFormat, Metavision::TimerHighRedundancyEvt2Default, 5000>>
    TestingTypes;

#endif // METAVISION_HAL_TENCODER_GTEST_INSTANTIATION_H
