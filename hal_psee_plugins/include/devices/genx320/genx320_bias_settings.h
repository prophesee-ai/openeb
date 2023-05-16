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

#ifndef METAVISION_HAL_GENX320_BIAS_SETTINGS_H
#define METAVISION_HAL_GENX320_BIAS_SETTINGS_H

#include <vector>
#include <metavision/psee_hw_layer/devices/common/bias_settings.h>

// Absolute recommended
static constexpr int BIAS_REC_FO_MIN       = 19;
static constexpr int BIAS_REC_FO_MAX       = 39;
static constexpr int BIAS_REC_HPF_MIN      = 0;
static constexpr int BIAS_REC_HPF_MAX      = 127;
static constexpr int BIAS_REC_DIFF_ON_MIN  = 24;
static constexpr int BIAS_REC_DIFF_ON_MAX  = 60;
static constexpr int BIAS_REC_DIFF_OFF_MIN = 19;
static constexpr int BIAS_REC_DIFF_OFF_MAX = 50;
static constexpr int BIAS_REC_DIFF_MIN     = 41;
static constexpr int BIAS_REC_DIFF_MAX     = 51;
static constexpr int BIAS_REC_REFR_MIN     = 0;
static constexpr int BIAS_REC_REFR_MAX     = 127;

// Absolute allowed
static constexpr int BIAS_RANGE_MIN = 0;
static constexpr int BIAS_RANGE_MAX = 127;

static std::vector<Metavision::bias_settings> genx320_biases_settings = {
    {"bias_fo", BIAS_RANGE_MIN, BIAS_RANGE_MAX, BIAS_REC_FO_MIN, BIAS_REC_FO_MAX, true},
    {"bias_hpf", BIAS_RANGE_MIN, BIAS_RANGE_MAX, BIAS_REC_HPF_MIN, BIAS_REC_HPF_MAX, true},
    {"bias_diff_on", BIAS_RANGE_MIN, BIAS_RANGE_MAX, BIAS_REC_DIFF_ON_MIN, BIAS_REC_DIFF_ON_MAX, true},
    {"bias_diff_off", BIAS_RANGE_MIN, BIAS_RANGE_MAX, BIAS_REC_DIFF_OFF_MIN, BIAS_REC_DIFF_OFF_MAX, true},
    {"bias_diff", BIAS_RANGE_MIN, BIAS_RANGE_MAX, BIAS_REC_DIFF_MIN, BIAS_REC_DIFF_MAX, true},
    {"bias_refr", BIAS_RANGE_MIN, BIAS_RANGE_MAX, BIAS_REC_REFR_MIN, BIAS_REC_REFR_MAX, true},
};

#endif // METAVISION_HAL_GENX320_BIAS_SETTINGS_H