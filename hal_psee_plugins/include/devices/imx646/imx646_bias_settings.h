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

#ifndef METAVISION_HAL_IMX646_BIAS_SETTINGS_H
#define METAVISION_HAL_IMX646_BIAS_SETTINGS_H

// Bias offsets are relative to factory programmed values.
// The usable range will be (default - min offset) : (default + max offset)

// Different from IMX636
static constexpr int BIAS_FO_MIN_OFFSET = -20;
static constexpr int BIAS_FO_MAX_OFFSET = 0;

static constexpr int BIAS_DIFF_ON_MIN_OFFSET = -80;
static constexpr int BIAS_DIFF_ON_MAX_OFFSET = 145;

static constexpr int BIAS_DIFF_OFF_MIN_OFFSET = -30;
static constexpr int BIAS_DIFF_OFF_MAX_OFFSET = 200;

// Identical to IMX636
static constexpr int BIAS_DIFF_MIN_OFFSET = -25;
static constexpr int BIAS_DIFF_MAX_OFFSET = 23;

static constexpr int BIAS_HPF_MIN_OFFSET = 0;
static constexpr int BIAS_HPF_MAX_OFFSET = 120;

static constexpr int BIAS_REFR_MIN_OFFSET = -20;
static constexpr int BIAS_REFR_MAX_OFFSET = 235;

#endif // METAVISION_HAL_IMX646_BIAS_SETTINGS_H
