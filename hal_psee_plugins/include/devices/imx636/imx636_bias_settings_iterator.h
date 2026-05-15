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

/* This file is meant to make imx636_bias_settings.h content iterable for bias construction */
/* It requires to include imx636_ll_biases.h first, and isn't protected against multiple inclusions on purpose */

static std::vector<Metavision::imx636_bias_setting> bias_settings = {
    {"bias_fo", -150, 200, BIAS_FO_MIN_OFFSET, BIAS_FO_MAX_OFFSET, true},
    {"bias_hpf", 0, 255, BIAS_HPF_MIN_OFFSET, BIAS_HPF_MAX_OFFSET, true},
    {"bias_diff_on", -150, 200, BIAS_DIFF_ON_MIN_OFFSET, BIAS_DIFF_ON_MAX_OFFSET, true},
    {"bias_diff", -150, 200, BIAS_DIFF_MIN_OFFSET, BIAS_DIFF_MAX_OFFSET, true},
    {"bias_diff_off", -150, 200, BIAS_DIFF_OFF_MIN_OFFSET, BIAS_DIFF_OFF_MAX_OFFSET, true},
    {"bias_refr", -50, 255, BIAS_REFR_MIN_OFFSET, BIAS_REFR_MAX_OFFSET, true},
};
