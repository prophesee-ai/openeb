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

#ifndef METAVISION_HAL_SAMPLE_ANTIFLICKER_H
#define METAVISION_HAL_SAMPLE_ANTIFLICKER_H

#include <metavision/hal/facilities/i_antiflicker_module.h>


/// @brief Anti-flicker module
///
/// This class is the implementation of HAL's class @ref Metavision::I_AntiFlickerModule
class SampleAntiFlicker : public Metavision::I_AntiFlickerModule {
public:
    bool enable(bool b) override final;
    bool is_enabled() const override final;
    bool set_frequency_band(uint32_t low_freq, uint32_t high_freq) override final;
    uint32_t get_band_low_frequency() const override final;
    uint32_t get_band_high_frequency() const override final;
    std::pair<uint32_t, uint32_t> get_frequency_band() const;
    uint32_t get_min_supported_frequency() const override final;
    uint32_t get_max_supported_frequency() const override final;
    bool set_filtering_mode(AntiFlickerMode mode) override final;
    AntiFlickerMode get_filtering_mode() const override final;
    bool set_duty_cycle(float duty_cycle) override final;
    float get_duty_cycle() const override final;
    float get_min_supported_duty_cycle() const override final;
    float get_max_supported_duty_cycle() const override final;
    bool set_start_threshold(uint32_t threshold) override final;
    bool set_stop_threshold(uint32_t threshold) override final;
    uint32_t get_start_threshold() const override final;
    uint32_t get_stop_threshold() const override final;
    uint32_t get_min_supported_start_threshold() const override final;
    uint32_t get_max_supported_start_threshold() const override final;
    uint32_t get_min_supported_stop_threshold() const override final;
    uint32_t get_max_supported_stop_threshold() const override final;
};

#endif // METAVISION_HAL_SAMPLE_ANTIFLICKER_H
