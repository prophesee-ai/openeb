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

#include "sample_antiflicker.h"

// Here we propose placeholder to insert an implementation
// Antiflicker implementation available in OpenEB is valid for IMX636 and GenX320 and can be used as a model:
// https://github.com/prophesee-ai/openeb/blob/main/hal_psee_plugins/src/devices/common/antiflicker_filter.cpp

bool SampleAntiFlicker::enable(bool b) {
    return false;
}

bool SampleAntiFlicker::is_enabled() const {
    return false;
}

bool SampleAntiFlicker::set_frequency_band(uint32_t low_freq, uint32_t high_freq) {
    return false;
}

uint32_t SampleAntiFlicker::get_band_low_frequency() const {
    return 50;
}

uint32_t SampleAntiFlicker::get_band_high_frequency() const {
    return 520;
}

std::pair<uint32_t, uint32_t> SampleAntiFlicker::get_frequency_band() const {
    return std::make_pair(0, 10);
}

uint32_t SampleAntiFlicker::get_min_supported_frequency() const {
    return 50;
}

uint32_t SampleAntiFlicker::get_max_supported_frequency() const {
    return 520;
}

bool SampleAntiFlicker::set_filtering_mode(AntiFlickerMode mode) {
    return false;
}

Metavision::I_AntiFlickerModule::AntiFlickerMode SampleAntiFlicker::get_filtering_mode() const {
    return Metavision::I_AntiFlickerModule::AntiFlickerMode::BAND_PASS;
}

bool SampleAntiFlicker::set_duty_cycle(float duty_cycle) {
    return false;
}

float SampleAntiFlicker::get_duty_cycle() const {
    return 0.0;
}

float SampleAntiFlicker::get_min_supported_duty_cycle() const {
    return 1.0 / 16.0;
}

float SampleAntiFlicker::get_max_supported_duty_cycle() const {
    return 100.0;
}

bool SampleAntiFlicker::set_start_threshold(uint32_t threshold) {
    return false;
}

bool SampleAntiFlicker::set_stop_threshold(uint32_t threshold) {
    return false;
}

uint32_t SampleAntiFlicker::get_start_threshold() const {
    return 0;
}

uint32_t SampleAntiFlicker::get_stop_threshold() const {
    return (1 << 3) - 1;
}

uint32_t SampleAntiFlicker::get_min_supported_start_threshold() const {
    return 0;
}

uint32_t SampleAntiFlicker::get_max_supported_start_threshold() const {
    return (1 << 3) - 1;
}

uint32_t SampleAntiFlicker::get_min_supported_stop_threshold() const {
    return 0;
}

uint32_t SampleAntiFlicker::get_max_supported_stop_threshold() const {
    return (1 << 3) - 1;
}