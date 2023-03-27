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

#include <iostream>

#include "devices/gen31/gen31_pattern_generator.h"
#include "metavision/psee_hw_layer/boards/fx3/fx3_libusb_board_command.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {
using vfield = std::map<std::string, uint32_t>;

static constexpr uint16_t DEFAULT_PG_LENGTH     = 3072;
static constexpr uint16_t DEFAULT_PG_STEP_COUNT = 1;

Gen31PatternGenerator::Gen31PatternGenerator(const std::shared_ptr<RegisterMap> &regmap) : register_map_(regmap) {
    disable();
}

Gen31PatternGenerator::~Gen31PatternGenerator() {
    disable();
}

bool Gen31PatternGenerator::enable(const PseePatternGenerator::Configuration &configuration) {
    if (configuration.pattern_type != PseePatternGenerator::Configuration::PatternType::Column &&
        configuration.pattern_type != PseePatternGenerator::Configuration::PatternType::Slash) {
        MV_HAL_LOG_ERROR()
            << "Failed to enable pattern generator. Unsupported input pattern type for this sensor. Supported "
               "types are Column or Slash.";
        return false;
    }

    // Some pattern generator configurations can be set only if it is disabled.
    disable();

    if (!is_period_rate_set_) {
        set_period_rate(DEFAULT_PG_STEP_COUNT, DEFAULT_PG_STEP_COUNT);
    }

    if (!is_period_length_set_) {
        set_period_step_count(DEFAULT_PG_LENGTH, DEFAULT_PG_LENGTH);
    }

    (*register_map_)["GEN31_IF/TEST_PATTERN_CONTROL"].write_value(vfield{
        {"ENABLE", true},
        {"TYPE", (uint32_t)configuration.pattern_type},
        {"PIXEL_TYPE", (uint32_t)configuration.pixel_type},
        {"PIXEL_POLARITY", (uint32_t)configuration.pixel_polarity},
    });

    return true;
}

void Gen31PatternGenerator::disable() {
    (*register_map_)["SENSOR_IF/GEN31_IF/TEST_PATTERN_CONTROL"]["ENABLE"] = 0;
}

bool Gen31PatternGenerator::is_enabled() {
    return (*register_map_)["SENSOR_IF/GEN31_IF/TEST_PATTERN_CONTROL"]["ENABLE"].read_value();
}

void Gen31PatternGenerator::get_pattern_geometry(int &width, int &height) const {
    width  = PATTERN_GENERATOR_WIDTH;
    height = PATTERN_GENERATOR_HEIGHT;
}

void Gen31PatternGenerator::set_period_step_count(uint16_t n_step_count, uint16_t p_step_count) {
    if (p_step_count == 0) {
        p_step_count = n_step_count;
    }
    if (n_step_count == 0) {
        return;
    }

    const auto n_period_length = n_step_count * 10; // 10 ns is the step length
    const auto p_period_length = p_step_count * 10; // 10 ns is the step length

    (*register_map_)["SENSOR_IF/GEN31_IF/TEST_PATTERN_N_PERIOD"]["LENGTH"] = n_period_length;
    (*register_map_)["SENSOR_IF/GEN31_IF/TEST_PATTERN_P_PERIOD"]["LENGTH"] = p_period_length;

    is_period_length_set_ = true;
}

void Gen31PatternGenerator::set_period_rate(uint8_t n_rate_Mev_s, uint8_t p_rate_Mev_s) {
    if (p_rate_Mev_s == 0) {
        p_rate_Mev_s = n_rate_Mev_s;
    }
    if (n_rate_Mev_s == 0) {
        return;
    }

    // PG can generate 1 event per clock cycle with a clock period of 10 ns. And there are 1024 cycles.
    // So at most you can generate up to 1023 ev / 10240 ns which is equivalent to 100*1023/1024 Mev/s.
    // Rate step is thus 100/1024 Mev/s.
    // In the register you write the ratio i.e. number of steps (actual rate requested/rate_step).
    const uint32_t n_period_ratio_reg_value = (1024 * n_rate_Mev_s / 100);
    const uint32_t p_period_ratio_reg_value = (1024 * p_rate_Mev_s / 100);

    (*register_map_)["SENSOR_IF/GEN31_IF/TEST_PATTERN_N_PERIOD"]["VALID_RATIO"] = n_period_ratio_reg_value;
    (*register_map_)["SENSOR_IF/GEN31_IF/TEST_PATTERN_P_PERIOD"]["VALID_RATIO"] = p_period_ratio_reg_value;

    is_period_rate_set_ = true;
}
} // namespace Metavision
