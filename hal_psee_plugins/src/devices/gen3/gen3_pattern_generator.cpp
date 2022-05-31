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

#include "boards/utils/psee_libusb_board_command.h"
#include "devices/gen3/gen3_pattern_generator.h"
#include "devices/gen3/legacy_regmap_headers/legacy/stereo_pc_mapping.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

static constexpr uint16_t DEFAULT_PG_LENGTH     = 3072;
static constexpr uint16_t DEFAULT_PG_STEP_COUNT = 1;
static constexpr uint32_t P_PERIOD_LENGTH_MASK  = ((1 << CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_LENGTH_WIDTH) - 1);
static constexpr uint32_t N_PERIOD_LENGTH_MASK  = ((1 << CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_LENGTH_WIDTH) - 1);
static constexpr uint32_t P_PERIOD_RATIO_MASK   = ((1 << CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_VALID_RATIO_WIDTH) - 1);
static constexpr uint32_t N_PERIOD_RATIO_MASK   = ((1 << CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_VALID_RATIO_WIDTH) - 1);

Gen3PatternGenerator::Gen3PatternGenerator(const std::shared_ptr<PseeLibUSBBoardCommand> &board_cmd) :
    board_command_(board_cmd) {
    disable();
}

Gen3PatternGenerator::~Gen3PatternGenerator() {
    disable();
}

bool Gen3PatternGenerator::enable(const PseePatternGenerator::Configuration &configuration) {
    if (configuration.pattern_type != PseePatternGenerator::Configuration::PatternType::Column &&
        configuration.pattern_type != PseePatternGenerator::Configuration::PatternType::Slash) {
        MV_HAL_LOG_ERROR()
            << "Failed to enable pattern generator. Unsupported input pattern type for this sensor. Supported "
               "types are Column or Slash.";
        return false;
    }

    // Some pattern generator configurations can be set only if it is disabled.
    disable();

    // set the pattern config
    uint32_t pattern_config = 0;

    pattern_config |=
        (static_cast<uint8_t>(configuration.pattern_type) << CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_TYPE_BIT_IDX);
    pattern_config |=
        (static_cast<uint8_t>(configuration.pixel_type) << CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_PIXEL_TYPE_BIT_IDX);
    pattern_config |= (configuration.pixel_polarity << CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_PIXEL_POLARITY_BIT_IDX);
    pattern_config |= (1 << CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_ENABLE_BIT_IDX); // enables

    if (!is_period_rate_set_) {
        set_period_rate(DEFAULT_PG_STEP_COUNT, DEFAULT_PG_STEP_COUNT);
    }

    if (!is_period_length_set_) {
        set_period_step_count(DEFAULT_PG_LENGTH, DEFAULT_PG_LENGTH);
    }

    board_command_->write_register(CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_ADDR, pattern_config);

    return true;
}

void Gen3PatternGenerator::disable() {
    board_command_->send_register_bit(CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_ADDR,
                                      CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_ENABLE_BIT_IDX, 0);
}

bool Gen3PatternGenerator::is_enabled() {
    return board_command_->read_register_bit(CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_ADDR,
                                             CCAM3_SISLEY_IF_TEST_PATTERN_CONTROL_ENABLE_BIT_IDX);
}

void Gen3PatternGenerator::get_pattern_geometry(int &width, int &height) const {
    width  = PATTERN_GENERATOR_WIDTH;
    height = PATTERN_GENERATOR_HEIGHT;
}

void Gen3PatternGenerator::set_period_step_count(uint16_t n_step_count, uint16_t p_step_count) {
    if (p_step_count == 0) {
        p_step_count = n_step_count;
    }
    if (n_step_count == 0) {
        return;
    }

    const auto n_period_length = n_step_count * 10; // 10 ns is the step length
    const auto p_period_length = p_step_count * 10; // 10 ns is the step length

    // Keep the period ratio unchanged
    uint32_t n_period_reg_value = board_command_->read_register(CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_ADDR) &
                                  (N_PERIOD_RATIO_MASK << CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_VALID_RATIO_BIT_IDX);
    uint32_t p_period_reg_value = board_command_->read_register(CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_ADDR) &
                                  (P_PERIOD_RATIO_MASK << CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_VALID_RATIO_BIT_IDX);

    const uint32_t n_period_length_reg_value = (n_period_length & N_PERIOD_LENGTH_MASK)
                                               << CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_LENGTH_BIT_IDX;
    const uint32_t p_period_length_reg_value = (p_period_length & P_PERIOD_LENGTH_MASK)
                                               << CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_LENGTH_BIT_IDX;

    // write the full bitset
    board_command_->write_register(CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_ADDR,
                                   n_period_length_reg_value | n_period_reg_value);
    board_command_->write_register(CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_ADDR,
                                   p_period_length_reg_value | p_period_reg_value);

    is_period_length_set_ = true;
}

void Gen3PatternGenerator::set_period_rate(uint8_t n_rate_Mev_s, uint8_t p_rate_Mev_s) {
    if (p_rate_Mev_s == 0) {
        p_rate_Mev_s = n_rate_Mev_s;
    }
    if (n_rate_Mev_s == 0) {
        return;
    }

    // Keep the period length unchanged
    uint32_t n_period_reg_value = board_command_->read_register(CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_ADDR) &
                                  (N_PERIOD_LENGTH_MASK << CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_LENGTH_BIT_IDX);
    uint32_t p_period_reg_value = board_command_->read_register(CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_ADDR) &
                                  (P_PERIOD_LENGTH_MASK << CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_LENGTH_BIT_IDX);

    // PG can generate 1 event per clock cycle with a clock period of 10 ns. And there are 1024 cycles.
    // So at most you can generate up to 1023 ev / 10240 ns which is equivalent to 100*1023/1024 Mev/s.
    // Rate step is thus 100/1024 Mev/s.
    // In the register you write the ratio i.e. number of steps (actual rate requested/rate_step).
    const uint32_t n_period_ratio_reg_value = ((1024 * n_rate_Mev_s / 100) & N_PERIOD_RATIO_MASK)
                                              << CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_VALID_RATIO_BIT_IDX;
    const uint32_t p_period_ratio_reg_value = ((1024 * p_rate_Mev_s / 100) & P_PERIOD_RATIO_MASK)
                                              << CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_VALID_RATIO_BIT_IDX;

    // write the full bitset
    board_command_->write_register(CCAM3_SISLEY_IF_TEST_PATTERN_N_PERIOD_ADDR,
                                   n_period_ratio_reg_value | n_period_reg_value);
    board_command_->write_register(CCAM3_SISLEY_IF_TEST_PATTERN_P_PERIOD_ADDR,
                                   p_period_ratio_reg_value | p_period_reg_value);

    is_period_rate_set_ = true;
}
} // namespace Metavision
