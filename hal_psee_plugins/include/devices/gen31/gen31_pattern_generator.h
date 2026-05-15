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

#ifndef METAVISION_HAL_GEN31_PATTERN_GENERATOR_H
#define METAVISION_HAL_GEN31_PATTERN_GENERATOR_H

#include <memory>

#include "utils/psee_pattern_generator.h"

namespace Metavision {
class Fx3LibUSBBoardCommand;
class RegisterMap;

class Gen31PatternGenerator : public PseePatternGenerator {
public:
    Gen31PatternGenerator(const std::shared_ptr<RegisterMap> &regmap);
    ~Gen31PatternGenerator();

    bool enable(const PseePatternGenerator::Configuration &configuration) override final;
    void disable() override final;
    bool is_enabled() override final;
    void get_pattern_geometry(int &width, int &height) const override final;

    void set_period_step_count(uint16_t n_step_count, uint16_t p_step_count) override final;
    void set_period_rate(uint8_t n_rate_Mev_s, uint8_t p_rate_Mev_s) override final;

public:
    static constexpr int PATTERN_GENERATOR_WIDTH  = 512;
    static constexpr int PATTERN_GENERATOR_HEIGHT = 1024;

private:
    std::shared_ptr<RegisterMap> register_map_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN31_PATTERN_GENERATOR_H
