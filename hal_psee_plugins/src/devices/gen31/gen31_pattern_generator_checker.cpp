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

#include "metavision/hal/utils/hal_log.h"
#include "devices/gen31/gen31_pattern_generator.h"
#include "devices/utils/gen3/gen3_pattern_checkers.h"
#include "devices/gen31/gen31_pattern_generator_checker.h"

namespace Metavision {

PseePatternGeneratorChecker::PatternChecker *Gen31PatternGeneratorChecker::build_pattern_checker(
    const PseePatternGenerator::Configuration &configuration) const {
    switch (configuration.pattern_type) {
    case PseePatternGenerator::Configuration::PatternType::Column:
        return new ColumnPatternChecker(Gen31PatternGenerator::PATTERN_GENERATOR_WIDTH,
                                        Gen31PatternGenerator::PATTERN_GENERATOR_HEIGHT);
    case PseePatternGenerator::Configuration::PatternType::Slash:
        return new SlashPatternChecker(Gen31PatternGenerator::PATTERN_GENERATOR_WIDTH,
                                       Gen31PatternGenerator::PATTERN_GENERATOR_HEIGHT);
    default:
        MV_HAL_LOG_ERROR() << "Unavailable pattern type for this sensor. Available patterns are Column or Slash.";
        return nullptr;
    }
}

} // namespace Metavision
