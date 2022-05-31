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

#ifndef METAVISION_HAL_GEN31_PATTERN_GENERATOR_CHECKER_H
#define METAVISION_HAL_GEN31_PATTERN_GENERATOR_CHECKER_H

#include "utils/psee_pattern_generator_checker.h"

namespace Metavision {

class Gen31PatternGeneratorChecker : public PseePatternGeneratorChecker {
public:
    PseePatternGeneratorChecker::PatternChecker *
        build_pattern_checker(const PseePatternGenerator::Configuration &configuration) const override final;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN31_PATTERN_GENERATOR_CHECKER_H
