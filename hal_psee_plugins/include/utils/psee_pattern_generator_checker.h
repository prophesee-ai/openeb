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

#ifndef METAVISION_HAL_PSEE_PATTERN_GENERATOR_CHECKER_H
#define METAVISION_HAL_PSEE_PATTERN_GENERATOR_CHECKER_H

#include <vector>

#include "metavision/sdk/base/events/event2d.h"
#include "utils/psee_pattern_generator.h"

using Event2d = Metavision::Event2d;

namespace Metavision {

class PseePatternGeneratorChecker {
public:
    struct Error {
        int idx;
        size_t lost;
    };

    class PatternChecker {
    public:
        virtual ~PatternChecker();
        virtual std::vector<Error> check(const Event2d *ev_begin, const Event2d *ev_end) = 0;
    };

    virtual ~PseePatternGeneratorChecker();

    /// @brief Returns a pattern checker for the sensor
    virtual PatternChecker *build_pattern_checker(const PseePatternGenerator::Configuration &configuration) const = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_PATTERN_GENERATOR_CHECKER_H
