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

#ifndef METAVISION_HAL_GEN3_PATTERN_CHECKERS_H
#define METAVISION_HAL_GEN3_PATTERN_CHECKERS_H

#include "utils/psee_pattern_generator_checker.h"

namespace Metavision {

class Gen3PatternChecker : public PseePatternGeneratorChecker::PatternChecker {
protected:
    virtual bool check_init(const Event2d *&ev_begin, const Event2d *ev_end);

protected:
    /// Last event processed
    Event2d last_ev_;

private:
    /// Checks if first event has been received to begin the pattern verification
    bool is_init_{false};
};

class ColumnPatternChecker : public Gen3PatternChecker {
public:
    ColumnPatternChecker(int width, int height);
    std::vector<PseePatternGeneratorChecker::Error> check(const Event2d *ev_begin,
                                                          const Event2d *ev_end) override final;

private:
    const int height_;
    const int pixels_count_;
};

class SlashPatternChecker : public Gen3PatternChecker {
public:
    SlashPatternChecker(int width, int height);
    std::vector<PseePatternGeneratorChecker::Error> check(const Event2d *ev_begin,
                                                          const Event2d *ev_end) override final;

private:
    const int pattern_min_dim_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN3_PATTERN_CHECKERS_H
