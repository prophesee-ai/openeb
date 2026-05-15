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

#include <algorithm>

#include "devices/utils/gen3/gen3_pattern_checkers.h"

namespace Metavision {

bool Gen3PatternChecker::check_init(const Event2d *&ev, const Event2d *ev_end) {
    if (is_init_) {
        return true;
    }

    if (ev < ev_end) {
        is_init_ = true;
        last_ev_ = *ev;
        ++ev;
        return true;
    }
    return false;
}

ColumnPatternChecker::ColumnPatternChecker(int width, int height) : height_(height), pixels_count_(width * height) {}

SlashPatternChecker::SlashPatternChecker(int width, int height) : pattern_min_dim_(std::min(width, height)) {}

std::vector<PseePatternGeneratorChecker::Error> ColumnPatternChecker::check(const Event2d *ev_begin,
                                                                            const Event2d *ev_end) {
    std::vector<PseePatternGeneratorChecker::Error> errors_found;
    const Event2d *ev = ev_begin;

    if (!check_init(ev, ev_end)) {
        return errors_found;
    }

    // Column scan: goes column wise from 0 to height, from 0 to width
    int new_ev_index;
    int last_ev_index = (last_ev_.x * height_ + last_ev_.y);

    while (ev < ev_end) {
        new_ev_index   = (ev->x * height_ + ev->y);
        int index_diff = new_ev_index - last_ev_index;

        /*
         * Two successive pixels always have an index diff of 1 except if we reach the bottom right corner.
         * In this case the new event will occur at the top left corner. Thus the index diff is 0 -
             ((pattern_width_
         * - 1) * patter_height + pattern_height - 1) Which is -(pattern_pixels_count_ - 1)
         */
        if ((index_diff != 1) && (index_diff != (-(pixels_count_ - 1)))) {
            /*
             * Compute the data missing from the index difference. When no data are missing, this
             * This line handles both cases when index_diff is > 1 or index_diff <= 0
             * if index diff is 0 (i.e. new event is the same as the previous) then we expect a full pattern gap
             * (pattern_pixels_count_ - 1).
             */
            uint32_t missing = (pixels_count_ + index_diff - 1) % pixels_count_;
            PseePatternGeneratorChecker::Error error;
            error.idx  = std::distance(ev_begin, ev);
            error.lost = missing;
            errors_found.push_back(error);
        } /* end if */

        last_ev_index = new_ev_index;
        last_ev_      = *ev;
        ++ev;
    } /* end while */
    return errors_found;
}

std::vector<PseePatternGeneratorChecker::Error> SlashPatternChecker::check(const Event2d *ev_begin,
                                                                           const Event2d *ev_end) {
    std::vector<PseePatternGeneratorChecker::Error> errors_found;
    const Event2d *ev = ev_begin;

    if (!check_init(ev, ev_end)) {
        return errors_found;
    }

    // slash scan: goes diagonaly from 0 to min(pattern_width, pattern_height)
    for (; ev < ev_end; ++ev) {
        int index_diff = ev->x - last_ev_.x;
        /*
         * Two successive pixels always have an index diff of 1 except if we reach the bottom right corner.
         * In this case the new event will occur at the top left corner. As we have a slash pattern type, we follow
         the
         * diagonal from 0,0 to (minimal pattern dimension, minimal pattern dimension). Thus the index diff is 0 -
         * (pattern_min_dimension_ - 1) which is -(pattern_min_dimension_ - 1)
         */
        if ((index_diff != 1) && (index_diff != (-(pattern_min_dim_ - 1)))) {
            /*
             * Compute the data missing from the index difference. When no data are missing, this
             * This line handles both cases when index_diff is > 1 or index_diff <= 0
             * if index diff is 0 (i.e. new event is the same as the previous) then we expect a full pattern gap
             * (pattern_pixels_count_ - 1).
             */
            uint32_t missing = (pattern_min_dim_ + index_diff - 1) % pattern_min_dim_;
            PseePatternGeneratorChecker::Error error;
            error.idx  = std::distance(ev_begin, ev);
            error.lost = missing;
            errors_found.push_back(error);
        } /* end if */

        last_ev_ = *ev;
    } /* end while */
    return errors_found;
}

} // namespace Metavision
