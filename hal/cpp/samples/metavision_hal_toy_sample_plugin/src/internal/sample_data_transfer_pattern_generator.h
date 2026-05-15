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

#ifndef METAVISION_HAL_SAMPLE_DATA_TRANSFER_PATTERN_GENERATOR_H
#define METAVISION_HAL_SAMPLE_DATA_TRANSFER_PATTERN_GENERATOR_H

#include <random>
#include <algorithm>

#include "sample_data_transfer.h"
#include "sample_events_format.h"
#include "sample_geometry.h"

// Utils class to generate some fake events (no need for it when writing your own plugin)
struct SampleDataTransfer::PatternGenerator {
    PatternGenerator() {
        current_x_ne_ = std::max(SIZE_SQUARE, d_x_(engine_));
        current_y_ne_ = std::max(SIZE_SQUARE, d_y_(engine_));
        current_x_so_ = current_x_ne_ - SIZE_SQUARE;
        current_y_so_ = current_y_ne_ - SIZE_SQUARE;
    }

    void operator()(SampleEventsFormat &ev, Metavision::timestamp &current_time) {
        // Update
        if (idx_ == 8 * SIZE_SQUARE) {
            idx_ = 0;
            if (x_step_ > 0) {
                if (current_x_ne_ == SampleGeometry::WIDTH_ - 1) {
                    x_step_ = -1;
                }
            } else {
                if (current_x_so_ == 0) {
                    x_step_ = 1;
                }
            }

            if (y_step_ > 0) {
                if (current_y_ne_ == SampleGeometry::HEIGHT_ - 1) {
                    y_step_ = -1;
                }
            } else {
                if (current_y_so_ == 0) {
                    y_step_ = +1;
                }
            }

            current_x_so_ += x_step_;
            current_x_ne_ += x_step_;
            current_y_so_ += y_step_;
            current_y_ne_ += y_step_;

            // do random
            doing_random_ = true;
        }

        if (doing_random_) {
            current_time += STEP_RANDOM;
            encode_sample_format(ev, d_x_(engine_), d_y_(engine_), d_p_(engine_), current_time);

            ++n_random_;
            if (n_random_ == N_RANDOM) {
                n_random_     = 0;
                doing_random_ = false;
            }
        } else {
            short idx_curr = idx_ + 1;
            if (idx_curr == 4 * SIZE_SQUARE) {
                current_time += 5;
            }
            idx_         = idx_ % (4 * SIZE_SQUARE);
            short offset = idx_ % SIZE_SQUARE;
            if (idx_ < SIZE_SQUARE) {
                encode_sample_format(ev, (current_x_so_ + idx_) % SampleGeometry::WIDTH_, current_y_so_, (y_step_ < 0),
                                     current_time);
            } else if (idx_ < 2 * SIZE_SQUARE) {
                encode_sample_format(ev, current_x_ne_, (current_y_so_ + offset) % SampleGeometry::HEIGHT_,
                                     (x_step_ > 0), current_time);
            } else if (idx_ < 3 * SIZE_SQUARE) {
                encode_sample_format(ev, (current_x_so_ + offset) % SampleGeometry::WIDTH_, current_y_ne_,
                                     (y_step_ > 0), current_time);
            } else {
                encode_sample_format(ev, current_x_so_, (current_y_so_ + offset) % SampleGeometry::HEIGHT_,
                                     (x_step_ < 0), current_time);
            }
            idx_ = idx_curr;
        }
    }

private:
    std::random_device rd_{};
    std::mt19937 engine_{rd_()};
    std::uniform_int_distribution<short> d_x_{0, SampleGeometry::WIDTH_ - 1};
    std::uniform_int_distribution<short> d_y_{0, SampleGeometry::HEIGHT_ - 1};
    std::uniform_int_distribution<short> d_p_{0, 1};

    static constexpr short SIZE_SQUARE = 40;
    short current_x_ne_, current_y_ne_, current_x_so_, current_y_so_;
    short x_step_ = 1, y_step_ = 1, idx_ = 0;

    static constexpr short N_RANDOM                    = 10;
    static constexpr Metavision::timestamp STEP_RANDOM = 200;
    bool doing_random_                                 = false;
    int n_random_                                      = 0;
};

#endif // METAVISION_HAL_SAMPLE_DATA_TRANSFER_PATTERN_GENERATOR_H
