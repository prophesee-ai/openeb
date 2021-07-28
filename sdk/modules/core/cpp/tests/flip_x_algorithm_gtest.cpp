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

#include <gtest/gtest.h>
#include <vector>

#include "metavision/sdk/core/algorithms/flip_x_algorithm.h"
#include "metavision/sdk/base/events/event2d.h"

TEST(FlipXAlgorithm_GTest, constructor) {
    // GIVEN a FlipXAlgorithm instance
    Metavision::FlipXAlgorithm algo(15);

    // WHEN getting the width minus one
    std::int16_t width_minus_one = algo.width_minus_one();

    // THEN the value of the width minus corresponds to the value passed in the constructor
    EXPECT_EQ(15, width_minus_one);
}

TEST(FlipXAlgorithm_GTest, set_width_minus_one) {
    // GIVEN a FlipXAlgorithm instance
    Metavision::FlipXAlgorithm algo(12);

    // WHEN setting the width minus one
    algo.set_width_minus_one(415);

    // THEN the value of the width minus corresponds to the value set
    EXPECT_EQ(415, algo.width_minus_one());
}

TEST(FlipXAlgorithm_GTest, function_call_operator_even_width) {
    // GIVEN a FlipXAlgorithm instance, where the width on the input events is even
    std::int16_t width = 240;
    Metavision::FlipXAlgorithm algo(width - 1);

    // WHEN processing an event
    Metavision::Event2d ev(100, 45, 0, 4563);
    algo(ev);

    // THEN the x value of the event is flipped
    EXPECT_EQ(139, ev.x);
}

TEST(FlipXAlgorithm_GTest, function_call_operator_odd_width) {
    // GIVEN a FlipXAlgorithm instance, where the width on the input events is odd
    std::int16_t width = 111;
    Metavision::FlipXAlgorithm algo(width - 1);

    // WHEN processing the event in the middle
    Metavision::Event2d ev(55, 45, 0, 4563);
    algo(ev);

    // THEN the x value of the event remains unchanged
    EXPECT_EQ(55, ev.x);
}

TEST(FlipXAlgorithm_GTest, process) {
    // GIVEN a FlipXAlgorithm instance
    std::int16_t width = 120;
    Metavision::FlipXAlgorithm algo(width - 1);

    // WHEN processing a batch of events
    std::vector<Metavision::Event2d> input_events = {
        Metavision::Event2d(0, 45, 0, 155),    Metavision::Event2d(119, 8, 1, 980),
        Metavision::Event2d(59, 77, 1, 1104),  Metavision::Event2d(100, 64, 0, 5200),
        Metavision::Event2d(60, 14, 1, 10697), Metavision::Event2d(4, 111, 0, 20145)};
    std::vector<Metavision::Event2d> output_events;
    algo.process_events(input_events.begin(), input_events.end(), std::back_inserter(output_events));

    // THEN the output events are the one expected
    std::vector<Metavision::Event2d> expected_events = {
        Metavision::Event2d(119, 45, 0, 155),  Metavision::Event2d(0, 8, 1, 980),
        Metavision::Event2d(60, 77, 1, 1104),  Metavision::Event2d(19, 64, 0, 5200),
        Metavision::Event2d(59, 14, 1, 10697), Metavision::Event2d(115, 111, 0, 20145)};
    ASSERT_EQ(expected_events.size(), output_events.size());
    for (auto it_exp = expected_events.begin(), it_exp_end = expected_events.end(), it = output_events.begin();
         it_exp != it_exp_end; ++it_exp, ++it) {
        EXPECT_EQ(it_exp->x, it->x);
        EXPECT_EQ(it_exp->y, it->y);
        EXPECT_EQ(it_exp->p, it->p);
        EXPECT_EQ(it_exp->t, it->t);
    }
}
