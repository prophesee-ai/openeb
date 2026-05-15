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

#include "metavision/sdk/core/algorithms/event_rescaler_algorithm.h"
#include "metavision/sdk/base/events/event2d.h"

using Event2d = Metavision::Event2d;

TEST(EventRescalerAlgorithm_GTest, constructor_valid_parameters) {
    // GIVEN correct input scales
    const float scale_width = 0.5f, scale_height = 0.5f;

    // WHEN instantiating the algorithm
    // THEN no error is thrown
    ASSERT_NO_THROW(Metavision::EventRescalerAlgorithm algo(scale_width, scale_height));
}

TEST(EventRescalerAlgorithm_GTest, constructor_invalid_parameters) {
    // GIVEN incorrect input scales
    float scale_width = -0.5f, scale_height = 0.5f;

    // WHEN instantiating the algorithm
    // THEN an error is thrown
    EXPECT_THROW(Metavision::EventRescalerAlgorithm algo(scale_width, scale_height), std::runtime_error);

    // GIVEN incorrect input scales
    scale_width  = 0.5f;
    scale_height = 0.f;

    // WHEN instantiating the algorithm
    // THEN an error is thrown
    EXPECT_THROW(Metavision::EventRescalerAlgorithm algo(scale_width, scale_height), std::runtime_error);
}

TEST(EventRescalerAlgorithm_GTest, process_events) {
    // GIVEN an instance of rescaler
    const float scale_width = 0.5f, scale_height = 0.5f;
    Metavision::EventRescalerAlgorithm algo(scale_width, scale_height);

    // WHEN processing an input vector of events
    std::vector<Event2d> events{Event2d(0, 0, 0, 0), Event2d(9, 10, 1, 10)};
    std::vector<Event2d> output_events;
    algo.process_events(events.cbegin(), events.cend(), std::back_inserter(output_events));

    // THEN the produced events are correct
    ASSERT_TRUE(output_events.size() == events.size());
    ASSERT_TRUE(output_events[0].x == 0 && output_events[0].y == 0 && output_events[0].p == 0 &&
                output_events[0].t == 0);
    ASSERT_TRUE(output_events[1].x == 4 && output_events[1].y == 5 && output_events[1].p == 1 &&
                output_events[1].t == 10);
}
