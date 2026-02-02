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
#include <opencv2/core.hpp>

#include "metavision/sdk/core/algorithms/event_frame_diff_generation_algorithm.h"

using namespace Metavision;
using InputIt = std::vector<EventCD>::const_iterator;

TEST(EventFrameDiffGenerationAlgorithm_GTest, nominal) {
    // GIVEN a 3x2 toy event stream
    const unsigned int width = 3, height = 2;
    std::vector<EventCD> events = {EventCD(0, 0, 0, 1), EventCD(2, 0, 0, 1), EventCD(0, 1, 1, 1), EventCD(2, 1, 1, 1),
                                   EventCD(0, 0, 1, 2), EventCD(2, 0, 1, 2), EventCD(0, 1, 1, 2), EventCD(2, 1, 0, 2),
                                   EventCD(0, 0, 0, 3), EventCD(2, 0, 1, 3), EventCD(0, 1, 0, 3), EventCD(2, 1, 0, 3)};
    // GIVEN a EventFrameDiffGenerationAlgorithm instance
    EventFrameDiffGenerationAlgorithm<InputIt> diff_generator(width, height);

    // WHEN we process the event stream and generate the event frame
    diff_generator.process_events(events.cbegin(), events.cend());
    RawEventFrameDiff event_frame;
    diff_generator.generate(event_frame);

    // THEN the generated event frame matches the expectations
    const auto &diff_data = event_frame.get_data();
    EXPECT_EQ(diff_data.size(), 3 * 2);
    EXPECT_EQ(diff_data[0 + 3 * 0], -1);
    EXPECT_EQ(diff_data[1 + 3 * 0], 0);
    EXPECT_EQ(diff_data[2 + 3 * 0], 1);
    EXPECT_EQ(diff_data[0 + 3 * 1], 1);
    EXPECT_EQ(diff_data[1 + 3 * 1], 0);
    EXPECT_EQ(diff_data[2 + 3 * 1], -1);
}

TEST(EventFrameDiffGenerationAlgorithm_GTest, many_negatives_no_rollover) {
    // GIVEN a 1x1 toy event stream with 150 negative events and 50 positive events
    const unsigned int width = 1, height = 1;
    std::vector<EventCD> events;
    timestamp t = 1;
    for (int i = 0; i < 150; ++i) {
        events.emplace_back(0, 0, 0, t++);
    }
    for (int i = 0; i < 50; ++i) {
        events.emplace_back(0, 0, 1, t++);
    }
    // GIVEN a EventFrameDiffGenerationAlgorithm instance
    const unsigned int bit_size = 8;
    const bool allow_rollover   = false;
    EventFrameDiffGenerationAlgorithm<InputIt> diff_generator(width, height, bit_size, allow_rollover);

    // WHEN we process the event stream and generate the event frame
    diff_generator.process_events(events.cbegin(), events.cend());
    RawEventFrameDiff event_frame;
    diff_generator.generate(event_frame);

    // THEN the generated event frame matches the expectations
    const auto &diff_data = event_frame.get_data();
    EXPECT_EQ(diff_data.size(), 1 * 1);
    EXPECT_EQ(diff_data[0], -128 + 50);
}

TEST(EventFrameDiffGenerationAlgorithm_GTest, many_negatives_with_rollover) {
    // GIVEN a 1x1 toy event stream with 150 negative events and 50 positive events
    const unsigned int width = 1, height = 1;
    std::vector<EventCD> events;
    timestamp t = 1;
    for (int i = 0; i < 150; ++i) {
        events.emplace_back(0, 0, 0, t++);
    }
    for (int i = 0; i < 50; ++i) {
        events.emplace_back(0, 0, 1, t++);
    }
    // GIVEN a EventFrameDiffGenerationAlgorithm instance
    const unsigned int bit_size = 8;
    const bool allow_rollover   = true;
    EventFrameDiffGenerationAlgorithm<InputIt> diff_generator(width, height, bit_size, allow_rollover);

    // WHEN we process the event stream and generate the event frame
    diff_generator.process_events(events.cbegin(), events.cend());
    RawEventFrameDiff event_frame;
    diff_generator.generate(event_frame);

    // THEN the generated event frame matches the expectations
    const auto &diff_data = event_frame.get_data();
    EXPECT_EQ(diff_data.size(), 1 * 1);
    EXPECT_EQ(diff_data[0], -150 + 50);
}

TEST(EventFrameDiffGenerationAlgorithm_GTest, many_negatives_no_rollover_low_bit_size) {
    // GIVEN a 1x1 toy event stream with 5 negative events
    const unsigned int width = 1, height = 1;
    std::vector<EventCD> events;
    for (int t = 0; t < 5; ++t) {
        events.emplace_back(0, 0, 0, 1 + t);
    }
    // GIVEN a EventFrameDiffGenerationAlgorithm instance
    const unsigned int bit_size = 3;
    const bool allow_rollover   = false;
    EventFrameDiffGenerationAlgorithm<InputIt> diff_generator(width, height, bit_size, allow_rollover);

    // WHEN we process the event stream and generate the event frame
    diff_generator.process_events(events.cbegin(), events.cend());
    RawEventFrameDiff event_frame;
    diff_generator.generate(event_frame);

    // THEN the generated event frame matches the expectations
    const auto &diff_data = event_frame.get_data();
    EXPECT_EQ(diff_data.size(), 1 * 1);
    EXPECT_EQ(diff_data[0], -4);
}

TEST(EventFrameDiffGenerationAlgorithm_GTest, many_positives_no_rollover) {
    // GIVEN a 1x1 toy event stream with 150 positive events and 50 negative events
    const unsigned int width = 1, height = 1;
    std::vector<EventCD> events;
    timestamp t = 1;
    for (int i = 0; i < 150; ++i) {
        events.emplace_back(0, 0, 1, t++);
    }
    for (int i = 0; i < 50; ++i) {
        events.emplace_back(0, 0, 0, t++);
    }
    // GIVEN a EventFrameDiffGenerationAlgorithm instance
    const unsigned int bit_size = 8;
    const bool allow_rollover   = false;
    EventFrameDiffGenerationAlgorithm<InputIt> diff_generator(width, height, bit_size, allow_rollover);

    // WHEN we process the event stream and generate the event frame
    diff_generator.process_events(events.cbegin(), events.cend());
    RawEventFrameDiff event_frame;
    diff_generator.generate(event_frame);

    // THEN the generated event frame matches the expectations
    const auto &diff_data = event_frame.get_data();
    EXPECT_EQ(diff_data.size(), 1 * 1);
    EXPECT_EQ(diff_data[0], 127 - 50);
}

TEST(EventFrameDiffGenerationAlgorithm_GTest, many_positives_with_rollover) {
    // GIVEN a 1x1 toy event stream with 150 positive events and 50 negative events
    const unsigned int width = 1, height = 1;
    std::vector<EventCD> events;
    timestamp t = 1;
    for (int i = 0; i < 150; ++i) {
        events.emplace_back(0, 0, 1, t++);
    }
    for (int i = 0; i < 50; ++i) {
        events.emplace_back(0, 0, 0, t++);
    }
    // GIVEN a EventFrameDiffGenerationAlgorithm instance
    const unsigned int bit_size = 8;
    const bool allow_rollover   = true;
    EventFrameDiffGenerationAlgorithm<InputIt> diff_generator(width, height, bit_size, allow_rollover);

    // WHEN we process the event stream and generate the event frame
    diff_generator.process_events(events.cbegin(), events.cend());
    RawEventFrameDiff event_frame;
    diff_generator.generate(event_frame);

    // THEN the generated event frame matches the expectations
    const auto &diff_data = event_frame.get_data();
    EXPECT_EQ(diff_data.size(), 1 * 1);
    EXPECT_EQ(diff_data[0], 150 - 50);
}

TEST(EventFrameDiffGenerationAlgorithm_GTest, many_positives_no_rollover_low_bit_size) {
    // GIVEN a 1x1 toy event stream with 5 positive events
    const unsigned int width = 1, height = 1;
    std::vector<EventCD> events;
    for (int t = 0; t < 5; ++t) {
        events.emplace_back(0, 0, 1, 1 + t);
    }
    // GIVEN a EventFrameDiffGenerationAlgorithm instance
    const unsigned int bit_size = 3;
    const bool allow_rollover   = false;
    EventFrameDiffGenerationAlgorithm<InputIt> diff_generator(width, height, bit_size, allow_rollover);

    // WHEN we process the event stream and generate the event frame
    diff_generator.process_events(events.cbegin(), events.cend());
    RawEventFrameDiff event_frame;
    diff_generator.generate(event_frame);

    // THEN the generated event frame matches the expectations
    const auto &diff_data = event_frame.get_data();
    EXPECT_EQ(diff_data.size(), 1 * 1);
    EXPECT_EQ(diff_data[0], 3);
}

TEST(EventFrameDiffGenerationAlgorithm_GTest, lowerbound_generation_period) {
    // GIVEN a 1x1 toy event stream with 21 positive events
    const unsigned int width = 1, height = 1;
    std::vector<EventCD> events;
    for (int t = 0; t < 21; ++t) {
        events.emplace_back(0, 0, 1, 1 + t);
    }
    // GIVEN a EventFrameDiffGenerationAlgorithm instance
    const unsigned int bit_size                     = 8;
    const bool allow_rollover                       = false;
    const timestamp lowerbound_generation_period_us = 10;
    EventFrameDiffGenerationAlgorithm<InputIt> diff_generator(width, height, bit_size, allow_rollover,
                                                              lowerbound_generation_period_us);

    // WHEN we process the event stream
    // THEN we can have feedback if the generation frequency is too high
    RawEventFrameDiff event_frame;
    diff_generator.process_events(events.cbegin(), events.cbegin() + 1);
    EXPECT_TRUE(diff_generator.generate(0, event_frame));
    EXPECT_EQ(event_frame.get_data().size(), 1 * 1);
    EXPECT_EQ(event_frame.get_data()[0], 1);

    diff_generator.process_events(events.cbegin() + 1, events.cbegin() + 6);
    EXPECT_FALSE(diff_generator.generate(5, event_frame));

    diff_generator.process_events(events.cbegin() + 6, events.cbegin() + 11);
    EXPECT_TRUE(diff_generator.generate(10, event_frame));
    EXPECT_EQ(event_frame.get_data().size(), 1 * 1);
    EXPECT_EQ(event_frame.get_data()[0], 10);

    diff_generator.process_events(events.cbegin() + 11, events.cbegin() + 20);
    EXPECT_FALSE(diff_generator.generate(19, event_frame));

    diff_generator.process_events(events.cbegin() + 20, events.cbegin() + 21);
    EXPECT_TRUE(diff_generator.generate(20, event_frame));
    EXPECT_EQ(event_frame.get_data().size(), 1 * 1);
    EXPECT_EQ(event_frame.get_data()[0], 10);
}
