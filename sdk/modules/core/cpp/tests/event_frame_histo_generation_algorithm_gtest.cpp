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

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/core/algorithms/event_frame_histo_generation_algorithm.h"
#include "metavision/sdk/core/utils/raw_event_frame_converter.h"

using namespace Metavision;
using InputIt = std::vector<EventCD>::const_iterator;

TEST(EventFrameHistoGenerationAlgorithm_GTest, nominal) {
    // GIVEN a 3x2 toy event stream
    const unsigned int width = 3, height = 2;
    std::vector<EventCD> events = {EventCD(0, 0, 0, 1), EventCD(2, 0, 0, 1), EventCD(0, 1, 1, 1), EventCD(2, 1, 1, 1),
                                   EventCD(0, 0, 1, 2), EventCD(2, 0, 1, 2), EventCD(0, 1, 1, 2), EventCD(2, 1, 0, 2),
                                   EventCD(0, 0, 0, 3), EventCD(2, 0, 1, 3), EventCD(0, 1, 0, 3), EventCD(2, 1, 0, 3)};
    // GIVEN a EventFrameHistoGenerationAlgorithm instance
    const unsigned int bit_size_neg = 4, bit_size_pos = 4;
    const bool packed = false;
    EventFrameHistoGenerationAlgorithm<InputIt> histo_generator(width, height, bit_size_neg, bit_size_pos, packed);

    // WHEN we process the event stream and generate the event frame
    histo_generator.process_events(events.cbegin(), events.cend());
    RawEventFrameHisto event_frame;
    histo_generator.generate(event_frame);

    // THEN the generated event frame matches the expectations
    const auto &histo_data = event_frame.get_data();
    EXPECT_EQ(histo_data.size(), 3 * 2 * 2);
    EXPECT_EQ(histo_data[0 + 2 * (0 + 3 * 0)], 2);
    EXPECT_EQ(histo_data[1 + 2 * (0 + 3 * 0)], 1);
    EXPECT_EQ(histo_data[0 + 2 * (1 + 3 * 0)], 0);
    EXPECT_EQ(histo_data[1 + 2 * (1 + 3 * 0)], 0);
    EXPECT_EQ(histo_data[0 + 2 * (2 + 3 * 0)], 1);
    EXPECT_EQ(histo_data[1 + 2 * (2 + 3 * 0)], 2);
    EXPECT_EQ(histo_data[0 + 2 * (0 + 3 * 1)], 1);
    EXPECT_EQ(histo_data[1 + 2 * (0 + 3 * 1)], 2);
    EXPECT_EQ(histo_data[0 + 2 * (1 + 3 * 1)], 0);
    EXPECT_EQ(histo_data[1 + 2 * (1 + 3 * 1)], 0);
    EXPECT_EQ(histo_data[0 + 2 * (2 + 3 * 1)], 2);
    EXPECT_EQ(histo_data[1 + 2 * (2 + 3 * 1)], 1);
}

TEST(EventFrameHistoGenerationAlgorithm_GTest, saturation) {
    // GIVEN a 1x1 toy event stream with 20 negative events and 20 positive events
    const unsigned int width = 1, height = 1;
    std::vector<EventCD> events;
    timestamp t = 1;
    for (int i = 0; i < 20; ++i) {
        events.emplace_back(0, 0, 0, t++);
    }
    for (int i = 0; i < 20; ++i) {
        events.emplace_back(0, 0, 1, t++);
    }
    // GIVEN a EventFrameHistoGenerationAlgorithm instance
    const unsigned int bit_size_neg = 4, bit_size_pos = 4;
    const bool packed = false;
    EventFrameHistoGenerationAlgorithm<InputIt> histo_generator(width, height, bit_size_neg, bit_size_pos, packed);

    // WHEN we process the event stream and generate the event frame
    histo_generator.process_events(events.cbegin(), events.cend());
    RawEventFrameHisto event_frame;
    histo_generator.generate(event_frame);

    // THEN the generated event frame matches the expectations
    const auto &histo_data = event_frame.get_data();
    EXPECT_EQ(histo_data.size(), 1 * 1 * 2);
    EXPECT_EQ(histo_data[0 + 2 * (0 + 3 * 0)], 15);
    EXPECT_EQ(histo_data[1 + 2 * (0 + 3 * 0)], 15);
}

TEST(EventFrameHistoGenerationAlgorithm_GTest, packed) {
    // GIVEN a 3x2 toy event stream
    const unsigned int width = 3, height = 2;
    std::vector<EventCD> events = {EventCD(0, 0, 0, 1), EventCD(2, 0, 0, 1), EventCD(0, 1, 1, 1), EventCD(2, 1, 1, 1),
                                   EventCD(0, 0, 1, 2), EventCD(2, 0, 1, 2), EventCD(0, 1, 1, 2), EventCD(2, 1, 0, 2),
                                   EventCD(0, 0, 0, 3), EventCD(2, 0, 1, 3), EventCD(0, 1, 0, 3), EventCD(2, 1, 0, 3)};
    // GIVEN a EventFrameHistoGenerationAlgorithm instance
    const unsigned int bit_size_neg = 4, bit_size_pos = 4;
    const bool packed = true;
    EventFrameHistoGenerationAlgorithm<InputIt> histo_generator(width, height, bit_size_neg, bit_size_pos, packed);

    // WHEN we process the event stream and generate the event frame
    histo_generator.process_events(events.cbegin(), events.cend());
    RawEventFrameHisto event_frame;
    histo_generator.generate(event_frame);

    // THEN the generated event frame matches the expectations
    const auto &histo_data = event_frame.get_data();
    EXPECT_EQ(histo_data.size(), 3 * 2);
    EXPECT_EQ(histo_data[0 + 3 * 0], (1 << 4) | (2));
    EXPECT_EQ(histo_data[1 + 3 * 0], 0);
    EXPECT_EQ(histo_data[2 + 3 * 0], (2 << 4) | (1));
    EXPECT_EQ(histo_data[0 + 3 * 1], (2 << 4) | (1));
    EXPECT_EQ(histo_data[1 + 3 * 1], 0);
    EXPECT_EQ(histo_data[2 + 3 * 1], (1 << 4) | (2));
}

TEST(EventFrameHistoGenerationAlgorithm_GTest, compatibility_with_RawEventFrameConverter) {
    // GIVEN a 3x2 toy event stream
    const unsigned int width = 3, height = 2;
    std::vector<EventCD> events = {EventCD(0, 0, 0, 1), EventCD(2, 0, 0, 1), EventCD(0, 1, 1, 1), EventCD(2, 1, 1, 1),
                                   EventCD(0, 0, 1, 2), EventCD(2, 0, 1, 2), EventCD(0, 1, 1, 2), EventCD(2, 1, 0, 2),
                                   EventCD(0, 0, 0, 3), EventCD(2, 0, 1, 3), EventCD(0, 1, 0, 3), EventCD(2, 1, 0, 3)};
    // GIVEN a EventFrameHistoGenerationAlgorithm instance
    const unsigned int bit_size_neg = 4, bit_size_pos = 4;
    const bool packed = true;
    EventFrameHistoGenerationAlgorithm<InputIt> histo_generator(width, height, bit_size_neg, bit_size_pos, packed);
    // GIVEN a RawEventFrameConverter instance
    RawEventFrameConverter converter(2, 3, 2, HistogramFormat::HWC);

    // WHEN we process the event stream, generate the RawEventFrameHisto and convert it to EventFrameHisto
    histo_generator.process_events(events.cbegin(), events.cend());
    RawEventFrameHisto event_frame_raw;
    histo_generator.generate(event_frame_raw);
    auto event_frame_converted = converter.convert<float>(event_frame_raw);

    // THEN the final event frame matches the expectations
    EXPECT_EQ(event_frame_converted->get_data().size(), 3 * 2 * 2);
    EXPECT_EQ((*event_frame_converted)(0, 0, HistogramChannel::NEGATIVE), 2.f);
    EXPECT_EQ((*event_frame_converted)(0, 0, HistogramChannel::POSITIVE), 1.f);
    EXPECT_EQ((*event_frame_converted)(1, 0, HistogramChannel::NEGATIVE), 0.f);
    EXPECT_EQ((*event_frame_converted)(1, 0, HistogramChannel::POSITIVE), 0.f);
    EXPECT_EQ((*event_frame_converted)(2, 0, HistogramChannel::NEGATIVE), 1.f);
    EXPECT_EQ((*event_frame_converted)(2, 0, HistogramChannel::POSITIVE), 2.f);
    EXPECT_EQ((*event_frame_converted)(0, 1, HistogramChannel::NEGATIVE), 1.f);
    EXPECT_EQ((*event_frame_converted)(0, 1, HistogramChannel::POSITIVE), 2.f);
    EXPECT_EQ((*event_frame_converted)(1, 1, HistogramChannel::NEGATIVE), 0.f);
    EXPECT_EQ((*event_frame_converted)(1, 1, HistogramChannel::POSITIVE), 0.f);
    EXPECT_EQ((*event_frame_converted)(2, 1, HistogramChannel::NEGATIVE), 2.f);
    EXPECT_EQ((*event_frame_converted)(2, 1, HistogramChannel::POSITIVE), 1.f);
}

TEST(EventFrameHistoGenerationAlgorithm_GTest, saturation_low_bit_sizes) {
    // GIVEN a 1x1 toy event stream with 20 negative events and 20 positive events
    const unsigned int width = 1, height = 1;
    std::vector<EventCD> events;
    timestamp t = 1;
    for (int i = 0; i < 5; ++i) {
        events.emplace_back(0, 0, 0, t++);
    }
    for (int i = 0; i < 5; ++i) {
        events.emplace_back(0, 0, 1, t++);
    }
    // GIVEN a EventFrameHistoGenerationAlgorithm instance
    const unsigned int bit_size_neg = 2, bit_size_pos = 2;
    const bool packed = false;
    EventFrameHistoGenerationAlgorithm<InputIt> histo_generator(width, height, bit_size_neg, bit_size_pos, packed);

    // WHEN we process the event stream and generate the event frame
    histo_generator.process_events(events.cbegin(), events.cend());
    RawEventFrameHisto event_frame;
    histo_generator.generate(event_frame);

    // THEN the generated event frame matches the expectations
    const auto &histo_data = event_frame.get_data();
    EXPECT_EQ(histo_data.size(), 1 * 1 * 2);
    EXPECT_EQ(histo_data[0 + 2 * (0 + 3 * 0)], 3);
    EXPECT_EQ(histo_data[1 + 2 * (0 + 3 * 0)], 3);
}

TEST(EventFrameHistoGenerationAlgorithm_GTest, lowerbound_generation_period) {
    // GIVEN a 1x1 toy event stream with 21 negative events
    const unsigned int width = 1, height = 1;
    std::vector<EventCD> events;
    for (int t = 0; t < 21; ++t) {
        events.emplace_back(0, 0, 0, 1 + t);
    }
    // GIVEN a EventFrameHistoGenerationAlgorithm instance
    const unsigned int bit_size_neg = 4, bit_size_pos = 4;
    const bool packed                               = false;
    const timestamp lowerbound_generation_period_us = 10;
    EventFrameHistoGenerationAlgorithm<InputIt> histo_generator(width, height, bit_size_neg, bit_size_pos, packed,
                                                                lowerbound_generation_period_us);

    // WHEN we process the event stream
    // THEN we can have feedback if the generation frequency is too high
    RawEventFrameHisto event_frame;
    histo_generator.process_events(events.cbegin(), events.cbegin() + 1);
    EXPECT_TRUE(histo_generator.generate(0, event_frame));
    EXPECT_EQ(event_frame.get_data().size(), 1 * 1 * 2);
    EXPECT_EQ(event_frame.get_data()[0], 1);

    histo_generator.process_events(events.cbegin() + 1, events.cbegin() + 6);
    EXPECT_FALSE(histo_generator.generate(5, event_frame));

    histo_generator.process_events(events.cbegin() + 6, events.cbegin() + 11);
    EXPECT_TRUE(histo_generator.generate(10, event_frame));
    EXPECT_EQ(event_frame.get_data().size(), 1 * 1 * 2);
    EXPECT_EQ(event_frame.get_data()[0], 10);

    histo_generator.process_events(events.cbegin() + 11, events.cbegin() + 20);
    EXPECT_FALSE(histo_generator.generate(19, event_frame));

    histo_generator.process_events(events.cbegin() + 20, events.cbegin() + 21);
    EXPECT_TRUE(histo_generator.generate(20, event_frame));
    EXPECT_EQ(event_frame.get_data().size(), 1 * 1 * 2);
    EXPECT_EQ(event_frame.get_data()[0], 10);
}
