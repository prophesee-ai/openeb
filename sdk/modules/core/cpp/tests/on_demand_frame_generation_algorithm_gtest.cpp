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

#include "metavision/sdk/core/algorithms/on_demand_frame_generation_algorithm.h"
#include "metavision/sdk/base/events/event_cd.h"

using namespace Metavision;

class OnDemandFrameGenerationAlgorithm_GTest : public ::testing::Test {
public:
    OnDemandFrameGenerationAlgorithm_GTest() {}

    virtual ~OnDemandFrameGenerationAlgorithm_GTest() {}

protected:
    virtual void SetUp() override {}

    virtual void TearDown() override {}
};

namespace {
cv::Vec3b bg_color  = OnDemandFrameGenerationAlgorithm::bg_color_default();
cv::Vec3b on_color  = OnDemandFrameGenerationAlgorithm::on_color_default();
cv::Vec3b off_color = OnDemandFrameGenerationAlgorithm::off_color_default();
} // namespace

TEST(OnDemandFrameGenerationAlgorithm_GTest, no_events_processed) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp accumulation_time_us = 10000;
    const bool colored                   = true;

    OnDemandFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_colors(bg_color, on_color, off_color, colored);

    // GIVEN no event to process and a frame generation algorithm
    cv::Mat generated;
    const timestamp generated_ts = 123456;

    // WHEN no events are given to processing...
    frame_generation.generate(generated_ts, generated);

    // THEN a frame with background only is generated
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    ASSERT_EQ(CV_8UC3, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), generated.begin<cv::Vec3b>()));
}

TEST(OnDemandFrameGenerationAlgorithm_GTest, get_dimension) {
    // GIVEN two frame generation algorithms: colored and in grayscale
    const int sensor_width              = 10;
    const int sensor_height             = 40;
    const uint32_t accumulation_time_us = 1000;

    OnDemandFrameGenerationAlgorithm colored_frame_gen(sensor_width, sensor_height);
    colored_frame_gen.set_accumulation_time_us(accumulation_time_us);
    colored_frame_gen.set_colors(bg_color, on_color, off_color, true);

    OnDemandFrameGenerationAlgorithm gray_frame_gen(sensor_width, sensor_height);
    gray_frame_gen.set_accumulation_time_us(accumulation_time_us);
    gray_frame_gen.set_colors(bg_color, on_color, off_color, false);

    // WHEN no events are given to processing and two frames are generated
    cv::Mat colored_generated, gray_generated;
    const timestamp generated_ts = 123456;
    colored_frame_gen.generate(generated_ts, colored_generated);
    gray_frame_gen.generate(generated_ts, gray_generated);

    // THEN the get_dimension method is consistent with the output image type
    uint32_t height, width, channels;
    colored_frame_gen.get_dimension(height, width, channels);
    ASSERT_EQ(sensor_height, height);
    ASSERT_EQ(sensor_width, width);
    ASSERT_EQ(3, channels);
    ASSERT_EQ(CV_8UC3, colored_generated.type());

    gray_frame_gen.get_dimension(height, width, channels);
    ASSERT_EQ(sensor_height, height);
    ASSERT_EQ(sensor_width, width);
    ASSERT_EQ(1, channels);
    ASSERT_EQ(CV_8UC1, gray_generated.type());
}

TEST(OnDemandFrameGenerationAlgorithm_GTest, default_parameters) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const timestamp accumulation_time_us = 10000;

    OnDemandFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);

    // GIVEN the following events in the timeslice [0, period_us[ and default parameters set
    std::vector<EventCD> events{{EventCD{5, 1, 0, period_us - accumulation_time_us - 30},
                                 EventCD{5, 5, 0, period_us - accumulation_time_us},
                                 EventCD{5, 8, 1, period_us - accumulation_time_us + 50}, EventCD{5, 9, 0, period_us}}};

    cv::Mat generated;
    timestamp generated_ts = period_us;

    // WHEN we process the events...
    frame_generation.process_events(events.cbegin(), events.cend());
    frame_generation.generate(generated_ts, generated);

    // THEN we generate a frame that holds all events, the timestamp being the last event's one
    // Explanations: the accumulation time has not been so all events are used
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(1, 5) = off_color;
    expected_frame.at<cv::Vec3b>(5, 5) = off_color;
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;
    expected_frame.at<cv::Vec3b>(9, 5) = off_color;

    ASSERT_EQ(CV_8UC3, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), generated.begin<cv::Vec3b>()));

    // WHEN we regenerate a frame
    frame_generation.generate(generated_ts, generated);

    // THEN the frame is only made of background as it must have been reset: no accumulation time set means events
    // are used once.
    expected_frame = cv::Mat(sensor_height, sensor_width, CV_8UC3, bg_color);
    ASSERT_EQ(CV_8UC3, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), generated.begin<cv::Vec3b>()));

    // WHEN we process the same events again
    frame_generation.process_events(events.cbegin(), events.cend());
    frame_generation.generate(generated_ts, generated);

    // THEN the frame is only made of background: no accumulation time set means events are used once and the class
    // must be reset in order to reprocess events from the past
    ASSERT_EQ(CV_8UC3, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), generated.begin<cv::Vec3b>()));

    // WHEN we process the same events but shifted in time by period_us
    events = std::vector<EventCD>{{EventCD{5, 1, 0, 2 * period_us - accumulation_time_us - 30},
                                   EventCD{5, 5, 0, 2 * period_us - accumulation_time_us},
                                   EventCD{5, 8, 1, 2 * period_us - accumulation_time_us + 50},
                                   EventCD{5, 9, 0, 2 * period_us}}};
    frame_generation.process_events(events.cbegin(), events.cend());
    generated_ts = 2 * period_us;
    frame_generation.generate(generated_ts, generated);

    // THEN we generate a frame that holds all events
    expected_frame.at<cv::Vec3b>(1, 5) = off_color;
    expected_frame.at<cv::Vec3b>(5, 5) = off_color;
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;
    expected_frame.at<cv::Vec3b>(9, 5) = off_color;

    ASSERT_EQ(CV_8UC3, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), generated.begin<cv::Vec3b>()));
}

TEST(OnDemandFrameGenerationAlgorithm_GTest, accumulation_time_only) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const timestamp accumulation_time_us = 10000;

    OnDemandFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);

    // GIVEN the following events in the timeslice [0, period_us] and only accumulation time parameter set (next
    // frame's timestamp is unknown)
    std::vector<EventCD> events{{EventCD{5, 1, 0, period_us - accumulation_time_us},
                                 EventCD{5, 5, 0, period_us - accumulation_time_us + 50}, EventCD{5, 8, 1, period_us}}};

    cv::Mat generated;
    const timestamp generated_ts = period_us;

    // WHEN we process the events...
    frame_generation.process_events(events.cbegin(), events.cend());
    frame_generation.generate(generated_ts, generated);

    // THEN we generate a frame that holds only the event in the interval ]period - acc, period], the timestamp
    // being the last event's one

    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;
    expected_frame.at<cv::Vec3b>(5, 5) = off_color;

    ASSERT_EQ(CV_8UC3, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), generated.begin<cv::Vec3b>()));
}

TEST(OnDemandFrameGenerationAlgorithm_GTest, nominal_with_ts_0_for_initialization) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const timestamp accumulation_time_us = 10000;
    const bool colored                   = true;

    OnDemandFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_colors(bg_color, on_color, off_color, colored);

    // GIVEN the following events in the timeslice [0, period_us[ and the above parameters
    std::vector<EventCD> events{{EventCD{5, 1, 0, period_us - accumulation_time_us - 30},
                                 EventCD{5, 5, 1, period_us - accumulation_time_us},
                                 EventCD{5, 8, 0, period_us - accumulation_time_us + 50}, EventCD{5, 9, 0, period_us}}};

    cv::Mat generated;
    const timestamp generated_ts = period_us;

    // WHEN we process the events...
    frame_generation.process_events(events.cbegin(), events.cend());
    frame_generation.generate(generated_ts, generated);

    // THEN we generate a frame that holds events in the timeslice ]period_us - accumulation_time_us, period_us] ,
    // and background otherwise
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(8, 5) = off_color;
    expected_frame.at<cv::Vec3b>(9, 5) = off_color;

    ASSERT_EQ(CV_8UC3, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), generated.begin<cv::Vec3b>()));
}

TEST(OnDemandFrameGenerationAlgorithm_GTest, nominal_with_offset_overflow) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const timestamp accumulation_time_us = 10000;
    const bool colored                   = true;

    OnDemandFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_colors(bg_color, on_color, off_color, colored);

    // GIVEN the following events in the timeslice [tbase, tbase + period_us[ and [tbase + period_us, tbase +
    // 2*period_us[ and the above parameters. The timeslice is chosen to be above the max value of int32 to test the
    // internal overflow state machine
    const timestamp offset_overflow_time_base =
        static_cast<timestamp>(std::numeric_limits<std::int32_t>::max()) + 150459;
    const timestamp expected_first_timeslice = period_us * (offset_overflow_time_base / period_us);
    std::vector<EventCD> events{{EventCD{5, 1, 0, expected_first_timeslice + period_us - accumulation_time_us - 30},
                                 EventCD{5, 5, 0, expected_first_timeslice + period_us - accumulation_time_us},
                                 EventCD{5, 8, 1, expected_first_timeslice + period_us - accumulation_time_us + 30},
                                 EventCD{5, 9, 0, expected_first_timeslice + period_us}}};

    cv::Mat generated;
    const timestamp generated_ts = expected_first_timeslice + period_us;

    // WHEN we process the events...
    frame_generation.process_events(events.cbegin(), events.cend());
    frame_generation.generate(generated_ts, generated);

    // THEN we generate a frame that holds events in the timeslice ]period_us - accumulation_time_us, period_us],
    // and background otherwise
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;
    expected_frame.at<cv::Vec3b>(9, 5) = off_color;

    ASSERT_EQ(CV_8UC3, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), generated.begin<cv::Vec3b>()));
}

TEST(OnDemandFrameGenerationAlgorithm_GTest, reset) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const timestamp accumulation_time_us = 10000;
    const bool colored                   = true;

    OnDemandFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_colors(bg_color, on_color, off_color, colored);

    // GIVEN the following events in the timeslice [0, period_us[ and the above parameters
    std::vector<EventCD> events{{EventCD{5, 1, 0, period_us - accumulation_time_us - 30},
                                 EventCD{5, 5, 0, period_us - accumulation_time_us},
                                 EventCD{5, 8, 1, period_us - accumulation_time_us + 50}, EventCD{5, 9, 0, period_us}}};

    cv::Mat generated;
    timestamp generated_ts = period_us;

    // WHEN we process the events...
    frame_generation.process_events(events.cbegin(), events.cend());
    frame_generation.generate(generated_ts, generated);

    // THEN we generate a frame that holds events in the timeslice ]period_us - accumulation_time_us, period_us] ,
    // and background otherwise
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;
    expected_frame.at<cv::Vec3b>(9, 5) = off_color;

    ASSERT_EQ(CV_8UC3, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), generated.begin<cv::Vec3b>()));

    // WHEN we reset and generate a frame
    frame_generation.reset();

    // THEN the generated frame is made only of background and frame ts is reset to 0
    generated_ts = 0;
    frame_generation.generate(generated_ts, generated);

    expected_frame = cv::Mat(sensor_height, sensor_width, CV_8UC3, bg_color);

    ASSERT_EQ(CV_8UC3, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), generated.begin<cv::Vec3b>()));
}

TEST(OnDemandFrameGenerationAlgorithm_GTest, generate_frame_in_past) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const timestamp accumulation_time_us = 10000;
    const bool colored                   = true;

    OnDemandFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_colors(bg_color, on_color, off_color, colored);

    // GIVEN the following events in the timeslice [0, period_us[ and the above parameters
    std::vector<EventCD> events{{EventCD{5, 1, 0, 5 * period_us - accumulation_time_us - 30},
                                 EventCD{5, 5, 0, 5 * period_us - accumulation_time_us},
                                 EventCD{5, 8, 1, 5 * period_us - accumulation_time_us + 50},
                                 EventCD{5, 9, 0, 5 * period_us}}};

    cv::Mat generated;
    timestamp generated_ts = 5 * period_us;

    // WHEN we process the events...
    frame_generation.process_events(events.cbegin(), events.cend());
    frame_generation.generate(generated_ts, generated);

    // THEN we generate a frame that holds events in the timeslice ]5*period_us - accumulation_time_us, 5*period_us]
    // , and background otherwise
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;
    expected_frame.at<cv::Vec3b>(9, 5) = off_color;

    ASSERT_EQ(CV_8UC3, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), generated.begin<cv::Vec3b>()));

    // WHEN we generate a frame in the past with events at same position
    frame_generation.reset();
    events = {{EventCD{5, 1, 0, period_us - accumulation_time_us - 30},
               EventCD{3, 2, 0, period_us - accumulation_time_us},
               EventCD{7, 4, 1, period_us - accumulation_time_us + 50}, EventCD{6, 5, 0, period_us}}};

    generated_ts = period_us;
    frame_generation.process_events(events.cbegin(), events.cend());
    frame_generation.generate(generated_ts, generated);

    // THEN we generate a frame that holds events in the timeslice [period_us - accumulation_time_us, period_us[ ,
    // and background otherwise i.e. events in the futur are no longer in the history
    expected_frame                     = cv::Mat(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(4, 7) = on_color;
    expected_frame.at<cv::Vec3b>(5, 6) = off_color;

    ASSERT_EQ(CV_8UC3, generated.type());
    ASSERT_EQ(period_us, generated_ts);
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), generated.begin<cv::Vec3b>()));
}

TEST(OnDemandFrameGenerationAlgorithm_GTest, change_accumulation_time) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const timestamp accumulation_time_us = 10000;
    const bool colored                   = true;

    OnDemandFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_colors(bg_color, on_color, off_color, colored);

    // GIVEN the following events in the timeslice [0, period_us[ and the above parameters
    std::vector<EventCD> events{{EventCD{5, 1, 0, period_us - accumulation_time_us - 30},
                                 EventCD{5, 5, 1, period_us - accumulation_time_us},
                                 EventCD{5, 3, 0, period_us - accumulation_time_us + 2},
                                 EventCD{5, 8, 1, period_us - accumulation_time_us + 50}, EventCD{5, 9, 0, period_us}}};

    cv::Mat generated;
    timestamp generated_ts = period_us;

    // WHEN we process the events...
    frame_generation.process_events(events.cbegin(), events.cend());
    frame_generation.generate(generated_ts, generated);

    // THEN we generate a frame that holds events in the timeslice ]period_us - accumulation_time_us, period_us] ,
    // and background otherwise
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(3, 5) = off_color;
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;
    expected_frame.at<cv::Vec3b>(9, 5) = off_color;

    ASSERT_EQ(CV_8UC3, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), generated.begin<cv::Vec3b>()));

    // WHEN we change the accumulation time
    frame_generation.set_accumulation_time_us(accumulation_time_us - 10);
    frame_generation.generate(generated_ts, generated);

    // THEN the frame only holds the last two events
    expected_frame                     = cv::Mat(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;
    expected_frame.at<cv::Vec3b>(9, 5) = off_color;

    ASSERT_EQ(CV_8UC3, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), generated.begin<cv::Vec3b>()));

    // WHEN we use new events
    events = {{EventCD{5, 1, 0, 2 * period_us - accumulation_time_us - 30},
               EventCD{5, 5, 1, 2 * period_us - accumulation_time_us},
               EventCD{5, 3, 0, 2 * period_us - accumulation_time_us + 2},
               EventCD{5, 8, 1, 2 * period_us - accumulation_time_us + 50}, EventCD{5, 9, 0, 2 * period_us}}};

    frame_generation.process_events(events.cbegin(), events.cend());
    generated_ts = 2 * period_us;
    frame_generation.generate(generated_ts, generated);

    // THEN the frame only holds the last two events as the new accumulation remained
    expected_frame                     = cv::Mat(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;
    expected_frame.at<cv::Vec3b>(9, 5) = off_color;

    ASSERT_EQ(CV_8UC3, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), generated.begin<cv::Vec3b>()));
}

TEST(OnDemandFrameGenerationAlgorithm_GTest, colors) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const timestamp accumulation_time_us = 10000;
    bool colored;

    colored = true;
    OnDemandFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_colors(bg_color, on_color, off_color, colored);

    // GIVEN the following events in the timeslice [0, period_us[ and the above parameters
    std::vector<EventCD> events{{EventCD{5, 1, 0, period_us - accumulation_time_us - 30},
                                 EventCD{5, 5, 0, period_us - accumulation_time_us},
                                 EventCD{5, 8, 1, period_us - accumulation_time_us + 50}, EventCD{5, 9, 0, period_us}}};

    cv::Mat generated;
    timestamp generated_ts = period_us;

    // WHEN we process the events...
    frame_generation.process_events(events.cbegin(), events.cend());
    frame_generation.generate(generated_ts, generated);

    // THEN we generate a frame that holds events in the timeslice ]period_us - accumulation_time_us, period_us] ,
    // and background otherwise
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;
    expected_frame.at<cv::Vec3b>(9, 5) = off_color;

    ASSERT_EQ(CV_8UC3, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), generated.begin<cv::Vec3b>()));

    // WHEN changing colors to gray scale and regenerating the frame
    colored = false;
    frame_generation.set_colors(bg_color, on_color, off_color, colored);
    frame_generation.generate(generated_ts, generated);

    // THEN the frame has now gray scale type
    expected_frame                   = cv::Mat(sensor_height, sensor_width, CV_8UC1, bg_color[0]);
    expected_frame.at<uint8_t>(8, 5) = on_color[0];
    expected_frame.at<uint8_t>(9, 5) = off_color[0];

    ASSERT_EQ(CV_8UC1, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<uint8_t>(), expected_frame.end<uint8_t>(), generated.begin<uint8_t>()));

    // WHEN we use new events
    events = {{EventCD{5, 1, 0, 2 * period_us - accumulation_time_us - 30},
               EventCD{5, 5, 0, 2 * period_us - accumulation_time_us},
               EventCD{5, 8, 1, 2 * period_us - accumulation_time_us + 50}, EventCD{5, 9, 0, 2 * period_us}}};

    frame_generation.process_events(events.cbegin(), events.cend());
    generated_ts = 2 * period_us;
    frame_generation.generate(generated_ts, generated);

    // THEN the frame is still generated in gray scale
    expected_frame                   = cv::Mat(sensor_height, sensor_width, CV_8UC1, bg_color[0]);
    expected_frame.at<uint8_t>(8, 5) = on_color[0];
    expected_frame.at<uint8_t>(9, 5) = off_color[0];

    ASSERT_EQ(CV_8UC1, generated.type());
    ASSERT_EQ(expected_frame.size(), generated.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<uint8_t>(), expected_frame.end<uint8_t>(), generated.begin<uint8_t>()));
}

TEST(OnDemandFrameGenerationAlgorithm_GTest, delay_frame_generation) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const timestamp accumulation_time_us = 10000;

    const int n_timeslices       = 4;
    const int n_events_per_slice = 5;

    OnDemandFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);

    // GIVEN following events spread over 4 timeslices
    auto get_pix_xy = [](const int k, const int i, int &x, int &y, int &p) {
        p = i % 2;
        x = 5 + k - i;
        y = i;
    };
    std::vector<EventCD> events;
    for (int k = 0; k < n_timeslices; k++)
        for (int i = 0; i < n_events_per_slice; i++) {
            int x, y, p;
            get_pix_xy(k, i, x, y, p);
            events.emplace_back(EventCD(x, y, p, (1 + k) * period_us - accumulation_time_us + 10 * i + 1));
        }

    std::vector<cv::Mat> generated_frames;
    timestamp generated_ts;

    // WHEN we first process the events and then generate frames at increasing timestamps...
    frame_generation.process_events(events.cbegin(), events.cend());
    for (int k = 0; k < n_timeslices; k++) {
        generated_ts = (1 + k) * period_us;
        generated_frames.emplace_back();
        frame_generation.generate(generated_ts, generated_frames.back());
    }

    // THEN we generate the expected frames
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3);
    auto it_generated = generated_frames.cbegin();
    for (int k = 0; k < n_timeslices; k++) {
        ASSERT_TRUE(it_generated != generated_frames.cend());
        expected_frame.setTo(bg_color);
        for (int i = 0; i < n_events_per_slice; i++) {
            int x, y, p;
            get_pix_xy(k, i, x, y, p);
            expected_frame.at<cv::Vec3b>(y, x) = (p == 1 ? on_color : off_color);
        }

        ASSERT_EQ(CV_8UC3, it_generated->type());
        ASSERT_EQ(expected_frame.size(), it_generated->size());
        ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                               it_generated->begin<cv::Vec3b>()));
        it_generated++;
    }
}

TEST(OnDemandFrameGenerationAlgorithm_GTest, delay_overlapping_frame_generation) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 40;
    const timestamp accumulation_time_us = 100;
    const timestamp ts_first_process     = 300;

    OnDemandFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);

    // GIVEN events spread over 4 timeslices, the timestamps of which are as follows:
    // Timestamps :      210, 230, 250,     290, 310, 330,           390, 410
    //  - ]200, 300]:   |  x   x    x        x |
    //  - ]240, 340]:             | x        x    x    x |
    //  - ]280, 380]:                      | x    x    x       |
    //  - ]320, 420]:                               |  x               x    x |
    // clang-format off
        std::vector<EventCD> events{
            EventCD(8, 1, 0, 210),
            EventCD(8, 2, 1, 230),
            EventCD(8, 3, 0, 250),
            EventCD(8, 4, 1, 290),
            EventCD(8, 5, 0, 310),
            EventCD(8, 6, 1, 330),
            EventCD(8, 7, 0, 390),
            EventCD(8, 8, 1, 410),
        };
    // clang-format on
    std::vector<std::vector<EventCD>> events_gt{
        {EventCD(8, 1, 0, 210), EventCD(8, 2, 1, 230), EventCD(8, 3, 0, 250), EventCD(8, 4, 1, 290)},
        {EventCD(8, 3, 0, 250), EventCD(8, 4, 1, 290), EventCD(8, 5, 0, 310), EventCD(8, 6, 1, 330)},
        {EventCD(8, 4, 1, 290), EventCD(8, 5, 0, 310), EventCD(8, 6, 1, 330)},
        {EventCD(8, 6, 1, 330), EventCD(8, 7, 0, 390), EventCD(8, 8, 1, 410)},
    };
    std::vector<cv::Mat> generated_frames;
    timestamp generated_ts;

    // WHEN we first process the events and then generate frames at increasing timestamps...
    frame_generation.process_events(events.cbegin(), events.cend());
    generated_ts = ts_first_process;
    for (int k = 0; k < 4; k++) {
        generated_frames.emplace_back();
        frame_generation.generate(generated_ts, generated_frames.back());
        generated_ts += period_us;
    }

    // THEN we generate the expected frames
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3);
    ASSERT_EQ(events_gt.size(), generated_frames.size());
    using SizeType = std::vector<std::vector<EventCD>>::size_type;
    for (SizeType i = 0; i < events_gt.size(); ++i) {
        expected_frame.setTo(bg_color);
        const auto &evs = events_gt[i];
        for (const auto &ev : evs)
            expected_frame.at<cv::Vec3b>(ev.y, ev.x) = (ev.p == 1 ? on_color : off_color);

        auto &generated_mat = generated_frames[i];
        ASSERT_EQ(CV_8UC3, generated_mat.type());
        ASSERT_EQ(expected_frame.size(), generated_mat.size());
        ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                               generated_mat.begin<cv::Vec3b>()));
    }
}