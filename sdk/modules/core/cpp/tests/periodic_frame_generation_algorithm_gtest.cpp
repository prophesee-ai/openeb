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

#include "metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h"
#include "metavision/sdk/base/events/event_cd.h"

using namespace Metavision;

class PeriodicFrameGenerationAlgorithm_GTest : public ::testing::Test {
public:
    PeriodicFrameGenerationAlgorithm_GTest() {}

    virtual ~PeriodicFrameGenerationAlgorithm_GTest() {}

protected:
    virtual void SetUp() override {}

    virtual void TearDown() override {}
};

struct FrameData {
    timestamp ts_us_;
    cv::Mat frame_;
};

// Prophesee Colors
namespace {
cv::Vec3b bg_color  = PeriodicFrameGenerationAlgorithm::bg_color_default();
cv::Vec3b on_color  = PeriodicFrameGenerationAlgorithm::on_color_default();
cv::Vec3b off_color = PeriodicFrameGenerationAlgorithm::off_color_default();
} // namespace

TEST(PeriodicFrameGenerationAlgorithm_GTest, nominal_with_ts_0_for_initialization) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const double fps                     = 1.e6 / period_us;
    const timestamp accumulation_time_us = 10000;
    const bool colored                   = true;
    PeriodicFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_colors(bg_color, on_color, off_color, colored);
    frame_generation.set_fps(fps);

    // GIVEN the following events in the time slice [0, period_us[ and [period_us, 2*period_us[ and the above parameters
    std::vector<EventCD> events{{EventCD{5, 1, 0, period_us - accumulation_time_us - 30},
                                 EventCD{5, 5, 0, period_us - accumulation_time_us},
                                 EventCD{5, 8, 1, period_us - accumulation_time_us + 50},
                                 EventCD{0, 0, 0, 2 * period_us - accumulation_time_us + 50}}};

    // WHEN we process the events...
    std::vector<FrameData> generated_frames;

    frame_generation.set_output_callback([&](timestamp ts, cv::Mat &frame) {
        generated_frames.push_back({ts, frame.clone()});
    });
    frame_generation.process_events(events.cbegin(), events.cend());

    // THEN we generate a frame that holds event data in the time slice [period_us - accumulation_time_us, period_us[ ,
    // and bg color otherwise
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;
    expected_frame.at<cv::Vec3b>(5, 5) = off_color;

    ASSERT_EQ(size_t(1), generated_frames.size());
    ASSERT_EQ(CV_8UC3, generated_frames.back().frame_.type());
    ASSERT_EQ(period_us, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames.back().frame_.begin<cv::Vec3b>()));

    // WHEN we force_generate, we generate a last frame with remaining processed events information -> a frame that
    // holds event in the time slice [2 * period_us - accumulation_time_us, 2 * period_us - accumulation_time_us + 50]
    frame_generation.force_generate();

    expected_frame.setTo(bg_color);
    expected_frame.at<cv::Vec3b>(0, 0) = off_color;

    ASSERT_EQ(size_t(2), generated_frames.size());
    ASSERT_EQ(CV_8UC3, generated_frames.back().frame_.type());
    ASSERT_EQ(events.back().t, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames.back().frame_.begin<cv::Vec3b>()));
}

TEST(PeriodicFrameGenerationAlgorithm_GTest, nominal_with_offset_overflow) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const double fps                     = 1.e6 / period_us;
    const timestamp accumulation_time_us = 10000;
    const bool colored                   = true;
    PeriodicFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_colors(bg_color, on_color, off_color, colored);
    frame_generation.set_fps(fps);

    // GIVEN the following events in the time slice [tbase, tbase + period_us[ and [tbase + period_us, tbase +
    // 2*period_us[ and the above parameters. The time slice is chosen to be above the max value of int32 to test the
    // internal overflow state machine
    const timestamp offset_overflow_time_base =
        static_cast<timestamp>(std::numeric_limits<std::int32_t>::max()) + 150459;
    const timestamp expected_first_timeslice = period_us * (offset_overflow_time_base / period_us);
    std::vector<EventCD> events{
        {EventCD{5, 1, 0, expected_first_timeslice + period_us - accumulation_time_us - 30},
         EventCD{5, 5, 0, expected_first_timeslice + period_us - accumulation_time_us},
         EventCD{5, 8, 1, expected_first_timeslice + period_us - accumulation_time_us + 30},
         EventCD{0, 0, 0, expected_first_timeslice + 2 * period_us - accumulation_time_us + 50}}};

    // WHEN we process the events...
    std::vector<FrameData> generated_frames;

    frame_generation.set_output_callback([&](timestamp ts, cv::Mat &frame) {
        generated_frames.push_back({ts, frame.clone()});
    });
    frame_generation.process_events(events.cbegin(), events.cend());

    // THEN we generate a frame that holds event data in the time slice [period_us - accumulation_time_us, period_us[ ,
    // and bg color otherwise
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;
    expected_frame.at<cv::Vec3b>(5, 5) = off_color;

    ASSERT_EQ(size_t(1), generated_frames.size());
    ASSERT_EQ(CV_8UC3, generated_frames.back().frame_.type());
    ASSERT_EQ(expected_first_timeslice + period_us, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames.back().frame_.begin<cv::Vec3b>()));

    // WHEN we force_generate, we generate a last frame with remaining processed events information -> a frame that
    // holds event in the time slice [2 * period_us - accumulation_time_us, 2 * period_us - accumulation_time_us + 50]
    frame_generation.force_generate();

    expected_frame.setTo(bg_color);
    expected_frame.at<cv::Vec3b>(0, 0) = off_color;

    ASSERT_EQ(size_t(2), generated_frames.size());
    ASSERT_EQ(CV_8UC3, generated_frames.back().frame_.type());
    ASSERT_EQ(events.back().t, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames.back().frame_.begin<cv::Vec3b>()));
}

TEST(PeriodicFrameGenerationAlgorithm_GTest, no_underflow_with_offset_overflow) {
    const int sensor_width                   = 10;
    const int sensor_height                  = 10;
    const timestamp period_us                = 100000;
    const double fps                         = 1.e6 / period_us;
    const std::uint32_t accumulation_time_us = 100000;
    const bool colored                       = true;
    PeriodicFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_colors(bg_color, on_color, off_color, colored);
    frame_generation.set_fps(fps);

    // GIVEN the following events and the above parameters. The event timestamps are chosen to potentially
    // lead to an underflow with 32-bits timestamp representations at pixel (0,0).
    const std::int32_t t32i_max_32i         = std::numeric_limits<std::int32_t>::max();
    const std::int32_t t32i_init            = 50;
    const std::int32_t t32i_after_underflow = 52; // t32i_before_underflow - t32i_max_32i;
    const timestamp t_max_32i               = t32i_max_32i;
    const timestamp t_init                  = t32i_init;
    const timestamp t_after_underflow       = t32i_after_underflow;
    const timestamp t_period_accumulating_underflowed_ev =
        period_us * ((t_after_underflow + 2 * t_max_32i) / period_us);
    std::vector<EventCD> events{{EventCD{0, 0, 0, t_init}, EventCD{0, 1, 0, t_max_32i + 1},
                                 EventCD{1, 0, 0, 2 * t_max_32i + 1},
                                 EventCD{2, 0, 0, t_period_accumulating_underflowed_ev + period_us}}};

    // WHEN we process the events
    FrameData last_generated_frame;
    frame_generation.set_output_callback([&](timestamp ts, cv::Mat &frame) {
        last_generated_frame.ts_us_ = ts;
        last_generated_frame.frame_ = frame.clone();
    });
    frame_generation.process_events(events.cbegin(), events.cend());

    // THEN the timestamp overflow mechanism correctly handles the potential underflow at pixel (0,0)
    // hence pixel (0,0) is not drawn in the generated frame
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(0, 1) = off_color;

    ASSERT_EQ(CV_8UC3, last_generated_frame.frame_.type());
    ASSERT_EQ(t_period_accumulating_underflowed_ev + period_us, last_generated_frame.ts_us_);
    ASSERT_EQ(expected_frame.size(), last_generated_frame.frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           last_generated_frame.frame_.begin<cv::Vec3b>()));
}

TEST(PeriodicFrameGenerationAlgorithm_GTest, continuity_preserved_with_offset_overflow) {
    const int sensor_width                   = 10;
    const int sensor_height                  = 10;
    const timestamp period_us                = 100000;
    const double fps                         = 1.e6 / period_us;
    const std::uint32_t accumulation_time_us = 200000;
    const bool colored                       = true;
    PeriodicFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_colors(bg_color, on_color, off_color, colored);
    frame_generation.set_fps(fps);

    // GIVEN the following events and the above parameters. The event timestamps are chosen in timeslices
    // overlapping the period before and after offset overflow, to ensure time continuity is preserved.
    const timestamp t_max_32i                      = std::numeric_limits<std::int32_t>::max();
    const timestamp t_first_overflowing_timeslice  = period_us * (t_max_32i / period_us) + period_us;
    const timestamp t_fourth_overflowing_timeslice = period_us * (4 * t_max_32i / period_us) + period_us;
    std::vector<EventCD> events{{EventCD{0, 0, 0, t_first_overflowing_timeslice - 2 * period_us + 10},
                                 EventCD{1, 0, 0, t_first_overflowing_timeslice - period_us + 10},
                                 EventCD{2, 0, 0, t_first_overflowing_timeslice + 10},
                                 EventCD{3, 0, 0, t_first_overflowing_timeslice + period_us},
                                 EventCD{0, 1, 0, t_fourth_overflowing_timeslice - 2 * period_us + 10},
                                 EventCD{1, 1, 0, t_fourth_overflowing_timeslice - period_us + 10},
                                 EventCD{2, 1, 0, t_fourth_overflowing_timeslice + 10},
                                 EventCD{3, 1, 0, t_fourth_overflowing_timeslice + period_us}}};

    // WHEN we process the events
    FrameData generated_frame_overflow1, generated_frame_overflow4;
    frame_generation.set_output_callback([&](timestamp ts, cv::Mat &frame) {
        if (ts == t_first_overflowing_timeslice + period_us) {
            generated_frame_overflow1.ts_us_ = ts;
            generated_frame_overflow1.frame_ = frame.clone();
        } else if (ts == t_fourth_overflowing_timeslice + period_us) {
            generated_frame_overflow4.ts_us_ = ts;
            generated_frame_overflow4.frame_ = frame.clone();
        }
    });
    frame_generation.process_events(events.cbegin(), events.cend());

    // THEN the timestamp overflow mechanism correctly preserves timestamp continuity upon overflow
    // hence pixels (0,0) (at 1st overflow) and (0,1) (at 4th overflow) are drawn in the generated frames.
    cv::Mat expected_frame_overflow1(sensor_height, sensor_width, CV_8UC3, bg_color);
    cv::Mat expected_frame_overflow4(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame_overflow1.at<cv::Vec3b>(0, 1) = off_color;
    expected_frame_overflow1.at<cv::Vec3b>(0, 2) = off_color;
    expected_frame_overflow4.at<cv::Vec3b>(1, 1) = off_color;
    expected_frame_overflow4.at<cv::Vec3b>(1, 2) = off_color;

    ASSERT_EQ(CV_8UC3, generated_frame_overflow1.frame_.type());
    ASSERT_EQ(expected_frame_overflow1.size(), generated_frame_overflow1.frame_.size());
    ASSERT_TRUE(std::equal(expected_frame_overflow1.begin<cv::Vec3b>(), expected_frame_overflow1.end<cv::Vec3b>(),
                           generated_frame_overflow1.frame_.begin<cv::Vec3b>()));
    ASSERT_EQ(CV_8UC3, generated_frame_overflow4.frame_.type());
    ASSERT_EQ(expected_frame_overflow4.size(), generated_frame_overflow4.frame_.size());
    ASSERT_TRUE(std::equal(expected_frame_overflow4.begin<cv::Vec3b>(), expected_frame_overflow4.end<cv::Vec3b>(),
                           generated_frame_overflow4.frame_.begin<cv::Vec3b>()));
}

TEST(PeriodicFrameGenerationAlgorithm_GTest, nominal_with_arbitrary_ts_for_initialization_reset) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const double fps                     = 1.e6 / period_us;
    const timestamp accumulation_time_us = 10000;
    const bool colored                   = true;
    PeriodicFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_colors(bg_color, on_color, off_color, colored);
    frame_generation.set_fps(fps);

    // GIVEN the following events in an arbitrary time slice [first_timeslice, first_timeslice + period_us[ and
    // [first_timeslice + period_us, first_timeslice + 2*period_us[, in other words, [1.200.000, 1.300.000[ and
    // [1.300.000, 1.400.000[, and the above parameters
    const timestamp arbitrary_time_base      = 1255341;
    const timestamp expected_first_timeslice = period_us * (arbitrary_time_base / period_us); // 1.200.000

    // clang-format off
    std::vector<EventCD> events{{
         EventCD{5, 1, 0, expected_first_timeslice + period_us - accumulation_time_us - 30},       // ts = 1.289.970
         EventCD{5, 5, 0, expected_first_timeslice + period_us - accumulation_time_us},            // ts = 1.290.000
         EventCD{5, 8, 1, expected_first_timeslice + period_us - accumulation_time_us + 30},       // ts = 1.290.030
         EventCD{0, 0, 0, expected_first_timeslice + 2 * period_us - accumulation_time_us + 50}}}; // ts = 1.390.050
    // clang-format on

    // WHEN we process the events...
    std::vector<FrameData> generated_frames;

    frame_generation.set_output_callback([&](timestamp ts, cv::Mat &frame) {
        generated_frames.push_back({ts, frame.clone()});
    });
    frame_generation.process_events(events.cbegin(), events.cend());

    // THEN we generate a frame that holds event data in the time slice [first_timeslice + period_us -
    // accumulation_time_us, first_timeslice + period_us[, in other words in [1.290.000, 1.300.000[
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(5, 5) = off_color;
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;

    ASSERT_EQ(size_t(1), generated_frames.size());
    ASSERT_EQ(CV_8UC3, generated_frames.back().frame_.type());
    ASSERT_EQ(expected_first_timeslice + period_us, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames.back().frame_.begin<cv::Vec3b>()));

    // WHEN we force_generate
    frame_generation.force_generate();

    // THEN we generate a last frame with remaining processed events information -> a frame that holds events
    // in the time slice [first_timeslice + 2 * period_us - accumulation_time_us, last event's timestamp ], in other
    // words in [1.390.000, 1.390.051[

    expected_frame.setTo(bg_color);
    expected_frame.at<cv::Vec3b>(0, 0) = off_color;

    ASSERT_EQ(size_t(2), generated_frames.size());
    ASSERT_EQ(CV_8UC3, generated_frames.back().frame_.type());
    ASSERT_EQ(events.back().t, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames.back().frame_.begin<cv::Vec3b>()));

    // WHEN
    // - we process a new event at timestamp ts = 1.400.000
    // - the frame generation algorithm is reset
    const EventCD ev{1, 1, 1, expected_first_timeslice + 3 * period_us - accumulation_time_us}; // ts = 1.490.000
    frame_generation.process_events(&ev, &ev + 1);
    frame_generation.force_generate();
    frame_generation.reset();

    // THEN Then process async is called and pending events are flushed.
    // The frame corresponds to the time slice
    // [last event's timestamp + 1 - accumulation_time_us, last event's timestamp ], in other words
    // [1.480.001, 1.490.001[, is generated
    expected_frame.setTo(bg_color);
    expected_frame.at<cv::Vec3b>(1, 1) = on_color;

    ASSERT_EQ(size_t(3), generated_frames.size());
    ASSERT_EQ(CV_8UC3, generated_frames.back().frame_.type());
    ASSERT_EQ(ev.t, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames.back().frame_.begin<cv::Vec3b>()));

    // WHEN we process the events after the reset
    frame_generation.process_events(events.cbegin(), events.cend());

    // THEN we generate the same frames we generated before
    expected_frame.setTo(bg_color);
    expected_frame.at<cv::Vec3b>(5, 5) = off_color;
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;

    ASSERT_EQ(size_t(4), generated_frames.size());
    ASSERT_EQ(CV_8UC3, generated_frames.back().frame_.type());
    ASSERT_EQ(expected_first_timeslice + period_us, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames.back().frame_.begin<cv::Vec3b>()));

    frame_generation.force_generate();

    expected_frame.setTo(bg_color);
    expected_frame.at<cv::Vec3b>(0, 0) = off_color;

    ASSERT_EQ(size_t(5), generated_frames.size());
    ASSERT_EQ(CV_8UC3, generated_frames.back().frame_.type());
    ASSERT_EQ(events.back().t, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames.back().frame_.begin<cv::Vec3b>()));
}

TEST(PeriodicFrameGenerationAlgorithm_GTest, reset_with_change_of_settings) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const double fps                     = 1.e6 / period_us;
    const timestamp accumulation_time_us = 10000;
    PeriodicFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    std::vector<FrameData> generated_frames;
    frame_generation.set_output_callback([&](timestamp ts, cv::Mat &frame) {
        generated_frames.push_back({ts, frame.clone()});
    });
    frame_generation.set_color_palette(Metavision::ColorPalette::Dark);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_fps(fps);

    // GIVEN the following events
    // clang-format off
    std::vector<EventCD> events{{
         EventCD{5, 1, 0, 970},
         EventCD{5, 5, 0, 1000},
         EventCD{5, 8, 1, 1030},
         EventCD{0, 0, 0, 1050}
    }};
    // clang-format on

    // WHEN we process the events then flush
    frame_generation.process_events(events.cbegin(), events.cend());
    frame_generation.force_generate();

    // WHEN we change the frame generation settings, reset and revert them
    frame_generation.set_color_palette(Metavision::ColorPalette::Light);
    frame_generation.set_accumulation_time_us(accumulation_time_us / 100);
    frame_generation.set_fps(fps * 2);
    frame_generation.set_color_palette(Metavision::ColorPalette::Dark);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_fps(fps);
    frame_generation.reset();

    // WHEN we process the same events again and flush
    frame_generation.process_events(events.cbegin(), events.cend());
    frame_generation.force_generate();

    // THEN the two generated frames are the same
    ASSERT_EQ(generated_frames.size(), 2);
    ASSERT_EQ(generated_frames[0].ts_us_, generated_frames[1].ts_us_);
    ASSERT_NEAR(cv::norm(generated_frames[0].frame_, generated_frames[1].frame_, cv::NORM_INF), 0.,
                std::numeric_limits<double>::epsilon());
}

TEST(PeriodicFrameGenerationAlgorithm_GTest, change_display_acc) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const double fps                     = 1.e6 / period_us;
    const timestamp accumulation_time_us = 10000;
    const bool colored                   = true;
    PeriodicFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_colors(bg_color, on_color, off_color, colored);
    frame_generation.set_fps(fps);

    // GIVEN the following events in an arbitrary time slice [first_timeslice, first_timeslice + period_us[ and
    // [first_timeslice + period_us, first_timeslice + 2*period_us[ and the above parameters
    const timestamp arbitrary_time_base      = 1255341;
    const timestamp expected_first_timeslice = period_us * (arbitrary_time_base / period_us);
    std::vector<EventCD> events{
        {EventCD{5, 1, 0, expected_first_timeslice + period_us - accumulation_time_us - 30},
         EventCD{5, 5, 0, expected_first_timeslice + period_us - accumulation_time_us},
         EventCD{5, 8, 1, expected_first_timeslice + period_us - accumulation_time_us + 30},
         EventCD{0, 0, 0, expected_first_timeslice + 2 * period_us - accumulation_time_us + 50}}};

    // WHEN we process the events...
    std::vector<FrameData> generated_frames;

    frame_generation.set_output_callback([&](timestamp ts, cv::Mat &frame) {
        generated_frames.push_back({ts, frame.clone()});
    });
    frame_generation.process_events(events.cbegin(), events.cend());

    // THEN we generate a frame that holds event data in the time slice [first_timeslice + period_us -
    // accumulation_time_us, first_timeslice + period_us[ , and bg color otherwise
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(5, 5) = off_color;
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;

    ASSERT_EQ(size_t(1), generated_frames.size());
    ASSERT_EQ(CV_8UC3, generated_frames.back().frame_.type());
    ASSERT_EQ(expected_first_timeslice + period_us, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames.back().frame_.begin<cv::Vec3b>()));

    // WHEN changing the accumulation time to a very large one
    frame_generation.set_accumulation_time_us(3 * period_us);

    // AND WHEN flushing to generate the last frame with remaining events
    frame_generation.force_generate();

    // THEN a new frame is generated but now holds all events in the previous frame + the last events
    expected_frame.setTo(bg_color);
    expected_frame.at<cv::Vec3b>(1, 5) = off_color;
    expected_frame.at<cv::Vec3b>(5, 5) = off_color;
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;
    expected_frame.at<cv::Vec3b>(0, 0) = off_color;

    ASSERT_EQ(size_t(2), generated_frames.size());
    ASSERT_EQ(CV_8UC3, generated_frames.back().frame_.type());
    ASSERT_EQ(events.back().t, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames.back().frame_.begin<cv::Vec3b>()));
}

TEST(PeriodicFrameGenerationAlgorithm_GTest, colors) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const double fps                     = 1.e6 / period_us;
    const timestamp accumulation_time_us = 10000;
    const bool colored                   = true;
    PeriodicFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_colors(bg_color, on_color, off_color, colored);
    frame_generation.set_fps(fps);

    // GIVEN the following events in an arbitrary time slice [first_timeslice, first_timeslice + period_us[ and
    // [first_timeslice + period_us, first_timeslice + 2*period_us[ and the above parameters
    const timestamp arbitrary_time_base      = 1255341;
    const timestamp expected_first_timeslice = period_us * (arbitrary_time_base / period_us);
    std::vector<EventCD> events{
        {EventCD{5, 1, 0, expected_first_timeslice + period_us - accumulation_time_us - 30},
         EventCD{5, 5, 0, expected_first_timeslice + period_us - accumulation_time_us},
         EventCD{5, 8, 1, expected_first_timeslice + period_us - accumulation_time_us + 30},
         EventCD{0, 0, 0, expected_first_timeslice + 2 * period_us - accumulation_time_us + 50}}};

    // WHEN we process the events...
    std::vector<FrameData> generated_frames;

    frame_generation.set_output_callback([&](timestamp ts, cv::Mat &frame) {
        generated_frames.push_back({ts, frame.clone()});
    });
    frame_generation.process_events(events.cbegin(), events.cend());

    // THEN we generate a frame that holds event data in the time slice [first_timeslice + period_us -
    // accumulation_time_us, first_timeslice + period_us[ , and colored bg color otherwise
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(5, 5) = off_color;
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;

    ASSERT_EQ(size_t(1), generated_frames.size());
    ASSERT_EQ(CV_8UC3, generated_frames.back().frame_.type());
    ASSERT_EQ(expected_first_timeslice + period_us, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames.back().frame_.begin<cv::Vec3b>()));

    // WHEN we change the color for a gray scale
    frame_generation.set_colors(bg_color, on_color, off_color, false);

    // AND WHEN we force_generate to process the last remaining events
    frame_generation.force_generate();

    // THEN we generate a last frame with remaining processed events information -> a frame that holds event
    // in the time slice [first_timeslice + 2 * period_us - accumulation_time_us, last event's timestamp] but in
    // grayscale representation

    expected_frame                   = cv::Mat(sensor_height, sensor_width, CV_8UC1, bg_color[0]);
    expected_frame.at<uint8_t>(0, 0) = off_color[0];

    ASSERT_EQ(size_t(2), generated_frames.size());
    ASSERT_EQ(CV_8UC1, generated_frames.back().frame_.type());
    ASSERT_EQ(events.back().t, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());

    ASSERT_TRUE(std::equal(expected_frame.begin<uint8_t>(), expected_frame.end<uint8_t>(),
                           generated_frames.back().frame_.begin<uint8_t>()));
}

TEST(PeriodicFrameGenerationAlgorithm_GTest, fps) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const timestamp next_period_us       = 10000;
    const double fps                     = 1.e6 / period_us;
    const timestamp accumulation_time_us = 10000;
    const bool colored                   = true;
    PeriodicFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_colors(bg_color, on_color, off_color, colored);
    frame_generation.set_fps(fps);

    // GIVEN the following events in an arbitrary time slice [first_timeslice, first_timeslice + period_us[ and
    // [first_timeslice + period_us, first_timeslice + 2*period_us[ and the above parameters
    const timestamp arbitrary_time_base      = 1255341;
    const timestamp expected_first_timeslice = period_us * (arbitrary_time_base / period_us); // = 1200000

    // clang-format off
    std::vector<EventCD> events{{
         EventCD{5, 1, 0, expected_first_timeslice + period_us - accumulation_time_us - 30},                  // ts = 1289970
         EventCD{5, 5, 0, expected_first_timeslice + period_us - accumulation_time_us},                       // ts = 1290000
         EventCD{5, 8, 1, expected_first_timeslice + period_us - accumulation_time_us + 30},                  // ts = 1290030
         EventCD{4, 0, 1, expected_first_timeslice + period_us + next_period_us - accumulation_time_us + 10}, // ts = 1300010
         EventCD{0, 0, 0, expected_first_timeslice + 2 * period_us - accumulation_time_us + 50}}};            // ts = 1390050
    // clang-format on

    // WHEN we process the first 3 events...
    std::vector<FrameData> generated_frames;

    frame_generation.set_output_callback([&](timestamp ts, cv::Mat &frame) {
        generated_frames.push_back({ts, frame.clone()});
    });
    frame_generation.process_events(events.cbegin(), events.cbegin() + 3);

    // THEN no frame is generated
    ASSERT_EQ(size_t(0), generated_frames.size());

    // WHEN we change the fps from 10 to 100 and process the other events
    frame_generation.set_fps(100);

    // THEN data is flushed: we generate a frame that holds event data in the time slice [1200000, 1290030 ]
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(1, 5) = off_color;
    expected_frame.at<cv::Vec3b>(5, 5) = off_color;
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;

    ASSERT_EQ(size_t(1), generated_frames.size());
    ASSERT_EQ(CV_8UC3, generated_frames.back().frame_.type());
    ASSERT_EQ(events[2].t, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames.back().frame_.begin<cv::Vec3b>()));

    // WHEN processing the remaining events
    frame_generation.process_events(events.cbegin() + 3, events.cend());

    // THEN we generate 10 more frames in [1290030, 1390030 [ (i.e one frame every 10000 us)
    expected_frame.setTo(bg_color);
    expected_frame.at<cv::Vec3b>(8, 5) = on_color;
    expected_frame.at<cv::Vec3b>(0, 4) = on_color;

    ASSERT_EQ(size_t(1 + 10), generated_frames.size());
    ASSERT_EQ(CV_8UC3, generated_frames[1].frame_.type());
    ASSERT_EQ(events[2].t + next_period_us, generated_frames[1].ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames[1].frame_.size());

    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames[1].frame_.begin<cv::Vec3b>()));

    expected_frame.setTo(bg_color);
    for (int i = 2; i < 11; ++i) {
        ASSERT_EQ(CV_8UC3, generated_frames[i].frame_.type());
        ASSERT_EQ(events[2].t + i * next_period_us, generated_frames[i].ts_us_);
        ASSERT_EQ(expected_frame.size(), generated_frames[i].frame_.size());

        ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                               generated_frames[i].frame_.begin<cv::Vec3b>()));
    }
    // WHEN we force_generate
    frame_generation.force_generate();

    // THEN generate one last frame that holds the event in the truncated last time slice [1390031, 1390050]
    expected_frame.setTo(bg_color);
    expected_frame.at<cv::Vec3b>(0, 0) = off_color;

    ASSERT_EQ(CV_8UC3, generated_frames.back().frame_.type());
    ASSERT_EQ(events.back().t, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames.back().frame_.begin<cv::Vec3b>()));
}

TEST(PeriodicFrameGenerationAlgorithm_GTest, skip_frames_to) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp period_us            = 100000;
    const double fps                     = 1.e6 / period_us;
    const timestamp accumulation_time_us = 10000;
    const bool colored                   = true;
    PeriodicFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height);
    frame_generation.set_accumulation_time_us(accumulation_time_us);
    frame_generation.set_colors(bg_color, on_color, off_color, colored);
    frame_generation.set_fps(fps);

    // GIVEN A vector of vector of events to process with timestamp belonging to consecutive time slices
    std::vector<std::vector<EventCD>> events{{EventCD{5, 1, 0, period_us - accumulation_time_us + 10}},
                                             {EventCD{5, 2, 1, 2 * period_us - accumulation_time_us + 10}},
                                             {EventCD{5, 8, 0, 3 * period_us - accumulation_time_us + 10}},
                                             {EventCD{5, 4, 1, 4 * period_us - accumulation_time_us + 10}}};

    // WHEN Skipping frame generation until the 4th buffer event's timestamp
    frame_generation.skip_frames_up_to(events.back().back().t);
    std::vector<FrameData> generated_frames;

    frame_generation.set_output_callback([&](timestamp ts, cv::Mat &frame) {
        generated_frames.push_back({ts, frame.clone()});
    });

    for (const auto &vec_ev : events) {
        frame_generation.process_events(vec_ev.cbegin(), vec_ev.cend());
    }

    // THEN only one frame is generated: the closest full time slice from the skip input i.e. the 3rd time slice
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, bg_color);
    expected_frame.at<cv::Vec3b>(8, 5) = off_color;

    ASSERT_EQ(size_t(1), generated_frames.size());
    ASSERT_EQ(CV_8UC3, generated_frames.back().frame_.type());
    ASSERT_EQ(3 * period_us, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames.back().frame_.begin<cv::Vec3b>()));

    // WHEN we force_generate
    frame_generation.force_generate();

    // THEN generate one last frame that holds the event in the last time slice
    expected_frame.setTo(bg_color);
    expected_frame.at<cv::Vec3b>(4, 5) = on_color;

    ASSERT_EQ(size_t(2), generated_frames.size());
    ASSERT_EQ(CV_8UC3, generated_frames.back().frame_.type());
    ASSERT_EQ(events.back().back().t, generated_frames.back().ts_us_);
    ASSERT_EQ(expected_frame.size(), generated_frames.back().frame_.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(),
                           generated_frames.back().frame_.begin<cv::Vec3b>()));
}

TEST(PeriodicFrameGenerationAlgorithm_GTest, test_doc_example_output) {
    // Initializes a frame generation algorithm that generates frames of size 5x1 every 5000 microseconds of events.
    // Each frame accumulating the last 10 milliseconds.
    PeriodicFrameGenerationAlgorithm frame_generation(5, 1, 10000, 200);
    std::stringstream is;

    // Sets a callback to print the frame's pixels each time one is generated.
    frame_generation.set_output_callback([&](const timestamp ts, cv::Mat &frame) {
        is << ">> Frame generated at timestamp t = " << ts << " microseconds." << std::endl;
        for (int i = 0; i < frame.rows; ++i) {
            for (int j = 0; j < frame.cols; ++j) {
                is << "(" << j << "," << i << ") = " << frame.at<cv::Vec3b>(i, j) << std::endl;
            }
        }
        is << std::endl;
    });
    std::vector<EventCD> stream_of_events;

    // Given the following events:
    // - x = 0, y = 0, p = 1, t = 0
    // - x = 1, y = 0, p = 0, t = 4000
    // - x = 2, y = 0, p = 1, t = 6000
    // - x = 3, y = 0, p = 1, t = 7000
    // - x = 4, y = 0, p = 0, t = 13000
    stream_of_events.push_back({0, 0, 1, 0});
    stream_of_events.push_back({1, 0, 0, 4000});
    stream_of_events.push_back({2, 0, 1, 6000});
    stream_of_events.push_back({3, 0, 1, 7000});
    stream_of_events.push_back({4, 0, 0, 13000});

    is << "[First call to 'process_events'] 2 events to process." << std::endl;
    frame_generation.process_events(stream_of_events.cbegin(), stream_of_events.cbegin() + 2);
    is << "[Second call to 'process_events'] 3 events to process." << std::endl;
    frame_generation.process_events(stream_of_events.cbegin() + 2, stream_of_events.cend());
    is << "[All events processed. Flushing...]" << std::endl;

    // Flush the event pending for asynchronous processing as no event will occur anymore.
    frame_generation.force_generate();
    is << "[Done]" << std::endl;

    // clang-format off
    std::string expected_message =
        "[First call to 'process_events'] 2 events to process.\n"
        "[Second call to 'process_events'] 3 events to process.\n"
        ">> Frame generated at timestamp t = 5000 microseconds.\n"
        "(0,0) = [255, 255, 255]\n"
        "(1,0) = [200, 126, 64]\n"
        "(2,0) = [52, 37, 30]\n"
        "(3,0) = [52, 37, 30]\n"
        "(4,0) = [52, 37, 30]\n"
        "\n"
        ">> Frame generated at timestamp t = 10000 microseconds.\n"
        "(0,0) = [255, 255, 255]\n"
        "(1,0) = [200, 126, 64]\n"
        "(2,0) = [255, 255, 255]\n"
        "(3,0) = [255, 255, 255]\n"
        "(4,0) = [52, 37, 30]\n"
        "\n"
        "[All events processed. Flushing...]\n"
        ">> Frame generated at timestamp t = 13000 microseconds.\n"
        "(0,0) = [52, 37, 30]\n"
        "(1,0) = [200, 126, 64]\n"
        "(2,0) = [255, 255, 255]\n"
        "(3,0) = [255, 255, 255]\n"
        "(4,0) = [200, 126, 64]\n"
        "\n"
        "[Done]\n";
    // clang-format on

    ASSERT_EQ(expected_message, is.str());
}