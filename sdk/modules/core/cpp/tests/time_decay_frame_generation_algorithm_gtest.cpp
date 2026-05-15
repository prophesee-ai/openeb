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

#include "metavision/sdk/core/algorithms/time_decay_frame_generation_algorithm.h"
#include "metavision/sdk/base/events/event_cd.h"

using namespace Metavision;

TEST(TimeDecayFrameGenerationAlgorithm_GTest, nominal_grayscale) {
    // GIVEN a TimeDecayFrameGenerationAlgorithm instance with below parameters
    const int sensor_width                    = 3;
    const int sensor_height                   = 3;
    const timestamp exponential_decay_time_us = 1000;
    const Metavision::ColorPalette palette    = Metavision::ColorPalette::Gray;
    TimeDecayFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height, exponential_decay_time_us, palette);

    // GIVEN the following events in the time slice [0, 10000]
    std::vector<EventCD> events{{EventCD{0, 0, 0, 100}, EventCD{1, 0, 0, 101}, EventCD{1, 1, 1, 101},
                                 EventCD{1, 2, 1, 1000}, EventCD{2, 0, 0, 7000}, EventCD{2, 1, 1, 9000},
                                 EventCD{2, 2, 1, 9500}, EventCD{0, 0, 0, 10000}, EventCD{0, 1, 1, 10000}}};

    // WHEN we process the events and generate a frame
    frame_generation.process_events(events.cbegin(), events.cend());
    cv::Mat generated_frame;
    frame_generation.generate(generated_frame, true);

    // THEN the generated frame is as expected
    EXPECT_EQ(frame_generation.get_exponential_decay_time_us(), exponential_decay_time_us);
    EXPECT_EQ(generated_frame.rows, sensor_height);
    EXPECT_EQ(generated_frame.cols, sensor_width);
    EXPECT_EQ(generated_frame.channels(), 1);
    EXPECT_EQ(generated_frame.depth(), CV_8U);

    EXPECT_EQ(generated_frame.at<uchar>(0, 0), 0);
    EXPECT_EQ(generated_frame.at<uchar>(1, 0), 255);
    EXPECT_EQ(generated_frame.at<uchar>(2, 0), 128);
    EXPECT_EQ(generated_frame.at<uchar>(0, 1), 128);
    EXPECT_EQ(generated_frame.at<uchar>(1, 1), 128);
    EXPECT_EQ(generated_frame.at<uchar>(2, 1), 128);
    EXPECT_NEAR(generated_frame.at<uchar>(0, 2), 121, 1.f);
    EXPECT_NEAR(generated_frame.at<uchar>(1, 2), 174, 1.f);
    EXPECT_NEAR(generated_frame.at<uchar>(2, 2), 205, 1.f);
}

TEST(TimeDecayFrameGenerationAlgorithm_GTest, nominal_colored) {
    // GIVEN a TimeDecayFrameGenerationAlgorithm instance with below parameters
    const int sensor_width                    = 3;
    const int sensor_height                   = 1;
    const timestamp exponential_decay_time_us = 1000;
    const Metavision::ColorPalette palette    = Metavision::ColorPalette::Dark;
    TimeDecayFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height, exponential_decay_time_us, palette);

    // GIVEN the following events in the time slice [0, 10000]
    std::vector<EventCD> events{{EventCD{0, 0, 0, 10000}, EventCD{1, 0, 0, 4000}, EventCD{2, 0, 1, 10000}}};

    // WHEN we process the events and generate a frame
    frame_generation.process_events(events.cbegin(), events.cend());
    cv::Mat generated_frame;
    frame_generation.generate(generated_frame, true);

    // THEN the generated frame is as expected
    EXPECT_EQ(frame_generation.get_exponential_decay_time_us(), exponential_decay_time_us);
    EXPECT_EQ(generated_frame.rows, sensor_height);
    EXPECT_EQ(generated_frame.cols, sensor_width);
    EXPECT_EQ(generated_frame.channels(), 3);
    EXPECT_EQ(generated_frame.depth(), CV_8U);

    EXPECT_EQ(generated_frame.at<cv::Vec3b>(0, 0)(0),
              cvRound(255 * get_color(palette, Metavision::ColorType::Negative).b));
    EXPECT_EQ(generated_frame.at<cv::Vec3b>(0, 0)(1),
              cvRound(255 * get_color(palette, Metavision::ColorType::Negative).g));
    EXPECT_EQ(generated_frame.at<cv::Vec3b>(0, 0)(2),
              cvRound(255 * get_color(palette, Metavision::ColorType::Negative).r));

    EXPECT_EQ(generated_frame.at<cv::Vec3b>(0, 1)(0),
              cvRound(255 * get_color(palette, Metavision::ColorType::Background).b));
    EXPECT_EQ(generated_frame.at<cv::Vec3b>(0, 1)(1),
              cvRound(255 * get_color(palette, Metavision::ColorType::Background).g));
    EXPECT_EQ(generated_frame.at<cv::Vec3b>(0, 1)(2),
              cvRound(255 * get_color(palette, Metavision::ColorType::Background).r));

    EXPECT_EQ(generated_frame.at<cv::Vec3b>(0, 2)(0),
              cvRound(255 * get_color(palette, Metavision::ColorType::Positive).b));
    EXPECT_EQ(generated_frame.at<cv::Vec3b>(0, 2)(1),
              cvRound(255 * get_color(palette, Metavision::ColorType::Positive).g));
    EXPECT_EQ(generated_frame.at<cv::Vec3b>(0, 2)(2),
              cvRound(255 * get_color(palette, Metavision::ColorType::Positive).r));
}

TEST(TimeDecayFrameGenerationAlgorithm_GTest, preallocated_grayscale) {
    // GIVEN a TimeDecayFrameGenerationAlgorithm instance with below parameters
    const int sensor_width                    = 3;
    const int sensor_height                   = 3;
    const timestamp exponential_decay_time_us = 1000;
    const Metavision::ColorPalette palette    = Metavision::ColorPalette::Gray;
    TimeDecayFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height, exponential_decay_time_us, palette);

    // GIVEN some events being processed
    std::vector<EventCD> events{{EventCD{0, 0, 0, 100}}};
    frame_generation.process_events(events.cbegin(), events.cend());

    // WHEN we generate the frame using a pre-allocated frame larger than needed, THEN it does not throw and the frame
    // still has the previous dimensions
    cv::Mat large_frame(2 * sensor_height, 2 * sensor_width, CV_8UC1);
    cv::Mat generated_frame = large_frame(cv::Rect(0, 0, sensor_width, sensor_height));
    ASSERT_NO_THROW(frame_generation.generate(generated_frame, false));
    EXPECT_EQ(large_frame.rows, 2 * sensor_height);
    EXPECT_EQ(large_frame.cols, 2 * sensor_width);
    EXPECT_EQ(large_frame.channels(), 1);
    EXPECT_EQ(large_frame.depth(), CV_8U);
}

TEST(TimeDecayFrameGenerationAlgorithm_GTest, preallocated_colored) {
    // GIVEN a TimeDecayFrameGenerationAlgorithm instance with below parameters
    const int sensor_width                    = 3;
    const int sensor_height                   = 3;
    const timestamp exponential_decay_time_us = 1000;
    const Metavision::ColorPalette palette    = Metavision::ColorPalette::Dark;
    TimeDecayFrameGenerationAlgorithm frame_generation(sensor_width, sensor_height, exponential_decay_time_us, palette);

    // GIVEN some events being processed
    std::vector<EventCD> events{{EventCD{0, 0, 0, 100}}};
    frame_generation.process_events(events.cbegin(), events.cend());

    // WHEN we generate the frame using a pre-allocated frame larger than needed, THEN it does not throw and the frame
    // still has the previous dimensions
    cv::Mat large_frame(2 * sensor_height, 2 * sensor_width, CV_8UC3);
    cv::Mat generated_frame = large_frame(cv::Rect(0, 0, sensor_width, sensor_height));
    ASSERT_NO_THROW(frame_generation.generate(generated_frame, false));
    EXPECT_EQ(large_frame.rows, 2 * sensor_height);
    EXPECT_EQ(large_frame.cols, 2 * sensor_width);
    EXPECT_EQ(large_frame.channels(), 3);
    EXPECT_EQ(large_frame.depth(), CV_8U);
}

TEST(TimeDecayFrameGenerationAlgorithm_GTest, wrong_frame_format) {
    // GIVEN a colored & grayscale TimeDecayFrameGenerationAlgorithm instances with below parameters
    const int sensor_width                    = 3;
    const int sensor_height                   = 3;
    const timestamp exponential_decay_time_us = 1000;
    TimeDecayFrameGenerationAlgorithm frame_generation_colored(sensor_width, sensor_height, exponential_decay_time_us,
                                                               Metavision::ColorPalette::Dark);
    TimeDecayFrameGenerationAlgorithm frame_generation_grayscale(sensor_width, sensor_height, exponential_decay_time_us,
                                                                 Metavision::ColorPalette::Gray);

    // GIVEN some events being processed
    std::vector<EventCD> events{{EventCD{0, 0, 0, 100}}};
    frame_generation_colored.process_events(events.cbegin(), events.cend());
    frame_generation_grayscale.process_events(events.cbegin(), events.cend());

    // WHEN we generate the frame using a pre-allocated frame than needed, THEN it does not throw and the frame still
    // has the right dimensions
    cv::Mat frame_bad_size(sensor_height / 2, sensor_width, CV_8UC3);
    cv::Mat frame_bad_chans_c3(sensor_height, sensor_width, CV_8UC3);
    cv::Mat frame_bad_chans_c1(sensor_height, sensor_width, CV_8UC1);
    cv::Mat frame_bad_depth_c3(sensor_height, sensor_width, CV_32FC3);
    cv::Mat frame_bad_depth_c1(sensor_height, sensor_width, CV_32FC1);
    ASSERT_THROW(frame_generation_colored.generate(frame_bad_size, false), std::invalid_argument);
    ASSERT_THROW(frame_generation_colored.generate(frame_bad_chans_c1, false), std::invalid_argument);
    ASSERT_THROW(frame_generation_colored.generate(frame_bad_depth_c3, false), std::invalid_argument);
    ASSERT_THROW(frame_generation_grayscale.generate(frame_bad_size, false), std::invalid_argument);
    ASSERT_THROW(frame_generation_grayscale.generate(frame_bad_chans_c3, false), std::invalid_argument);
    ASSERT_THROW(frame_generation_grayscale.generate(frame_bad_depth_c1, false), std::invalid_argument);
}
