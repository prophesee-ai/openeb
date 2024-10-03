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

#include "metavision/sdk/core/algorithms/base_frame_generation_algorithm.h"
#include "metavision/sdk/base/events/event_cd.h"

using namespace Metavision;

class BaseFrameGenerationAlgorithm_GTest : public ::testing::Test {
public:
    BaseFrameGenerationAlgorithm_GTest() {}

    virtual ~BaseFrameGenerationAlgorithm_GTest() {}

protected:
    virtual void SetUp() override {}

    virtual void TearDown() override {}
};

TEST(BaseFrameGenerationAlgorithm_GTest, static_frame_generation_no_accumulation_time) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp accumulation_time_us = 10000;
    cv::Mat frame(sensor_height, sensor_width, CV_8UC3);

    // GIVEN the following events
    std::vector<EventCD> events{{EventCD{5, 1, 0, accumulation_time_us - 30},
                                 EventCD{5, 5, 0, accumulation_time_us - 10}, EventCD{5, 8, 1, accumulation_time_us}}};

    // WHEN we generate a frame from the input events
    BaseFrameGenerationAlgorithm::generate_frame_from_events(events.cbegin(), events.cend(), frame);

    // THEN we generate a frame that holds all events
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, BaseFrameGenerationAlgorithm::bg_color_default());
    expected_frame.at<cv::Vec3b>(1, 5) = BaseFrameGenerationAlgorithm::off_color_default();
    expected_frame.at<cv::Vec3b>(8, 5) = BaseFrameGenerationAlgorithm::on_color_default();
    expected_frame.at<cv::Vec3b>(5, 5) = BaseFrameGenerationAlgorithm::off_color_default();

    ASSERT_EQ(CV_8UC3, frame.type());
    ASSERT_EQ(expected_frame.size(), frame.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), frame.begin<cv::Vec3b>()));
}

TEST(BaseFrameGenerationAlgorithm_GTest, static_frame_generation_with_accumulation_time) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp accumulation_time_us = 10000;
    cv::Mat frame(sensor_height, sensor_width, CV_8UC3);

    // GIVEN the following events
    std::vector<EventCD> events{
        {EventCD{5, 1, 0, 0}, EventCD{5, 5, 0, accumulation_time_us + 10}, EventCD{5, 8, 1, 2 * accumulation_time_us}}};

    // WHEN we generate a frame from the input events
    BaseFrameGenerationAlgorithm::generate_frame_from_events(events.cbegin(), events.cend(), frame,
                                                             accumulation_time_us);

    // THEN we generate a frame that holds only the last 2 events as the first is not in the accumulation time interval
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC3, BaseFrameGenerationAlgorithm::bg_color_default());
    expected_frame.at<cv::Vec3b>(8, 5) = BaseFrameGenerationAlgorithm::on_color_default();
    expected_frame.at<cv::Vec3b>(5, 5) = BaseFrameGenerationAlgorithm::off_color_default();

    ASSERT_EQ(CV_8UC3, frame.type());
    ASSERT_EQ(expected_frame.size(), frame.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), frame.begin<cv::Vec3b>()));
}

TEST(BaseFrameGenerationAlgorithm_GTest, static_frame_generation_with_accumulation_time_and_color) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp accumulation_time_us = 10000;
    cv::Mat frame(sensor_height, sensor_width, CV_8UC1);

    // GIVEN the following events
    std::vector<EventCD> events{
        {EventCD{5, 1, 0, 0}, EventCD{5, 5, 0, accumulation_time_us + 10}, EventCD{5, 8, 1, 2 * accumulation_time_us}}};

    // WHEN we generate a frame from the input events
    BaseFrameGenerationAlgorithm::generate_frame_from_events(events.cbegin(), events.cend(), frame,
                                                             accumulation_time_us, Metavision::ColorPalette::Gray);

    // THEN we generate a frame that holds only the last 2 events as the first is not in the accumulation time interval
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC1,
                           get_bgr_color(Metavision::ColorPalette::Gray, Metavision::ColorType::Background)[0]);
    expected_frame.at<uint8_t>(8, 5) =
        get_bgr_color(Metavision::ColorPalette::Gray, Metavision::ColorType::Positive)[0];
    expected_frame.at<uint8_t>(5, 5) =
        get_bgr_color(Metavision::ColorPalette::Gray, Metavision::ColorType::Negative)[0];

    ASSERT_EQ(CV_8UC1, frame.type());
    ASSERT_EQ(expected_frame.size(), frame.size());
    ASSERT_TRUE(std::equal(expected_frame.begin<uint8_t>(), expected_frame.end<uint8_t>(), frame.begin<uint8_t>()));
}

TEST(BaseFrameGenerationAlgorithm_GTest, static_frame_generation_no_accumulation_time_and_alpha) {
    const int sensor_width               = 10;
    const int sensor_height              = 10;
    const timestamp accumulation_time_us = 10000;
    cv::Mat frame(sensor_height, sensor_width, CV_8UC4);

    // GIVEN the following events
    std::vector<EventCD> events{{EventCD{5, 1, 0, accumulation_time_us - 30},
                                 EventCD{5, 5, 0, accumulation_time_us - 10}, EventCD{5, 8, 1, accumulation_time_us}}};

    // WHEN we generate a frame from the input events
    BaseFrameGenerationAlgorithm::generate_frame_from_events(events.cbegin(), events.cend(), frame, 0,
                                                             Metavision::ColorPalette::Dark,
                                                             BaseFrameGenerationAlgorithm::Parameters::BGRA);

    // THEN we generate a frame that holds all events
    auto bg_color  = BaseFrameGenerationAlgorithm::bg_color_default();
    auto off_color = BaseFrameGenerationAlgorithm::off_color_default();
    auto on_color  = BaseFrameGenerationAlgorithm::on_color_default();
    cv::Mat expected_frame(sensor_height, sensor_width, CV_8UC4, cv::Vec4b(bg_color[0], bg_color[1], bg_color[2], 255));
    expected_frame.at<cv::Vec4b>(1, 5) = cv::Vec4b(off_color[0], off_color[1], off_color[2], 255);
    expected_frame.at<cv::Vec4b>(8, 5) = cv::Vec4b(on_color[0], on_color[1], on_color[2], 255);
    expected_frame.at<cv::Vec4b>(5, 5) = cv::Vec4b(off_color[0], off_color[1], off_color[2], 255);

    ASSERT_EQ(CV_8UC4, frame.type());
    ASSERT_EQ(expected_frame.size(), frame.size());
    ASSERT_TRUE(
        std::equal(expected_frame.begin<cv::Vec3b>(), expected_frame.end<cv::Vec3b>(), frame.begin<cv::Vec3b>()));
}