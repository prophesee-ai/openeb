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
#include <atomic>

#include "metavision/sdk/core/utils/cd_frame_generator.h"
#include "metavision/sdk/base/utils/timestamp.h"

class CDFrameGenerator_GTest : public ::testing::Test {
public:
    CDFrameGenerator_GTest() {}

    virtual ~CDFrameGenerator_GTest() {}

protected:
    virtual void SetUp() override {}

    virtual void TearDown() override {}
};

TEST_F(CDFrameGenerator_GTest, frame_size) {
    int width         = 100;
    int height        = 50;
    std::uint16_t fps = 1;
    Metavision::CDFrameGenerator cd_frame_generator(width, height, true);

    int obtained_width, obtained_height;
    std::atomic<bool> cb_called(false);
    cd_frame_generator.start(
        fps, [&obtained_width, &obtained_height, &cb_called](const Metavision::timestamp &ts, cv::Mat &frame) {
            obtained_width  = frame.cols;
            obtained_height = frame.rows;
            cb_called       = true;
        });

    std::vector<Metavision::EventCD> events = {Metavision::EventCD(5, 10, 0, 1500),
                                               Metavision::EventCD(5, 10, 0, 1500000)};
    cd_frame_generator.add_events(events.data(), events.data() + events.size());

    while (!cb_called) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    cd_frame_generator.stop();
    EXPECT_EQ(width, obtained_width);
    EXPECT_EQ(height, obtained_height);
}

TEST_F(CDFrameGenerator_GTest, generate_only_latest_frame) {
    int width         = 415;
    int height        = 225;
    std::uint16_t fps = 1;

    std::vector<cv::Mat> cd_frames;
    std::vector<cv::Mat> expected_cd_frames;

    Metavision::CDFrameGenerator cd_frame_generator(width, height, false);
    cd_frame_generator.set_display_accumulation_time_us(500000);
    cd_frame_generator.set_colors(cv::Scalar::all(128), cv::Scalar::all(255), cv::Scalar::all(0), false);

    std::atomic<int> cb_count{0};
    std::vector<Metavision::timestamp> time_cb;
    cd_frame_generator.start(fps, [&](const Metavision::timestamp &ts, const cv::Mat &frame) {
        cd_frames.push_back(frame.clone());
        time_cb.push_back(ts);
        ++cb_count;
    });

    std::vector<Metavision::EventCD> events = {
        Metavision::EventCD(234, 193, 1, 1000000 - 500000 + 2), Metavision::EventCD(252, 193, 1, 1000000 - 500000 + 2),
        Metavision::EventCD(213, 193, 1, 1000000 - 500000 + 2), Metavision::EventCD(219, 193, 0, 1000000 - 500000 + 2),
        Metavision::EventCD(218, 193, 0, 2000000 - 500000 + 2), Metavision::EventCD(207, 193, 0, 2000000 - 500000 + 2),
        Metavision::EventCD(287, 193, 0, 2000000 - 500000 + 2), Metavision::EventCD(278, 193, 0, 2000000 - 500000 + 2),
        Metavision::EventCD(389, 193, 0, 3000000 - 500000 + 2), Metavision::EventCD(387, 193, 0, 3000000 - 500000 + 2),
        Metavision::EventCD(393, 193, 0, 3000000 - 500000 + 2), Metavision::EventCD(249, 223, 1, 3000000 - 500000 + 2),
        Metavision::EventCD(240, 223, 1, 4000000 - 500000 + 2), Metavision::EventCD(237, 223, 1, 4000000 - 500000 + 2),
        Metavision::EventCD(260, 223, 1, 4000000 - 500000 + 2), Metavision::EventCD(356, 223, 1, 4000000 - 500000 + 2),
        Metavision::EventCD(413, 223, 1, 5000000 - 500000 + 2), Metavision::EventCD(209, 222, 1, 5000000 - 500000 + 2),
        Metavision::EventCD(240, 222, 1, 5000000 - 500000 + 2), Metavision::EventCD(255, 222, 1, 5000000 - 500000 + 2),
        Metavision::EventCD(236, 222, 1, 6000000 - 500000 + 2), Metavision::EventCD(263, 222, 1, 6000000 - 500000 + 2),
        Metavision::EventCD(260, 222, 1, 6000000 - 500000 + 2), Metavision::EventCD(228, 220, 1, 6000000 - 500000 + 2),
        Metavision::EventCD(239, 220, 1, 7000000 - 500000 + 2), Metavision::EventCD(244, 220, 1, 7000000 - 500000 + 2),
        Metavision::EventCD(242, 220, 1, 7000000 - 500000 + 2), Metavision::EventCD(257, 220, 1, 7000000 - 500000 + 2)};

    cd_frame_generator.add_events(events.data(), events.data() + events.size());

    while (cb_count != 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    cd_frame_generator.stop();

    // In this context, we must call the callback only twice. The first one holds event of full frame from
    // [6000000 - acc_time, 6000000 [, and the last one if due to the flush of the last events in
    // [7000000 - acc_time, 7000000 - 500000 + 2 ]
    ASSERT_EQ(2, time_cb.size());
    ASSERT_EQ(6000000, time_cb[0]);
    ASSERT_EQ(events.back().t, time_cb[1]);
    ASSERT_EQ(2, cd_frames.size());

    expected_cd_frames.push_back(cv::Mat(height, width, CV_8UC1, 128));
    expected_cd_frames.push_back(cv::Mat(height, width, CV_8UC1, 128));

    for (auto ev = events.cend() - 8; ev < events.cend() - 4; ++ev) {
        expected_cd_frames[0].at<std::uint8_t>(ev->y, ev->x) = ev->p * 255;
    }
    for (auto ev = events.cend() - 4; ev < events.cend(); ++ev) {
        expected_cd_frames[1].at<std::uint8_t>(ev->y, ev->x) = ev->p * 255;
    }

    for (int i = 0; i < 2; ++i) {
        ASSERT_TRUE(std::equal(expected_cd_frames[i].begin<uint8_t>(), expected_cd_frames[i].end<uint8_t>(),
                               cd_frames[i].begin<uint8_t>()));
    }
}

TEST_F(CDFrameGenerator_GTest, generate_all_frames) {
    int width         = 415;
    int height        = 225;
    std::uint16_t fps = 1;

    std::vector<cv::Mat> cd_frames;
    std::vector<cv::Mat> expected_cd_frames;

    Metavision::CDFrameGenerator cd_frame_generator(width, height, true);
    cd_frame_generator.set_display_accumulation_time_us(500000);
    cd_frame_generator.set_colors(cv::Scalar::all(128), cv::Scalar::all(255), cv::Scalar::all(0), false);

    std::atomic<int> cb_count{0};
    std::vector<Metavision::timestamp> time_cb;
    cd_frame_generator.start(fps, [&](const Metavision::timestamp &ts, const cv::Mat &frame) {
        cd_frames.push_back(frame.clone());
        time_cb.push_back(ts);
        ++cb_count;
    });

    std::vector<Metavision::EventCD> events = {
        Metavision::EventCD(234, 193, 1, 1000000 - 500000 + 2), Metavision::EventCD(252, 193, 1, 1000000 - 500000 + 2),
        Metavision::EventCD(213, 193, 1, 1000000 - 500000 + 2), Metavision::EventCD(219, 193, 0, 1000000 - 500000 + 2),
        Metavision::EventCD(218, 193, 0, 2000000 - 500000 + 2), Metavision::EventCD(207, 193, 0, 2000000 - 500000 + 2),
        Metavision::EventCD(287, 193, 0, 2000000 - 500000 + 2), Metavision::EventCD(278, 193, 0, 2000000 - 500000 + 2),
        Metavision::EventCD(389, 193, 0, 3000000 - 500000 + 2), Metavision::EventCD(387, 193, 0, 3000000 - 500000 + 2),
        Metavision::EventCD(393, 193, 0, 3000000 - 500000 + 2), Metavision::EventCD(249, 223, 1, 3000000 - 500000 + 2),
        Metavision::EventCD(240, 223, 1, 4000000 - 500000 + 2), Metavision::EventCD(237, 223, 1, 4000000 - 500000 + 2),
        Metavision::EventCD(260, 223, 1, 4000000 - 500000 + 2), Metavision::EventCD(356, 223, 1, 4000000 - 500000 + 2),
        Metavision::EventCD(413, 223, 1, 5000000 - 500000 + 2), Metavision::EventCD(209, 222, 1, 5000000 - 500000 + 2),
        Metavision::EventCD(240, 222, 1, 5000000 - 500000 + 2), Metavision::EventCD(255, 222, 1, 5000000 - 500000 + 2),
        Metavision::EventCD(236, 222, 1, 6000000 - 500000 + 2), Metavision::EventCD(263, 222, 1, 6000000 - 500000 + 2),
        Metavision::EventCD(260, 222, 1, 6000000 - 500000 + 2), Metavision::EventCD(228, 220, 1, 6000000 - 500000 + 2),
        Metavision::EventCD(239, 220, 1, 7000000 - 500000 + 2), Metavision::EventCD(244, 220, 1, 7000000 - 500000 + 2),
        Metavision::EventCD(242, 220, 1, 7000000 - 500000 + 2), Metavision::EventCD(257, 220, 1, 7000000 - 500000 + 2)};

    cd_frame_generator.add_events(events.data(), events.data() + events.size());

    while (cb_count != 6) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    cd_frame_generator.stop();

    // In this context, we must call the callback 7 times, once for each slice of 1000000
    ASSERT_EQ(7, time_cb.size());
    ASSERT_EQ(7, cd_frames.size());

    for (int i = 0; i < 7; ++i) {
        expected_cd_frames.push_back(cv::Mat(height, width, CV_8UC1, 128));
        for (auto ev = events.cbegin() + i * 4; ev < events.cbegin() + (i + 1) * 4; ++ev) {
            expected_cd_frames.back().at<std::uint8_t>(ev->y, ev->x) = ev->p * 255;
        }
    }

    for (int i = 0; i < 6; ++i) {
        ASSERT_EQ((i + 1) * 1000000, time_cb[i]);
        ASSERT_TRUE(std::equal(expected_cd_frames[i].begin<uint8_t>(), expected_cd_frames[i].end<uint8_t>(),
                               cd_frames[i].begin<uint8_t>()));
    }

    // the last frame corresponds the one generated during the flush caused when stopping the algorithm
    // this last frame corresponds to the time slice [6 * 1000000, last event's timestamp ]
    ASSERT_EQ(events.back().t, time_cb.back());
    ASSERT_TRUE(std::equal(expected_cd_frames.back().begin<uint8_t>(), expected_cd_frames.back().end<uint8_t>(),
                           cd_frames.back().begin<uint8_t>()));
}
