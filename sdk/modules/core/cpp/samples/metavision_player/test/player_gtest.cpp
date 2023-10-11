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

#include <atomic>
#include <gtest/gtest.h>
#include <stdexcept>

#include "analysis_utils.h"

TEST(PlayerTest, frame_period) {
    EXPECT_EQ(1'000'000, compute_frame_period(1));
    EXPECT_EQ(40'000, compute_frame_period(25));
    EXPECT_EQ(10'000, compute_frame_period(100));
    EXPECT_EQ(500, compute_frame_period(2000));
}

TEST(PlayerTest, accumulation_time) {
    EXPECT_EQ(250'000, compute_accumulation_time(25, 1'000'000));
    EXPECT_EQ(4'000'000, compute_accumulation_time(400, 1'000'000));
    EXPECT_EQ(40'000, compute_accumulation_time(100, 40'000));
    EXPECT_EQ(10'000, compute_accumulation_time(25, 40'000));
    EXPECT_EQ(160'000, compute_accumulation_time(400, 40'000));
}

TEST(PlayerTest, sequence_start_time) {
    EXPECT_EQ(Metavision::timestamp(10), compute_sequence_start_time(0, 7, 10, 0));
    EXPECT_EQ(Metavision::timestamp(10), compute_sequence_start_time(0, 7, 10, 50));
    EXPECT_EQ(Metavision::timestamp(10), compute_sequence_start_time(0, 7, 10, 100));
    EXPECT_EQ(Metavision::timestamp(100), compute_sequence_start_time(0, 100, 100, 0));
    EXPECT_EQ(Metavision::timestamp(200), compute_sequence_start_time(0, 100, 100, 100));
    EXPECT_EQ(Metavision::timestamp(100), compute_sequence_start_time(0, 1000, 100, 0));
    EXPECT_EQ(Metavision::timestamp(1100), compute_sequence_start_time(0, 1000, 100, 100));
    EXPECT_EQ(Metavision::timestamp(39), compute_sequence_start_time(29, 93, 10, 0));
    EXPECT_EQ(Metavision::timestamp(99), compute_sequence_start_time(29, 93, 10, 100));
}

TEST(PlayerTest, sequence_duration) {
    EXPECT_EQ(Metavision::timestamp(100), compute_sequence_duration(0, 100, 100, 0, 1));
    EXPECT_EQ(Metavision::timestamp(100), compute_sequence_duration(0, 100, 100, 100, 1));
    EXPECT_EQ(Metavision::timestamp(100), compute_sequence_duration(0, 100, 100, 0, 100));
    EXPECT_EQ(Metavision::timestamp(100), compute_sequence_duration(0, 100, 100, 100, 100));
    EXPECT_EQ(Metavision::timestamp(200), compute_sequence_duration(0, 1000, 100, 0, 1));
    EXPECT_EQ(Metavision::timestamp(200), compute_sequence_duration(0, 1000, 100, 0, 2));
    EXPECT_EQ(Metavision::timestamp(600), compute_sequence_duration(0, 1000, 100, 0, 50));
    EXPECT_EQ(Metavision::timestamp(1000), compute_sequence_duration(0, 1000, 100, 0, 100));
    EXPECT_EQ(Metavision::timestamp(100), compute_sequence_duration(0, 1000, 100, 100, 1));
    EXPECT_EQ(Metavision::timestamp(100), compute_sequence_duration(0, 1000, 100, 100, 100));
    EXPECT_EQ(Metavision::timestamp(60), compute_sequence_duration(29, 93, 10, 0, 100));
    EXPECT_EQ(Metavision::timestamp(10), compute_sequence_duration(29, 93, 10, 100, 100));
    EXPECT_EQ(Metavision::timestamp(200), compute_sequence_duration(0, 10000, 100, 0, 1));
    EXPECT_EQ(Metavision::timestamp(300), compute_sequence_duration(0, 10000, 100, 0, 2));
    EXPECT_EQ(Metavision::timestamp(400), compute_sequence_duration(0, 10000, 100, 0, 3));
    EXPECT_EQ(Metavision::timestamp(10000), compute_sequence_duration(0, 10000, 100, 0, 100));
}

TEST(PlayerTest, current_time) {
    EXPECT_EQ(Metavision::timestamp(1000), compute_current_time(1000, 0, 100));
    EXPECT_EQ(Metavision::timestamp(1100), compute_current_time(1000, 1, 100));
    EXPECT_EQ(Metavision::timestamp(3700), compute_current_time(1000, 27, 100));
}

TEST(PlayerTest, analysis_data_fps_not_changed) {
    // Given a fps of 25 (frame period of 40ms), a first time of 0ms and a last time of 1s
    // When the analysis data is updated
    // Then the fps should not be changed (as a frame period of 40ms still makes sense)
    AnalysisData data = compute_analysis_data(0, 1'000'000, 25, 100, 0, 100, 0);
    EXPECT_EQ(25, data.fps);
    EXPECT_EQ(1, data.min_fps);
    EXPECT_EQ(2'000, data.max_fps);
}

TEST(PlayerTest, analysis_data_fps_min_changed) {
    // Given a fps of 25 (frame period of 40ms), a first time of 0ms and a last time of 10ms
    // When the analysis data is updated
    // Then the fps should be changed (as a frame period of 40ms is now too big, and 10ms is the max = 100 FPS)
    AnalysisData data = compute_analysis_data(0, 10'000, 25, 100, 0, 100, 0);
    EXPECT_EQ(100, data.fps);
    EXPECT_EQ(100, data.min_fps);
    EXPECT_EQ(2'000, data.max_fps);
}

TEST(PlayerTest, analysis_data_fps_max_changed) {
    // Given a fps of 5'000
    // When the analysis data is updated
    // Then the fps should be changed (as the max is 2'000)
    AnalysisData data = compute_analysis_data(0, 1'000'000, 5'000, 100, 0, 100, 0);
    EXPECT_EQ(2'000, data.fps);
    EXPECT_EQ(1, data.min_fps);
    EXPECT_EQ(2'000, data.max_fps);
}

TEST(PlayerTest, analysis_data_accum_not_changed) {
    // Given an accumulation ratio of 100, a frame period of 10ms and a starting time of 500ms
    // When the analysis data is updated
    // Then the accumulation ratio should not be changed (as an accumulation of 10ms still makes sense)
    AnalysisData data = compute_analysis_data(0, 1'000'000, 100, 100, 50, 100, 0);
    EXPECT_EQ(100, data.accumulation_ratio);
    EXPECT_EQ(25, data.min_accumulation_ratio);
    EXPECT_EQ(400, data.max_accumulation_ratio);
}

TEST(PlayerTest, analysis_data_accum_changed) {
    // Given an accumulation ratio of 200, a frame period of 10ms and a starting time of 10ms
    // When the analysis data is updated
    // Then the accumulation ratio should be changed (as an accumulation of 20ms is now too big, and 10ms is now the
    // max)
    AnalysisData data = compute_analysis_data(0, 1'000'000, 100, 200, 0, 100, 0);
    EXPECT_EQ(100, data.accumulation_ratio);
    EXPECT_EQ(25, data.min_accumulation_ratio);
    EXPECT_EQ(100, data.max_accumulation_ratio);
}

TEST(PlayerTest, analysis_data_sequence_start) {
    AnalysisData data = compute_analysis_data(0, 1'000'000, 25, 100, 0, 100, 0);
    EXPECT_EQ(0, data.min_sequence_start_ratio);
    EXPECT_EQ(100, data.max_sequence_start_ratio);
}

TEST(PlayerTest, analysis_data_sequence_duration) {
    AnalysisData data = compute_analysis_data(0, 1'000'000, 25, 100, 0, 100, 0);
    EXPECT_EQ(1, data.min_sequence_duration_ratio);
    EXPECT_EQ(100, data.max_sequence_duration_ratio);
}

TEST(PlayerTest, analysis_data_frame_id_not_changed) {
    // Given coherent data
    // When the analysis data is updated
    // Then the frame id should not be changed
    AnalysisData data = compute_analysis_data(0, 1'000'000, 25, 100, 0, 100, 360'000);
    EXPECT_EQ(8, data.frame_id);
}

TEST(PlayerTest, analysis_data_frame_id_changed) {
    // Given incoherent data
    // When the analysis data is updated
    // Then the frame id should change to represent the same current time
    Metavision::timestamp first_time_us = 0, last_time_us = 1'000'000;
    AnalysisData data;
    int fps, frame_period_us;

    fps  = 25 * 2;
    data = compute_analysis_data(0, 1'000'000, fps, 100, 0, 100, 360'000);
    EXPECT_EQ(8 * 2 + 1, data.frame_id);
    frame_period_us = compute_frame_period(fps);
    EXPECT_EQ(360'000,
              compute_current_time(compute_sequence_start_time(first_time_us, last_time_us, frame_period_us, 0),
                                   data.frame_id, frame_period_us));

    fps  = 25 * 4;
    data = compute_analysis_data(0, 1'000'000, fps, 100, 0, 100, 360'000);
    EXPECT_EQ(8 * 4 + 3, data.frame_id);
    frame_period_us = compute_frame_period(fps);
    EXPECT_EQ(360'000,
              compute_current_time(compute_sequence_start_time(first_time_us, last_time_us, frame_period_us, 0),
                                   data.frame_id, frame_period_us));

    fps  = 25 * 8;
    data = compute_analysis_data(0, 1'000'000, fps, 100, 0, 100, 360'000);
    EXPECT_EQ(8 * 8 + 7, data.frame_id);
    frame_period_us = compute_frame_period(fps);
    EXPECT_EQ(360'000,
              compute_current_time(compute_sequence_start_time(first_time_us, last_time_us, frame_period_us, 0),
                                   data.frame_id, frame_period_us));
}