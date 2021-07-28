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
#include <thread>
#include <future>
#include <chrono>
#include <boost/any.hpp>
#include <gtest/gtest.h>
#include <stdexcept>

#include "metavision/sdk/core/pipeline/pipeline.h"
#include "metavision/sdk/core/pipeline/frame_generation_stage.h"

using namespace Metavision;

using FramePool = SharedObjectPool<cv::Mat>;
using FramePtr  = FramePool::ptr_type;
using FrameData = std::pair<timestamp, FramePtr>;

namespace {
struct MockProducingStage : public BaseStage {
    MockProducingStage(const std::vector<std::vector<Metavision::EventCD>> &evts) :
        events(evts), step(0), pool(EventBufferPool::make_bounded()) {
        set_starting_callback([this] {
            thread = std::thread([this] {
                while (true) {
                    if (stopped)
                        break;
                    if (step < events.size()) {
                        auto buffer = pool.acquire();
                        *buffer     = events[step++];
                        produce(buffer);
                    } else {
                        break;
                    }
                }
                if (!stopped)
                    complete();
            });
        });
        set_stopping_callback([this] {
            stopped = true;
            if (thread.joinable()) {
                thread.join();
            }
        });
    }

    std::thread thread;
    std::atomic<bool> stopped{false};
    std::vector<std::vector<Metavision::EventCD>> events;
    EventBufferPool pool;
    size_t step;
};

struct MockConsumingStage : public BaseStage {
    MockConsumingStage(std::vector<FrameData> &ds) : datas(ds) {
        set_consuming_callback([this](const boost::any &data) {
            try {
                datas.emplace_back(boost::any_cast<FrameData>(data));
            } catch (boost::bad_any_cast &) {}
        });
    }
    std::vector<FrameData> &datas;
};
} // namespace

// Prophesee Colors
namespace {
cv::Vec3b bg_color  = BaseFrameGenerationAlgorithm::bg_color_default();
cv::Vec3b on_color  = BaseFrameGenerationAlgorithm::on_color_default();
cv::Vec3b off_color = BaseFrameGenerationAlgorithm::off_color_default();
} // namespace

TEST(FrameGenerationStageTest, basic) {
    // GIVEN 2 events in [0, 1e5[ and 1 event in [1e5, 2e5[
    std::vector<std::vector<Metavision::EventCD>> events;

    events.push_back({Metavision::EventCD{5, 5, 0, 0}, // Event used to set the first timestamp at 0
                      Metavision::EventCD{5, 5, 0, 101}, Metavision::EventCD{5, 8, 1, 95001},
                      Metavision::EventCD{0, 0, 0, 100001}});

    // WHEN we run the pipeline with a display period of 1e5us and an accumulation time of 1e4us to generate images
    std::vector<FrameData> frames;
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<MockProducingStage>(events));
    auto &s2 = p.add_stage(std::make_unique<FrameGenerationStage>(10, 10, 10, 10), s1);
    auto &s3 = p.add_stage(std::make_unique<MockConsumingStage>(frames), s2);
    p.run();

    // THEN we only display the event in the last 1e4us of the time slice
    cv::Mat frame(10, 10, CV_8UC3, bg_color);
    frame.at<cv::Vec3b>(8, 5) = on_color;

    ASSERT_EQ(size_t(1), frames.size());
    ASSERT_EQ(100000, frames[0].first);
    ASSERT_EQ(frame.size(), frames[0].second->size());
    ASSERT_TRUE(std::equal(frame.begin<cv::Vec3b>(), frame.end<cv::Vec3b>(), frames[0].second->begin<cv::Vec3b>()));
}

TEST(FrameGenerationStageTest, round_timeshift) {
    // GIVEN 2 events in [0, 1e5[ and 1 event in [1e5, 2e5[, with an overall time shift of 10s
    const uint32_t accumulation_time_ms = 10, timeshift = 10000000;
    const int fps = 1000 / accumulation_time_ms; // We generate exactly one frame over the input events
    std::vector<std::vector<Metavision::EventCD>> events;
    events.push_back({Metavision::EventCD{5, 5, 0, timeshift},
                      Metavision::EventCD{5, 8, 1, timeshift + accumulation_time_ms * 1000 - 10},
                      Metavision::EventCD{0, 0, 0, timeshift + accumulation_time_ms * 1000}});

    // WHEN we run the pipeline with a display period of 1e5us and an accumulation time of 1e4us to generate images
    std::vector<FrameData> frames;
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<MockProducingStage>(events));
    auto &s2 = p.add_stage(std::make_unique<FrameGenerationStage>(10, 10, accumulation_time_ms, fps), s1);
    auto &s3 = p.add_stage(std::make_unique<MockConsumingStage>(frames), s2);
    p.run();

    // THEN we only display the event between [timeshift , timeshift + accumulation_time_ms[
    // accumulation time of the frame
    cv::Mat frame(10, 10, CV_8UC3, bg_color);
    frame.at<cv::Vec3b>(5, 5) = off_color;
    frame.at<cv::Vec3b>(8, 5) = on_color;
    frame.at<cv::Vec3b>(0, 0) = bg_color;

    ASSERT_EQ(size_t(1), frames.size());
    ASSERT_EQ(timeshift + accumulation_time_ms * 1000, frames[0].first);
    ASSERT_EQ(frame.size(), frames[0].second->size());

    ASSERT_TRUE(std::equal(frame.begin<cv::Vec3b>(), frame.end<cv::Vec3b>(), frames[0].second->begin<cv::Vec3b>()));
}

TEST(FrameGenerationStageTest, not_round_timeshift) {
    // GIVEN 2 events in [0, 1e5[, 2 events in [1e5, 2e5[ and 1 event in [2e5, 3e5[,
    // with an overall time shift of 10s+5us
    const timestamp step_us = 100000, timeshift = 10000000, ts_first = 5;
    std::vector<std::vector<Metavision::EventCD>> events;
    events.push_back({Metavision::EventCD{5, 5, 0, timeshift + ts_first},
                      Metavision::EventCD{5, 8, 1, timeshift + step_us - 10},
                      Metavision::EventCD{0, 0, 0, timeshift + step_us + 1}});

    // WHEN we run the pipeline with a display period of 1e5us and an accumulation time of 1e4us to generate images
    std::vector<FrameData> frames;
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<MockProducingStage>(events));
    auto &s2 = p.add_stage(std::make_unique<FrameGenerationStage>(10, 10, 10, 10), s1);
    auto &s3 = p.add_stage(std::make_unique<MockConsumingStage>(frames), s2);
    p.run();

    // THEN we only display the event in the last 1e4us of the time slice
    cv::Mat frame(10, 10, CV_8UC3, bg_color);
    frame.at<cv::Vec3b>(8, 5) = on_color;
    frame.at<cv::Vec3b>(5, 5) = bg_color;

    ASSERT_EQ(size_t(1), frames.size());
    ASSERT_EQ(timeshift + step_us, frames[0].first);
    ASSERT_EQ(frame.size(), frames[0].second->size());

    ASSERT_TRUE(std::equal(frame.begin<cv::Vec3b>(), frame.end<cv::Vec3b>(), frames[0].second->begin<cv::Vec3b>()));
}

TEST(FrameGenerationStageTest, overflow_pool) {
    // GIVEN a lot of dummy eventbuffers exhausting the producing stage's object pool memory
    std::vector<std::vector<Metavision::EventCD>> events;
    for (int i = 0; i < 300; ++i) {
        events.push_back({Metavision::EventCD{5, 5, 0, 10000 + i}});
    }
    events.push_back({Metavision::EventCD{5, 5, 0, 95001}, Metavision::EventCD{0, 0, 0, 100001}});

    // WHEN we run the pipeline with a display period of 1e5us and an accumulation time of 1e4us to generate images
    std::vector<FrameData> frames;
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<MockProducingStage>(events));
    auto &s2 = p.add_stage(std::make_unique<FrameGenerationStage>(10, 10, 10, 10), s1);
    auto &s3 = p.add_stage(std::make_unique<MockConsumingStage>(frames), s2);
    p.run();

    // THEN the frame generation stage does not stall or block, since it does not store
    // directly the buffer's shared pointer and the eventbuffers can be reused.
    cv::Mat frame(10, 10, CV_8UC3, bg_color);
    frame.at<cv::Vec3b>(5, 5) = off_color;

    ASSERT_EQ(size_t(1), frames.size());
    ASSERT_EQ(100000, frames[0].first);
    ASSERT_EQ(frame.size(), frames[0].second->size());

    ASSERT_TRUE(std::equal(frame.begin<cv::Vec3b>(), frame.end<cv::Vec3b>(), frames[0].second->begin<cv::Vec3b>()));
}
