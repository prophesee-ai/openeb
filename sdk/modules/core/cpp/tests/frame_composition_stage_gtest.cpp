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
#include "metavision/sdk/core/pipeline/frame_composition_stage.h"

using namespace Metavision;

using FramePool = SharedObjectPool<cv::Mat>;
using FramePtr  = FramePool::ptr_type;
using FrameData = std::pair<timestamp, FramePtr>;

namespace {

/// @brief Structure defining a colored dot in an image
struct Dot {
    Dot(){};
    Dot(int x, int y, const cv::Vec3b &color) : x_(x), y_(y), color_(color) {}

    int x_, y_;
    cv::Vec3b color_;
};

class FrameCompositionStageTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        dots.reserve(10);
        img_ptrs.reserve(10);
        for (size_t i = 0; i < 10; i++) {
            dots.emplace_back(Dot(i, 10 - i, cv::Vec3b(3 * i, 255 - 4 * i, 25 * i)));
            img_ptrs.emplace_back(get_new_dot_img(width, height, dots.back()));
        }
    }

    virtual void TearDown() {}

    /// @brief Generates new image with a single colored dot on a gray background
    FramePtr get_new_dot_img(int width, int height, const Dot &dot) {
        auto img_ptr = pool.acquire();
        img_ptr->create(height, width, CV_8UC3);
        img_ptr->setTo(bg_color);
        img_ptr->at<cv::Vec3b>(dot.y_, dot.x_) = dot.color_;
        return img_ptr;
    }

    /// @brief Generates the horizontal concatenation of two images with a single colored dot on a gray background
    FramePtr hconcat_dot_imgs(int width_left, int width_right, int height, const Dot &left_dot, const Dot &right_dot) {
        auto img_ptr = pool.acquire();
        img_ptr->create(height, width_left + width_right, CV_8UC3);
        img_ptr->setTo(bg_color);
        img_ptr->at<cv::Vec3b>(left_dot.y_, left_dot.x_)                = left_dot.color_;
        img_ptr->at<cv::Vec3b>(right_dot.y_, width_left + right_dot.x_) = right_dot.color_;
        return img_ptr;
    }

    const int width  = 10;
    const int height = 15;

    std::vector<Dot> dots;
    std::vector<FramePtr> img_ptrs;
    FramePool pool;
    cv::Vec3b bg_color = cv::Vec3b::all(128);
};

struct MockProducingStage : public BaseStage {
    MockProducingStage(const std::vector<FrameData> &fs) : frames(fs), step(0) {
        set_starting_callback([this] {
            thread = std::thread([this] {
                while (true) {
                    if (stopped)
                        break;
                    if (step < frames.size()) {
                        produce(frames[step++]);
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
    std::vector<FrameData> frames;
    size_t step;
};

struct MockConsumingStage : public BaseStage {
    MockConsumingStage(std::vector<FrameData> &fs) : frames(fs) {
        set_consuming_callback([this](const boost::any &data) {
            try {
                frames.emplace_back(boost::any_cast<FrameData>(data));
            } catch (boost::bad_any_cast &c) {}
        });
    }
    std::vector<FrameData> &frames;
};
} // namespace

TEST_F(FrameCompositionStageTest, basic_sync) {
    /// GIVEN a basic configuration with two synchronous left and right frame streams.
    //
    // Left :          v0      v2         v
    // Time : ---0--------1e5--------2e5--------3e5---
    // Right:          ^1      ^3         ^
    //
    // (The number besides ^ or v corresponds to the index of the test image.
    //  If there is none, it means the image is a nullptr)

    std::vector<FrameData> frames_left, frames_right;
    frames_left.push_back(std::make_pair(1e5 - 10, img_ptrs[0]));
    frames_right.push_back(std::make_pair(1e5 - 10, img_ptrs[1]));

    frames_left.push_back(std::make_pair(1e5 + 1, img_ptrs[2]));
    frames_right.push_back(std::make_pair(1e5 + 1, img_ptrs[3]));

    /// WHEN we run the pipeline
    std::vector<FrameData> full_frames;
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<MockProducingStage>(frames_left));
    auto &s2 = p.add_stage(std::make_unique<MockProducingStage>(frames_right));
    auto &s3 = p.add_stage(std::make_unique<FrameCompositionStage>(10)); // 10 FPS
    auto &s4 = p.add_stage(std::make_unique<MockConsumingStage>(full_frames), s3);
    s3.add_previous_frame_stage(s1, 0, 0, width, height);
    s3.add_previous_frame_stage(s2, width, 0, width, height);
    p.run();

    /// THEN we get the expected composed images
    std::vector<FrameData> full_frames_ref;
    FramePtr first_ref_ptr = hconcat_dot_imgs(width, width, height, dots[0], dots[1]);
    full_frames_ref.push_back(std::make_pair(1e5, first_ref_ptr));

    FramePtr second_ref_ptr = hconcat_dot_imgs(width, width, height, dots[2], dots[3]);
    full_frames_ref.push_back(std::make_pair(2e5, second_ref_ptr));

    // Check the two first images
    EXPECT_EQ(size_t(2), full_frames.size());
    for (size_t i = 0; i < 2; i++) {
        EXPECT_EQ(full_frames_ref[i].first, full_frames[i].first);
        ASSERT_EQ(full_frames_ref[i].second->size(), full_frames[i].second->size());
        EXPECT_GT(1e-6, cv::norm(*full_frames[i].second - *full_frames_ref[i].second));
    }
}

TEST_F(FrameCompositionStageTest, basic_overwrite_and_sync) {
    /// GIVEN two synchronous left and right frame streams, with an output frequency
    /// larger than the display frequency.
    //
    // Left :      v0  v2     v4         v
    // Time : ---0--------1e5--------2e5--------3e5---
    // Right:      ^1  ^3     ^5         ^
    //
    // (The number besides ^ or v corresponds to the index of the test image.
    //  If there is none, it means the image is a nullptr)

    std::vector<FrameData> frames_left, frames_right;
    frames_left.push_back(std::make_pair(1e5 - 100, img_ptrs[0]));
    frames_right.push_back(std::make_pair(1e5 - 100, img_ptrs[1]));
    frames_left.push_back(std::make_pair(1e5 - 10, img_ptrs[2]));
    frames_right.push_back(std::make_pair(1e5 - 10, img_ptrs[3]));

    frames_left.push_back(std::make_pair(1e5 + 10, img_ptrs[4]));
    frames_right.push_back(std::make_pair(1e5 + 10, img_ptrs[5]));

    /// WHEN we run the pipeline
    std::vector<FrameData> full_frames;
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<MockProducingStage>(frames_left));
    auto &s2 = p.add_stage(std::make_unique<MockProducingStage>(frames_right));
    auto &s3 = p.add_stage(std::make_unique<FrameCompositionStage>(10)); // 10 FPS
    auto &s4 = p.add_stage(std::make_unique<MockConsumingStage>(full_frames), s3);
    s3.add_previous_frame_stage(s1, 0, 0, width, height);
    s3.add_previous_frame_stage(s2, width, 0, width, height);
    p.run();

    /// THEN we get the expected composed images
    std::vector<FrameData> full_frames_ref;
    FramePtr first_ref_ptr = hconcat_dot_imgs(width, width, height, dots[2], dots[3]);
    full_frames_ref.push_back(std::make_pair(1e5, first_ref_ptr));

    FramePtr second_ref_ptr = hconcat_dot_imgs(width, width, height, dots[4], dots[5]);
    full_frames_ref.push_back(std::make_pair(2e5, second_ref_ptr));

    // Check the two first images
    EXPECT_EQ(size_t(2), full_frames.size());
    for (size_t i = 0; i < 2; i++) {
        EXPECT_EQ(full_frames_ref[i].first, full_frames[i].first);
        ASSERT_EQ(full_frames_ref[i].second->size(), full_frames[i].second->size());
        EXPECT_GT(1e-6, cv::norm(*full_frames[i].second - *full_frames_ref[i].second));
    }
}

TEST_F(FrameCompositionStageTest, unbalanced_streams) {
    /// GIVEN two left and right frame streams, such as the frequency of the left stream
    /// is much larger than the right one.
    //
    // Left :      v0  v1 v2  v3  v4      v7    v8             v
    // Time : ---0-------------------1e5-------------------2e5------
    // Right:      ^5           ^6        ^9                   ^
    //
    // (The number besides ^ or v corresponds to the index of the test image.
    //  If there is none, it means the image is a nullptr)

    std::vector<FrameData> frames_left, frames_right;
    frames_left.push_back(std::make_pair(1e5 - 100, img_ptrs[0]));
    frames_left.push_back(std::make_pair(1e5 - 80, img_ptrs[1]));
    frames_left.push_back(std::make_pair(1e5 - 60, img_ptrs[2]));
    frames_left.push_back(std::make_pair(1e5 - 40, img_ptrs[3]));
    frames_left.push_back(std::make_pair(1e5 - 20, img_ptrs[4]));

    frames_right.push_back(std::make_pair(1e5 - 100, img_ptrs[5]));
    frames_right.push_back(std::make_pair(1e5 - 30, img_ptrs[6]));

    frames_left.push_back(std::make_pair(1e5 + 1, img_ptrs[7]));
    frames_left.push_back(std::make_pair(1e5 + 10, img_ptrs[8]));
    frames_right.push_back(std::make_pair(1e5 + 1, img_ptrs[9]));

    /// WHEN we run the pipeline
    std::vector<FrameData> full_frames;
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<MockProducingStage>(frames_left));
    auto &s2 = p.add_stage(std::make_unique<MockProducingStage>(frames_right));
    auto &s3 = p.add_stage(std::make_unique<FrameCompositionStage>(10)); // 10 FPS
    auto &s4 = p.add_stage(std::make_unique<MockConsumingStage>(full_frames), s3);
    s3.add_previous_frame_stage(s1, 0, 0, width, height);
    s3.add_previous_frame_stage(s2, width, 0, width, height);
    p.run();

    /// THEN we get the expected composed images
    std::vector<FrameData> full_frames_ref;
    FramePtr first_ref_ptr = hconcat_dot_imgs(width, width, height, dots[4], dots[6]);
    full_frames_ref.push_back(std::make_pair(1e5, first_ref_ptr));

    FramePtr second_ref_ptr = hconcat_dot_imgs(width, width, height, dots[8], dots[9]);
    full_frames_ref.push_back(std::make_pair(2e5, second_ref_ptr));

    // Check the two first images
    EXPECT_EQ(size_t(2), full_frames.size());
    for (size_t i = 0; i < 2; i++) {
        EXPECT_EQ(full_frames_ref[i].first, full_frames[i].first);
        ASSERT_EQ(full_frames_ref[i].second->size(), full_frames[i].second->size());
        EXPECT_GT(1e-6, cv::norm(*full_frames[i].second - *full_frames_ref[i].second));
    }
}

TEST_F(FrameCompositionStageTest, nullptrs_as_temporal_markers) {
    /// GIVEN two synchronous left and right frame streams, using nullptrs as temporal markers.
    //
    // Left :          v0         v2         v          v
    // Time : ---0--------1e5--------2e5--------3e5--------4e5---
    // Right:          ^1         ^          ^3         ^
    //
    // (The number besides ^ or v corresponds to the index of the test image.
    //  If there is none, it means the image is a nullptr)

    std::vector<FrameData> frames_left, frames_right;
    frames_left.push_back(std::make_pair(1e5 - 10, img_ptrs[0]));
    frames_right.push_back(std::make_pair(1e5 - 10, img_ptrs[1]));

    frames_left.push_back(std::make_pair(2e5 - 10, img_ptrs[2]));
    frames_right.push_back(std::make_pair(2e5 - 10, FramePtr()));

    frames_left.push_back(std::make_pair(3e5 - 10, FramePtr()));
    frames_right.push_back(std::make_pair(3e5 - 10, img_ptrs[3]));

    /// WHEN we run the pipeline
    std::vector<FrameData> full_frames;
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<MockProducingStage>(frames_left));
    auto &s2 = p.add_stage(std::make_unique<MockProducingStage>(frames_right));
    auto &s3 = p.add_stage(std::make_unique<FrameCompositionStage>(10)); // 10 FPS
    auto &s4 = p.add_stage(std::make_unique<MockConsumingStage>(full_frames), s3);
    s3.add_previous_frame_stage(s1, 0, 0, width, height);
    s3.add_previous_frame_stage(s2, width, 0, width, height);
    p.run();

    /// THEN we get the expected composed images
    std::vector<FrameData> full_frames_ref;
    FramePtr first_ref_ptr = hconcat_dot_imgs(width, width, height, dots[0], dots[1]);
    full_frames_ref.push_back(std::make_pair(1e5, first_ref_ptr));

    FramePtr second_ref_ptr = hconcat_dot_imgs(width, width, height, dots[2], dots[1]);
    full_frames_ref.push_back(std::make_pair(2e5, second_ref_ptr));

    FramePtr third_ref_ptr = hconcat_dot_imgs(width, width, height, dots[2], dots[3]);
    full_frames_ref.push_back(std::make_pair(3e5, third_ref_ptr));

    // Check the three first images
    EXPECT_EQ(size_t(3), full_frames.size());
    for (size_t i = 0; i < 3; i++) {
        EXPECT_EQ(full_frames_ref[i].first, full_frames[i].first);
        ASSERT_EQ(full_frames_ref[i].second->size(), full_frames[i].second->size());
        EXPECT_GT(1e-6, cv::norm(*full_frames[i].second - *full_frames_ref[i].second));
    }
}

TEST_F(FrameCompositionStageTest, nullptr_overwriting_risk) {
    /// GIVEN two synchronous left and right frame streams, using nullptrs as temporal markers.
    //
    // Left :       v0  v v1 v        v v v2  v         v
    // Time : ---0---------------1e5---------------2e5--------3e5--
    // Right:          ^3   ^         ^   ^4  ^         ^
    //
    // (The number besides ^ or v corresponds to the index of the test image.
    //  If there is none, it means the image is a nullptr)

    std::vector<FrameData> frames_left, frames_right;
    frames_left.push_back(std::make_pair(1e5 - 40, img_ptrs[0]));
    frames_left.push_back(std::make_pair(1e5 - 30, FramePtr()));
    frames_left.push_back(std::make_pair(1e5 - 20, img_ptrs[1]));
    frames_left.push_back(std::make_pair(1e5 - 10, FramePtr()));

    frames_right.push_back(std::make_pair(1e5 - 20, img_ptrs[3]));
    frames_right.push_back(std::make_pair(1e5 - 10, FramePtr()));

    frames_left.push_back(std::make_pair(2e5 - 40, FramePtr()));
    frames_left.push_back(std::make_pair(2e5 - 30, FramePtr()));
    frames_left.push_back(std::make_pair(2e5 - 20, img_ptrs[2]));
    frames_left.push_back(std::make_pair(2e5 - 10, FramePtr()));

    frames_right.push_back(std::make_pair(2e5 - 30, FramePtr()));
    frames_right.push_back(std::make_pair(2e5 - 20, img_ptrs[4]));
    frames_right.push_back(std::make_pair(2e5 - 10, FramePtr()));

    /// WHEN we run the pipeline
    std::vector<FrameData> full_frames;
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<MockProducingStage>(frames_left));
    auto &s2 = p.add_stage(std::make_unique<MockProducingStage>(frames_right));
    auto &s3 = p.add_stage(std::make_unique<FrameCompositionStage>(10)); // 10 FPS
    auto &s4 = p.add_stage(std::make_unique<MockConsumingStage>(full_frames), s3);
    s3.add_previous_frame_stage(s1, 0, 0, width, height);
    s3.add_previous_frame_stage(s2, width, 0, width, height);
    p.run();

    /// THEN we get the expected composed images and the nulltprs haven't overwritten the valid ptrs
    std::vector<FrameData> full_frames_ref;
    FramePtr first_ref_ptr = hconcat_dot_imgs(width, width, height, dots[1], dots[3]);
    full_frames_ref.push_back(std::make_pair(1e5, first_ref_ptr));

    FramePtr second_ref_ptr = hconcat_dot_imgs(width, width, height, dots[2], dots[4]);
    full_frames_ref.push_back(std::make_pair(2e5, second_ref_ptr));

    // Check the three first images
    EXPECT_EQ(size_t(2), full_frames.size());
    for (size_t i = 0; i < 2; i++) {
        EXPECT_EQ(full_frames_ref[i].first, full_frames[i].first);
        ASSERT_EQ(full_frames_ref[i].second->size(), full_frames[i].second->size());
        EXPECT_GT(1e-6, cv::norm(*full_frames[i].second - *full_frames_ref[i].second));
    }
}

TEST_F(FrameCompositionStageTest, slow_streams) {
    /// GIVEN two left and right frame streams, such as their output frequency is smaller
    /// than the display frequency.
    //
    // Left :          v0                              v3        v
    // Time : ---0--------1e5--------2e5--------3e5--------4e5--------5e5
    // Right:          ^1                   ^2                   ^
    //
    // (The number besides ^ or v corresponds to the index of the test image.
    //  If there is none, it means the image is a nullptr)

    std::vector<FrameData> frames_left, frames_right;
    frames_left.push_back(std::make_pair(1e5 - 10, img_ptrs[0]));
    frames_right.push_back(std::make_pair(1e5 - 10, img_ptrs[1]));

    frames_right.push_back(std::make_pair(3e5 - 10, img_ptrs[2]));

    frames_left.push_back(std::make_pair(4e5 - 10, img_ptrs[3]));

    /// WHEN we run the pipeline
    std::vector<FrameData> full_frames;
    Pipeline p;
    auto &s1 = p.add_stage(std::make_unique<MockProducingStage>(frames_left));
    auto &s2 = p.add_stage(std::make_unique<MockProducingStage>(frames_right));
    auto &s3 = p.add_stage(std::make_unique<FrameCompositionStage>(10)); // 10 FPS
    auto &s4 = p.add_stage(std::make_unique<MockConsumingStage>(full_frames), s3);
    s3.add_previous_frame_stage(s1, 0, 0, width, height);
    s3.add_previous_frame_stage(s2, width, 0, width, height);
    p.run();

    /// THEN we get the expected composed images
    std::vector<FrameData> full_frames_ref;
    FramePtr first_ref_ptr = hconcat_dot_imgs(width, width, height, dots[0], dots[1]);
    full_frames_ref.push_back(std::make_pair(1e5, first_ref_ptr));

    FramePtr second_ref_ptr = hconcat_dot_imgs(width, width, height, dots[0], dots[1]);
    full_frames_ref.push_back(std::make_pair(2e5, second_ref_ptr));

    FramePtr third_ref_ptr = hconcat_dot_imgs(width, width, height, dots[0], dots[2]);
    full_frames_ref.push_back(std::make_pair(3e5, third_ref_ptr));

    FramePtr fourth_ref_ptr = hconcat_dot_imgs(width, width, height, dots[3], dots[2]);
    full_frames_ref.push_back(std::make_pair(4e5, fourth_ref_ptr));

    // Check the three first images
    EXPECT_EQ(size_t(4), full_frames.size());
    for (size_t i = 0; i < 4; i++) {
        EXPECT_EQ(full_frames_ref[i].first, full_frames[i].first);
        ASSERT_EQ(full_frames_ref[i].second->size(), full_frames[i].second->size());
        EXPECT_GT(1e-6, cv::norm(*full_frames[i].second - *full_frames_ref[i].second));
    }
}
