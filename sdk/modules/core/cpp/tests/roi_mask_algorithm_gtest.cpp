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

#include "metavision/sdk/base/events/event2d.h"
#include "metavision/sdk/core/algorithms/roi_mask_algorithm.h"

using namespace Metavision;

template<typename EventType>
struct ProcesserWithBackInserter {
    void operator()(const std::vector<EventType> &input_events, std::vector<EventType> &output_events,
                    RoiMaskAlgorithm &algorithm) {
        output_events.clear();
        algorithm.process_events(input_events.cbegin(), input_events.cend(), std::back_inserter(output_events));
    }
};

template<typename EventType>
struct ProcesserWithIterator {
    void operator()(const std::vector<EventType> &input_events, std::vector<EventType> &output_events,
                    RoiMaskAlgorithm &algorithm) {
        output_events.resize(std::distance(input_events.cbegin(), input_events.cend())); // max size that can expect
        auto it_end = algorithm.process_events(input_events.cbegin(), input_events.cend(), output_events.begin());
        output_events.resize(std::distance(output_events.begin(), it_end));
    }
};

template<typename EventType, template<typename> class Func>
struct ParamsSet {
    using Event             = EventType;
    using ProcesserTypeFunc = Func<EventType>;
};

struct XyEvent {
    unsigned short x;
    unsigned short y;
};

typedef ::testing::Types<ParamsSet<XyEvent, ProcesserWithBackInserter>, ParamsSet<XyEvent, ProcesserWithIterator>>
    TestingCases;

template<typename ParamsSetType>
class RoiMaskAlgorithm_GTest : public ::testing::Test {
public:
    using Event                        = typename ParamsSetType::Event;
    using EventPosition                = std::pair<std::uint16_t, std::uint16_t>;
    RoiMaskAlgorithm_GTest()           = default;
    ~RoiMaskAlgorithm_GTest() override = default;

    virtual void SetUp() override {
        input_.clear();
        output_.clear();

        cv::Mat pixel_mask_(max_height, max_width, CV_64F);
        ;
        for (std::size_t i = 0; i < max_height; i++) {
            for (std::size_t j = 0; j < max_width; j++) {
                if (j >= x0_corner_invalid_mask && j <= x1_corner_invalid_mask && i >= y0_corner_invalid_mask &&
                    i <= y1_corner_invalid_mask) {
                    pixel_mask_.at<double>(i, j) = 0;
                } else {
                    pixel_mask_.at<double>(i, j) = 1;
                }
            }
        }
        algorithm_ = std::make_unique<RoiMaskAlgorithm>(pixel_mask_);
    }

    void initialize(EventPosition initial, EventPosition last, std::uint16_t step) {
        for (auto x = initial.first, y = initial.second; x <= last.first && y <= last.second; x += step, y += step) {
            Event event = Event();
            event.x     = x;
            event.y     = y;
            input_.push_back(event);
        }
    }

    void process_output() {
        processer_(input_, output_, *algorithm_);
    }

protected:
    typename ParamsSetType::ProcesserTypeFunc processer_;

    // Defines the size of the mask
    const std::size_t max_width  = 304;
    const std::size_t max_height = 240;

    // Defines the rectangle region that is invalid inside the mask
    const std::size_t x0_corner_invalid_mask = 60;
    const std::size_t y0_corner_invalid_mask = 40;
    const std::size_t x1_corner_invalid_mask = 235;
    const std::size_t y1_corner_invalid_mask = 195;

    std::unique_ptr<RoiMaskAlgorithm> algorithm_ = nullptr;
    std::vector<Event> input_;
    std::vector<Event> output_;
    cv::Mat pixel_mask_;
};

TYPED_TEST_CASE(RoiMaskAlgorithm_GTest, TestingCases);
TYPED_TEST(RoiMaskAlgorithm_GTest, test_few_in_valid_region_mask) {
    this->initialize({0, 0}, {this->x0_corner_invalid_mask - 10, this->y0_corner_invalid_mask - 10}, 1);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, It should preserve the same size
    ASSERT_EQ(this->input_.size(), this->output_.size());
}

TYPED_TEST(RoiMaskAlgorithm_GTest, test_few_close_to_invalid_region_mask) {
    this->initialize({this->x0_corner_invalid_mask - 10, this->y0_corner_invalid_mask - 10},
                     {this->x0_corner_invalid_mask - 1, this->y0_corner_invalid_mask - 1}, 1);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, It should preserve the same size
    ASSERT_EQ(this->input_.size(), this->output_.size());
}

TYPED_TEST(RoiMaskAlgorithm_GTest, test_few_in_corner_invalid_region_mask) {
    this->initialize({this->x0_corner_invalid_mask, this->y0_corner_invalid_mask},
                     {this->x1_corner_invalid_mask, this->y0_corner_invalid_mask}, 1);

    // Some values in the border bottom
    this->initialize({this->x0_corner_invalid_mask, this->y1_corner_invalid_mask},
                     {this->x1_corner_invalid_mask, this->y1_corner_invalid_mask}, 1);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, the output buffer should be empty
    ASSERT_NE(this->input_.size(), this->output_.size());
    ASSERT_EQ(this->output_.empty(), true);
}

TYPED_TEST(RoiMaskAlgorithm_GTest, test_few_inside_invalid_region_mask) {
    this->initialize({this->x0_corner_invalid_mask, this->y0_corner_invalid_mask},
                     {this->x1_corner_invalid_mask, this->y1_corner_invalid_mask}, 1);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, the output buffer should be empty
    ASSERT_NE(this->input_.size(), this->output_.size());
    ASSERT_EQ(this->output_.empty(), true);
}

TYPED_TEST(RoiMaskAlgorithm_GTest, test_few_in_rectangle_equivalent_to_invalid_region_in_mask) {
    this->algorithm_->enable_rectangle(this->x0_corner_invalid_mask, this->y0_corner_invalid_mask,
                                       this->x1_corner_invalid_mask, this->y1_corner_invalid_mask);

    // Some values in the border top
    this->initialize({this->x0_corner_invalid_mask, this->y0_corner_invalid_mask},
                     {this->x1_corner_invalid_mask, this->y0_corner_invalid_mask}, 1);

    // Some values in the border bottom
    this->initialize({this->x0_corner_invalid_mask, this->y1_corner_invalid_mask},
                     {this->x1_corner_invalid_mask, this->y1_corner_invalid_mask}, 1);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, It should preserve the same size
    ASSERT_EQ(this->input_.size(), this->output_.size());
}

TYPED_TEST(RoiMaskAlgorithm_GTest, test_few_in_rectangle_inside_invalid_region_in_mask) {
    this->algorithm_->enable_rectangle(this->x0_corner_invalid_mask * 2, this->y0_corner_invalid_mask * 2,
                                       this->x1_corner_invalid_mask, this->y1_corner_invalid_mask);

    // Some values in the border top
    this->initialize({this->x0_corner_invalid_mask * 2, this->y0_corner_invalid_mask * 2},
                     {this->x1_corner_invalid_mask, this->y0_corner_invalid_mask}, 1);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, It should preserve the same size
    ASSERT_EQ(this->input_.size(), this->output_.size());
}

TYPED_TEST(RoiMaskAlgorithm_GTest, test_few_in_rectangle_random_region_outside_defined_size) {
    this->algorithm_->enable_rectangle(this->x1_corner_invalid_mask, this->y1_corner_invalid_mask,
                                       this->x1_corner_invalid_mask + 10, this->y1_corner_invalid_mask + 10);

    // Some values in the border top
    this->initialize({this->x1_corner_invalid_mask, this->y1_corner_invalid_mask},
                     {this->x1_corner_invalid_mask + 10, this->y1_corner_invalid_mask + 10}, 1);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, It should preserve the same size
    ASSERT_EQ(this->input_.size(), this->output_.size());
}
