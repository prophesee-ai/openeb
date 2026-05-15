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
#include <type_traits>
#include <list>
#include <set>

#include "metavision/sdk/core/algorithms/roi_filter_algorithm.h"

using namespace Metavision;

template<typename EventType>
struct ProcesserWithBackInserter {
    void operator()(const std::vector<EventType> &input_events, std::vector<EventType> &output_events,
                    RoiFilterAlgorithm &algorithm) {
        output_events.clear();
        algorithm.process_events(input_events.cbegin(), input_events.cend(), std::back_inserter(output_events));
    }
};

template<typename EventType>
struct ProcesserWithIterator {
    void operator()(const std::vector<EventType> &input_events, std::vector<EventType> &output_events,
                    RoiFilterAlgorithm &algorithm) {
        output_events.resize(std::distance(input_events.cbegin(), input_events.cend())); // max size that can expect
        auto it_end = algorithm.process_events(input_events.cbegin(), input_events.cend(), output_events.begin());
        output_events.resize(std::distance(output_events.begin(), it_end));
    }
};

struct XyEvent {
    unsigned short x;
    unsigned short y;
};

template<typename EventType, template<typename> class Func>
struct ParamsSet {
    using Event             = EventType;
    using ProcesserTypeFunc = Func<EventType>;
};

typedef ::testing::Types<ParamsSet<XyEvent, ProcesserWithBackInserter>, ParamsSet<XyEvent, ProcesserWithIterator>>
    TestingCases;

template<typename ParamsSetType>
class RoiFilterAlgorithm_GTest : public ::testing::Test {
public:
    using Event                          = typename ParamsSetType::Event;
    using EventPosition                  = std::pair<std::uint16_t, std::uint16_t>;
    RoiFilterAlgorithm_GTest()           = default;
    ~RoiFilterAlgorithm_GTest() override = default;

    virtual void SetUp() override {
        input_.clear();
        output_.clear();
        algorithm_ = std::make_unique<RoiFilterAlgorithm>(x0, y0, x1, y1, false);
    }

    void initialize(EventPosition initial, EventPosition last, std::uint16_t step) {
        std::vector<Event> events;
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
    const std::uint16_t x0 = 0;
    const std::uint16_t y0 = 0;
    const std::uint16_t x1 = 203;
    const std::uint16_t y1 = 140;

    std::unique_ptr<RoiFilterAlgorithm> algorithm_ = nullptr;
    std::vector<Event> input_;
    std::vector<Event> output_;
};

// An event inside the region of interest

TYPED_TEST_CASE(RoiFilterAlgorithm_GTest, TestingCases);

TYPED_TEST(RoiFilterAlgorithm_GTest, test_few_inside_region) {
    this->initialize({this->x0, this->y0}, {this->x1, this->y1}, 10);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, It should preserve the same size
    ASSERT_EQ(this->input_.size(), this->output_.size());
}

TYPED_TEST(RoiFilterAlgorithm_GTest, test_few_close_region) {
    this->initialize({this->x1 - 10, this->y1 - 10}, {this->x1, this->y1}, 1);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, It should preserve the same size
    ASSERT_EQ(this->input_.size(), this->output_.size());
}

TYPED_TEST(RoiFilterAlgorithm_GTest, test_in_upper_left_corner) {
    this->initialize({this->x0, this->y0}, {this->x0, this->y0}, 1);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, It should preserve the same size
    ASSERT_EQ(this->input_.size(), this->output_.size());
}

TYPED_TEST(RoiFilterAlgorithm_GTest, test_in_lower_right_corner) {
    this->initialize({this->x1, this->y1}, {this->x1, this->y1}, 1);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, It should preserve the same size
    ASSERT_EQ(this->input_.size(), this->output_.size());
}

TYPED_TEST(RoiFilterAlgorithm_GTest, test_outside_but_close_to_border) {
    this->initialize({this->x1 + 1, this->y1 + 1}, {this->x1 + 10, this->y1 + 10}, 1);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, the output buffer should be empty
    ASSERT_NE(this->input_.size(), this->output_.size());
    ASSERT_EQ(this->output_.empty(), true);
}

TYPED_TEST(RoiFilterAlgorithm_GTest, test_outside_but_far_from_border) {
    this->initialize({this->x1 + 100, this->y1 + 100}, {this->x1 + 150, this->y1 + 150}, 1);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, the output buffer should be empty
    ASSERT_NE(this->input_.size(), this->output_.size());
    ASSERT_EQ(this->output_.empty(), true);
}

TYPED_TEST(RoiFilterAlgorithm_GTest, test_outside_the_border_width) {
    this->initialize({this->x1 + 1, this->y0}, {this->x1 + 10, this->y1}, 1);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, the output buffer should be empty
    ASSERT_NE(this->input_.size(), this->output_.size());
    ASSERT_EQ(this->output_.empty(), true);
}

TYPED_TEST(RoiFilterAlgorithm_GTest, test_outside_the_border_height) {
    // Initialize a set of valid elements inside the box
    this->initialize({this->x0, this->y1 + 1}, {this->x1, this->y1 + 10}, 1);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, the output buffer should be empty
    ASSERT_NE(this->input_.size(), this->output_.size());
    ASSERT_EQ(this->output_.empty(), true);
}

TYPED_TEST(RoiFilterAlgorithm_GTest, test_some_valid_some_outside_border) {
    // Initialize a set of valid elements inside the box
    constexpr auto number_valid = 10;
    this->initialize({this->x1 + 1 - number_valid, this->y1 + 1 - number_valid}, {this->x1, this->y1}, 1);

    constexpr auto number_invalid = 10;
    this->initialize({this->x1 + 1, this->y1 + 1}, {this->x1 + 1 + number_invalid, this->y1 + 1 + number_invalid}, 1);

    // Check that the number of elements inside the buffer are the correct ones
    ASSERT_NE(this->input_.size(), number_invalid + number_valid);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, the output buffer should be made only of valid events
    ASSERT_NE(this->input_.size(), this->output_.size());
    ASSERT_EQ(this->output_.size(), number_valid);
}
