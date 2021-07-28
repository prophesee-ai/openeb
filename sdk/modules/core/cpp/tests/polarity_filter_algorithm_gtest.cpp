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
#include "metavision/sdk/core/algorithms/polarity_filter_algorithm.h"

using namespace Metavision;

template<typename EventType>
struct ProcesserWithBackInserter {
    void operator()(const std::vector<EventType> &input_events, std::vector<EventType> &output_events,
                    PolarityFilterAlgorithm &algorithm) {
        output_events.clear();
        algorithm.process_events(input_events.cbegin(), input_events.cend(), std::back_inserter(output_events));
    }
};

template<typename EventType>
struct ProcesserWithIterator {
    void operator()(const std::vector<EventType> &input_events, std::vector<EventType> &output_events,
                    PolarityFilterAlgorithm &algorithm) {
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

typedef ::testing::Types<ParamsSet<Event2d, ProcesserWithBackInserter>, ParamsSet<Event2d, ProcesserWithIterator>>
    TestingCases;

template<typename ParamsSetType>
class PolarityFilterAlgorithm_GTest : public ::testing::Test {
public:
    using Event                               = typename ParamsSetType::Event;
    using EventPosition                       = std::pair<std::uint16_t, std::uint16_t>;
    PolarityFilterAlgorithm_GTest()           = default;
    ~PolarityFilterAlgorithm_GTest() override = default;

    virtual void SetUp() override {
        input_.clear();
        output_.clear();
        algorithm_ = std::make_unique<PolarityFilterAlgorithm>(polarity_);
    }

    void initialize(EventPosition initial, EventPosition last, std::uint16_t step, std::int16_t polarity) {
        for (auto x = initial.first, y = initial.second; x <= last.first && y <= last.second; x += step, y += step) {
            Event event = Event();
            event.x     = x;
            event.y     = y;
            event.p     = polarity;
            event.t     = 0;
            input_.push_back(event);
        }
    }

    void process_output() {
        processer_(input_, output_, *algorithm_);
    }

protected:
    typename ParamsSetType::ProcesserTypeFunc processer_;
    std::unique_ptr<PolarityFilterAlgorithm> algorithm_ = nullptr;
    std::vector<Event> input_;
    std::vector<Event> output_;
    const std::int16_t polarity_ = 512;
};

TYPED_TEST_CASE(PolarityFilterAlgorithm_GTest, TestingCases);
TYPED_TEST(PolarityFilterAlgorithm_GTest, test_few_same_polarity) {
    this->initialize({0, 0}, {10, 10}, 1, this->polarity_);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, It should remain of the same size
    ASSERT_EQ(this->input_.size(), this->output_.size());

    // Finally, it should store the same data
    auto iter_input     = this->input_.cbegin();
    auto iter_input_end = this->input_.cend();
    auto iter_output    = this->output_.cbegin();
    for (; iter_input != iter_input_end; ++iter_input, ++iter_output) {
        ASSERT_LE(*iter_input, *iter_output);
    }
}

TYPED_TEST(PolarityFilterAlgorithm_GTest, test_few_close_polarity) {
    this->initialize({0, 0}, {10, 10}, 1, static_cast<std::int16_t>(this->polarity_ * 5. / 6.));

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, the output buffer should be empty
    ASSERT_NE(this->input_.size(), this->output_.size());
    ASSERT_EQ(this->output_.empty(), true);
}

TYPED_TEST(PolarityFilterAlgorithm_GTest, test_different_polarity) {
    this->initialize({0, 0}, {10, 10}, 1, this->polarity_ * 5);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, the output buffer should be empty
    ASSERT_NE(this->input_.size(), this->output_.size());
    ASSERT_EQ(this->output_.empty(), true);
}

TYPED_TEST(PolarityFilterAlgorithm_GTest, test_some_valid_some_invalid) {
    // Initialize a set of valid elements inside the box
    constexpr auto number_valid = 10;
    this->initialize({0, 0}, {number_valid - 1, number_valid - 1}, 1, this->polarity_);

    constexpr auto number_invalid = 10;
    this->initialize({0, 0}, {number_invalid - 1, number_invalid - 1}, 1, this->polarity_ * 5);

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, the output buffer should contain only some events
    ASSERT_NE(this->input_.size(), this->output_.size());
    ASSERT_EQ(this->output_.size(), number_valid);
}
