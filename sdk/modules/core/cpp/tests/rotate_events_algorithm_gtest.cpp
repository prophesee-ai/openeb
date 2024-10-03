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
#include <boost/math/constants/constants.hpp>

#include "metavision/sdk/core/algorithms/rotate_events_algorithm.h"

#define boost_pif boost::math::constants::pi<float>()
using namespace Metavision;

template<typename EventType>
struct ProcesserWithBackInserter {
    void operator()(const std::vector<EventType> &input_events, std::vector<EventType> &output_events,
                    RotateEventsAlgorithm &algorithm) {
        output_events.clear();
        algorithm.process_events(input_events.cbegin(), input_events.cend(), std::back_inserter(output_events));
    }
};

template<typename EventType>
struct ProcesserWithIterator {
    void operator()(const std::vector<EventType> &input_events, std::vector<EventType> &output_events,
                    RotateEventsAlgorithm &algorithm) {
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

// The list of events types we want to test the filter on
typedef ::testing::Types<ParamsSet<Event2d, ProcesserWithBackInserter>, ParamsSet<Event2d, ProcesserWithIterator>>
    TestingCases;

template<typename ParamsSetType>
class RotateEventsAlgorithm_GTest : public ::testing::Test {
public:
    using Event                             = typename ParamsSetType::Event;
    RotateEventsAlgorithm_GTest()           = default;
    ~RotateEventsAlgorithm_GTest() override = default;

    void initialize() {
        const int width       = 5;
        const int height      = 3;
        const float angle_rad = boost_pif / 2;
        algorithm_            = std::make_unique<RotateEventsAlgorithm>(width - 1, height - 1, angle_rad);

        timestamp t = 0;
        for (int idx_row = 0; idx_row < height; ++idx_row) {
            for (int idx_col = 0; idx_col < width; ++idx_col) {
                Event ev = Event(idx_col, idx_row, 0, t++);
                input_.push_back(ev);
            }
        }

        // only events in a square of 3x3 around the center are rotated clockwise by M_PI / 2
        expected_output_.push_back(Event(3, 0, 0, 1));
        expected_output_.push_back(Event(3, 1, 0, 2));
        expected_output_.push_back(Event(3, 2, 0, 3));
        expected_output_.push_back(Event(2, 0, 0, 6));
        expected_output_.push_back(Event(2, 1, 0, 7));
        expected_output_.push_back(Event(2, 2, 0, 8));
        expected_output_.push_back(Event(1, 0, 0, 11));
        expected_output_.push_back(Event(1, 1, 0, 12));
        expected_output_.push_back(Event(1, 2, 0, 13));
    }

    RotateEventsAlgorithm *algorithm() const {
        return algorithm_.get();
    }

    void process_output() {
        processer_(input_, output_, *algorithm_);
    }

protected:
    typename ParamsSetType::ProcesserTypeFunc processer_;
    std::unique_ptr<RotateEventsAlgorithm> algorithm_ = nullptr;
    std::vector<Event> input_;
    std::vector<Event> expected_output_;
    std::vector<Event> output_;
};

TYPED_TEST_CASE(RotateEventsAlgorithm_GTest, TestingCases);

TYPED_TEST(RotateEventsAlgorithm_GTest, few_events_rotated) {
    // Initialize the buffer data
    this->initialize();

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, It should remain of the same size
    ASSERT_EQ(this->expected_output_.size(), this->output_.size());

    // Finally, it should store the correct data
    auto iter_expected_output     = this->expected_output_.cbegin();
    auto iter_expected_output_end = this->expected_output_.cend();
    auto iter_output              = this->output_.cbegin();
    for (; iter_expected_output != iter_expected_output_end; ++iter_expected_output, ++iter_output) {
        EXPECT_EQ(iter_expected_output->x, iter_output->x);
        EXPECT_EQ(iter_expected_output->y, iter_output->y);
        EXPECT_EQ(iter_expected_output->p, iter_output->p);
        EXPECT_EQ(iter_expected_output->t, iter_output->t);
    }
}
