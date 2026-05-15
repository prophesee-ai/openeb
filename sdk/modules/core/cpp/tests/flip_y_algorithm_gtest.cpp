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

#include "metavision/sdk/core/algorithms/flip_y_algorithm.h"

using namespace Metavision;

// The list of events types we want to test the filter on
typedef ::testing::Types<std::vector<Event2d>> TestingCases;

template<class EventBufferType>
class FlipYAlgorithm_GTest : public ::testing::Test {
public:
    using Event                      = typename EventBufferType::value_type;
    FlipYAlgorithm_GTest()           = default;
    ~FlipYAlgorithm_GTest() override = default;

    void restart() {
        algorithm_ = std::make_unique<FlipYAlgorithm>(0);
    }

    void initialize(std::int64_t xmax, std::int64_t ymax) {
        std::int64_t x(36), y(43), xstep(10), ystep(11);

        input_.clear();
        for (auto lastts = 0, end_time = 100000; lastts < end_time; lastts += 10000) {
            Event ev = Event();
            ev.x     = x;
            ev.y     = y;
            ev.p     = 0;
            ev.t     = 0;
            input_.push_back(ev);
            x = (x + xstep) % xmax;
            y = (y + ystep) % ymax;
        }
    }

    void initialize_fixed(std::uint16_t x, std::uint16_t y) {
        input_.clear();
        for (auto lastts = 0, end_time = 100000; lastts < end_time; lastts += 10000) {
            Event ev = Event();
            ev.x     = x;
            ev.y     = y;
            ev.p     = 0;
            ev.t     = 0;
            input_.push_back(ev);
        }
    }

    FlipYAlgorithm *algorithm() const {
        return algorithm_.get();
    }

    void process_output() {
        output_.clear();
        algorithm_->process_events(input_.cbegin(), input_.cend(), std::back_inserter(output_));
    }

protected:
    std::unique_ptr<FlipYAlgorithm> algorithm_ = nullptr;
    EventBufferType input_;
    EventBufferType output_;
};

TYPED_TEST_CASE(FlipYAlgorithm_GTest, TestingCases);

TYPED_TEST(FlipYAlgorithm_GTest, few_events_flipped) {
    // Restart the test
    this->restart();

    // Initialize the buffer data
    this->initialize(200, 300);

    // If we update the height, we expect the same value
    const std::int16_t height_minus_one = 240 - 1;
    this->algorithm()->set_height_minus_one(height_minus_one);
    ASSERT_EQ(height_minus_one, this->algorithm()->height_minus_one());

    // Process the input buffer, we don't expect any throw
    EXPECT_NO_THROW({ this->process_output(); });

    // So, It should remain of the same size
    ASSERT_EQ(this->input_.size(), this->output_.size());

    // Finally, it should store the correct data
    auto iter_input     = this->input_.cbegin();
    auto iter_input_end = this->input_.cend();
    auto iter_output    = this->output_.cbegin();
    for (; iter_input != iter_input_end; ++iter_input, ++iter_output) {
        EXPECT_EQ(iter_input->x, iter_output->x);
        EXPECT_EQ(iter_input->p, iter_output->p);
        EXPECT_EQ(iter_input->t, iter_output->t);
        EXPECT_EQ(height_minus_one - iter_input->y, iter_output->y);
    }
}
