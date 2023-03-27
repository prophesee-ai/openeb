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

#include <algorithm>
#include <random>
#include <gtest/gtest.h>

#include "metavision/sdk/core/utils/shared_buffer_queue.h"

class SharedBufferQueue_GTest : public ::testing::Test {
public:
    SharedBufferQueue_GTest() {
        while (gt_.empty()) {
            generate_test_data_and_gt();
        }
    }

protected:
    struct Foo {
        Foo() : Foo(0) {}

        Foo(int v) {
            value = v;
        }

        bool operator==(const Foo &rhs) const {
            return value == rhs.value;
        }

        int value;
    };

    using FooBuffer = std::vector<Foo>;
    FooBuffer gt_;
    Metavision::SharedBufferQueue<Foo> shared_queue_;

private:
    void generate_test_data_and_gt() {
        std::mt19937 mt(42);

        std::uniform_int_distribution<> dist_buffer_number(10, 50);
        std::uniform_int_distribution<> dist_value_number(1, 100);
        std::uniform_int_distribution<> dist_instances_number(0, 5);

        int value             = 0;
        const auto num_buffer = dist_buffer_number(mt);
        for (int i = 0; i < num_buffer; ++i) {
            const auto value_number = dist_value_number(mt);

            std::vector<Foo> buffer;
            for (int j = 0; j < value_number; ++j) {
                const auto instances_number = dist_instances_number(mt);
                for (int k = 0; k < instances_number; ++k)
                    buffer.emplace_back(Foo(value));
                ++value;
            }

            shared_queue_.insert(std::make_shared<FooBuffer>(buffer));
            gt_.insert(gt_.end(), buffer.cbegin(), buffer.cend());
        }
    }
};

TEST_F(SharedBufferQueue_GTest, empty_queue) {
    // GIVEN an empty shared buffer queue
    Metavision::SharedBufferQueue<Foo> queue;

    // WHEN we check if the queue is empty
    // THEN it is
    ASSERT_TRUE(queue.empty());

    // WHEN we test the begin and end iterators
    // THEN they are the same
    ASSERT_TRUE(queue.begin() == queue.end());
    ASSERT_TRUE(queue.cbegin() == queue.cend());
    ASSERT_TRUE(std::begin(queue) == std::end(queue));
}

TEST_F(SharedBufferQueue_GTest, size) {
    // GIVEN a shared buffer queue filled with random buffers and values
    // WHEN we check its size
    // THEN it corresponds to the GT
    ASSERT_TRUE(shared_queue_.size() == gt_.size());
}

TEST_F(SharedBufferQueue_GTest, loop) {
    // GIVEN a shared buffer queue filled with random buffers and values
    auto it    = shared_queue_.cbegin();
    auto gt_it = gt_.cbegin();

    // WHEN we loop over its elements
    // THEN they are all equal to the GT's ones
    for (; it != shared_queue_.cend(); ++it, ++gt_it)
        ASSERT_TRUE(*it == *gt_it);
}

TEST_F(SharedBufferQueue_GTest, find) {
    // GIVEN a shared buffer queue filled with random buffers and values
    // WHEN
    // - we pick up a value at a random location in the buffer
    // - we look for that value in the shared buffer queue
    std::mt19937 mt(42);
    std::uniform_int_distribution<> dist_index(0, gt_.size() - 1);

    const auto index         = dist_index(mt);
    const auto value_to_find = gt_[index].value;

    auto it    = std::find(shared_queue_.cbegin(), shared_queue_.cend(), Foo(value_to_find));
    auto gt_it = std::find(gt_.cbegin(), gt_.cend(), Foo(value_to_find));

    // THEN
    // - the value is found
    // - the distance between the first element and the found one is the same in the shared buffer queue and the GT
    // buffer
    ASSERT_TRUE(it != shared_queue_.cend());
    ASSERT_TRUE(gt_it != gt_.cend());
    ASSERT_TRUE(*it == *gt_it);
    ASSERT_TRUE(std::distance(it, shared_queue_.cend()) == std::distance(gt_it, gt_.cend()));
}

TEST_F(SharedBufferQueue_GTest, lower_bound) {
    // GIVEN a shared buffer queue filled with random buffers and values
    // WHEN
    // - we pick up a random value (that we know is in the shared buffer)
    // - we look for the lower bound of this value in the shared buffer
    std::mt19937 mt(42);
    std::uniform_int_distribution<> dist_value_to_find(0, gt_.back().value);

    const auto value_to_find = dist_value_to_find(mt);

    auto it = std::lower_bound(shared_queue_.cbegin(), shared_queue_.cend(), value_to_find,
                               [](const Foo &foo, int v) { return foo.value < v; });

    auto gt_it =
        std::lower_bound(gt_.cbegin(), gt_.cend(), value_to_find, [](const Foo &foo, int v) { return foo.value < v; });

    // THEN
    // - the lower bound is found
    // - it has the same value as the GT's lower bound
    // - the distance between the first element and the lower bound is the same in the shared buffer queue and the GT
    // buffer
    ASSERT_TRUE(it != shared_queue_.cend());
    ASSERT_TRUE(gt_it != gt_.cend());

    ASSERT_TRUE(*it == *gt_it);
    ASSERT_TRUE(std::distance(it, shared_queue_.cend()) == std::distance(gt_it, gt_.cend()));
}

TEST_F(SharedBufferQueue_GTest, upper_bound) {
    // GIVEN a shared buffer queue filled with random buffers and values
    // WHEN
    // - we pick up a random value (that may not exist in the shared buffer)
    // - we look for the upper bound of this value in the shared buffer
    std::mt19937 mt(42);
    std::uniform_int_distribution<> dist_value_to_find(0, gt_.back().value);

    const auto value_to_find = dist_value_to_find(mt);

    auto it = std::upper_bound(shared_queue_.cbegin(), shared_queue_.cend(), value_to_find,
                               [](int v, const Foo &foo) { return v < foo.value; });

    auto gt_it =
        std::upper_bound(gt_.cbegin(), gt_.cend(), value_to_find, [](int v, const Foo &foo) { return v < foo.value; });

    if (gt_it == gt_.cend()) {
        // THEN if the upper bound is not found in the GT buffer, then it is not found either in the shared buffer queue
        ASSERT_TRUE(it == shared_queue_.cend());
    } else {
        // THEN
        // - the upper bound is found
        // - it has the same value as the GT's upper bound
        // - the distance between the first element and the upper bound is the same in the shared buffer queue and the
        // GT buffer
        ASSERT_TRUE(*it == *gt_it);
        ASSERT_TRUE(std::distance(it, shared_queue_.cend()) == std::distance(gt_it, gt_.cend()));
    }
}

TEST_F(SharedBufferQueue_GTest, clear) {
    // GIVEN a non empty shared buffer queue
    ASSERT_TRUE(!shared_queue_.empty());

    // WHEN we clear it
    shared_queue_.clear();

    // THEN it becomes empty
    ASSERT_TRUE(shared_queue_.empty());
}

TEST_F(SharedBufferQueue_GTest, erase) {
    using IntBuffer = std::vector<int>;

    struct Deleter {
        explicit Deleter(bool &b) : has_been_freed(b) {}

        void operator()(IntBuffer *buffer) {
            has_been_freed = true;
        }

        bool &has_been_freed;
    };

    bool have_been_freed[2] = {false, false};

    const auto make_shared = [&](IntBuffer &buffer, int bool_idx) {
        return std::shared_ptr<IntBuffer>(&buffer, Deleter(have_been_freed[bool_idx]));
    };

    // GIVEN a shared buffer queue built upon two shared integer buffers [0, 1, 2, 3, 4] and [5, 6, 7, 8, 9, 10]
    IntBuffer b1{0, 1, 2, 3, 4};
    IntBuffer b2{5, 6, 7, 8, 9, 10};

    Metavision::SharedBufferQueue<int> shared_queue;

    shared_queue.insert(make_shared(b1, 0));
    shared_queue.insert(make_shared(b2, 1));

    // WHEN we clean the first 3 elements
    auto it = std::find(shared_queue.cbegin(), shared_queue.cend(), 3);
    shared_queue.erase_up_to(it);

    // THEN
    // - no internal shared buffer is freed
    // - the first element of the shared buffer queue corresponds to the fourth one in the first buffer
    ASSERT_TRUE(have_been_freed[0] == false);
    ASSERT_TRUE(*shared_queue.cbegin() == 3);
    ASSERT_TRUE(&(*shared_queue.cbegin()) == &b1[3]);

    // WHEN we clean the next two elements
    it = std::find(shared_queue.cbegin(), shared_queue.cend(), 5);
    shared_queue.erase_up_to(it);

    // THEN
    // - the first internal buffer is freed
    // - the first element of the shared buffer queue corresponds to the first one in the second buffer
    ASSERT_TRUE(have_been_freed[0]);
    ASSERT_TRUE(have_been_freed[1] == false);
    ASSERT_TRUE(&(*shared_queue.cbegin()) == &b2[0]);

    // WHEN we clean all the remaining elements
    shared_queue.erase_up_to(shared_queue.cend());

    // THEN
    // - the second internal buffer is freed
    // - the shared buffer queue is empty
    ASSERT_TRUE(have_been_freed[1]);
    ASSERT_TRUE(shared_queue.empty());
}
