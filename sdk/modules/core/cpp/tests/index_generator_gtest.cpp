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
#include <thread>
#include <atomic>

#include <gtest/gtest.h>

#include "metavision/sdk/core/utils/index_generator.h"

class Indexer_GTest : public ::testing::Test {
public:
    Indexer_GTest() {}

    virtual ~Indexer_GTest() {}

protected:
    virtual void SetUp() override {}

    virtual void TearDown() override {}
};

TEST_F(Indexer_GTest, constructor) {
    Metavision::IndexGenerator indexer;
    size_t index_expected(0);
    ASSERT_EQ(index_expected, indexer.get_next_index());
}

TEST_F(Indexer_GTest, get_index_monothread) {
    Metavision::IndexGenerator indexer;
    size_t max_index(1000000);
    for (size_t index_expected = 0; index_expected < max_index; ++index_expected) {
        ASSERT_EQ(index_expected, indexer.get_next_index());
    }
}

TEST_F(Indexer_GTest, get_index_multithread) {
    Metavision::IndexGenerator indexer;
    std::atomic<bool> to_start(false), to_stop(false);
    std::thread t1, t2;
    std::vector<size_t> v1, v2;
    t1 = std::thread([&] {
        while (!to_start) {}
        do { // To make sure to push in the vector at least once
            v1.push_back(indexer.get_next_index());
        } while (!to_stop);
    });

    t2 = std::thread([&] {
        while (!to_start) {}

        do { // To make sure to push in the vector at least once
            v2.push_back(indexer.get_next_index());
        } while (!to_stop);
    });

    to_start = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    to_stop = true;
    t1.join();
    t2.join();

    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());

    // First, check that the elements in the two vectors are unique :
    ASSERT_TRUE(std::adjacent_find(v1.begin(), v1.end()) == v1.end());
    ASSERT_TRUE(std::adjacent_find(v2.begin(), v2.end()) == v2.end());

    // Now, check there is no value present in both vectors
    std::vector<size_t> v3;
    std::set_difference(v2.begin(), v2.end(), v1.begin(), v1.end(), std::back_inserter(v3));
    std::set_difference(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(v3));

    ASSERT_EQ(v2.size() + v1.size(), v3.size());
    ASSERT_EQ(indexer.get_next_index(), v3.size());
}
