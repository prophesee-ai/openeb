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
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#include "metavision/sdk/core/utils/timing_profiler.h"

using namespace Metavision;

class TimingProfiler_GTest : public ::testing::Test {
public:
    TimingProfiler_GTest() {}

    ~TimingProfiler_GTest() {}

    virtual void SetUp() override {}

    virtual void TearDown() override {}
};

TEST_F(TimingProfiler_GTest, test_empty) {
    TimingProfiler<true> profiler;
    const auto &storage = profiler.get_storage_policy();
    ASSERT_TRUE(storage.get_ordered_keys().empty());
}

TEST_F(TimingProfiler_GTest, test_simple) {
    TimingProfiler<true> profiler;
    {
        TimingProfiler<true>::TimedOperation op("test", &profiler);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const auto &storage_policy = profiler.get_storage_policy();
    const auto &keys           = storage_policy.get_ordered_keys();
    ASSERT_EQ(size_t(1), keys.size());
    bool found;
    auto time = storage_policy.get_time("test", &found);
    ASSERT_TRUE(found);
    ASSERT_LE(std::chrono::milliseconds(1), time.wall);
    auto count = storage_policy.get_count("test", &found);
    ASSERT_TRUE(found);
    ASSERT_EQ(size_t(1), count);
}

TEST_F(TimingProfiler_GTest, test_duplicate) {
    TimingProfiler<true> profiler;
    {
        TimingProfiler<true>::TimedOperation op("test", &profiler);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    {
        TimingProfiler<true>::TimedOperation op("test", &profiler);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    const auto &storage_policy = profiler.get_storage_policy();
    const auto &keys           = storage_policy.get_ordered_keys();
    ASSERT_EQ(size_t(1), keys.size());
    bool found = false;
    auto time  = storage_policy.get_time("test", &found);
    ASSERT_TRUE(found);
    ASSERT_LE(std::chrono::milliseconds(3), time.wall);
    found      = false;
    auto count = storage_policy.get_count("test", &found);
    ASSERT_TRUE(found);
    ASSERT_EQ(size_t(2), count);
}

TEST_F(TimingProfiler_GTest, test_multiple) {
    TimingProfiler<true> profiler;
    {
        TimingProfiler<true>::TimedOperation op("test", &profiler);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    {
        TimingProfiler<true>::TimedOperation op("test2", &profiler);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const auto &storage_policy = profiler.get_storage_policy();
    const auto &keys           = storage_policy.get_ordered_keys();
    ASSERT_EQ(size_t(2), keys.size());
    bool found = false;
    auto time  = storage_policy.get_time("test", &found);
    ASSERT_TRUE(found);
    ASSERT_LE(std::chrono::milliseconds(1), time.wall);
    found = false;
    time  = storage_policy.get_time("test2", &found);
    ASSERT_TRUE(found);
    ASSERT_LE(std::chrono::milliseconds(1), time.wall);
    found      = false;
    auto count = storage_policy.get_count("test", &found);
    ASSERT_TRUE(found);
    ASSERT_EQ(size_t(1), count);
    found = false;
    count = storage_policy.get_count("test2", &found);
    ASSERT_TRUE(found);
    ASSERT_EQ(size_t(1), count);
}

TEST_F(TimingProfiler_GTest, test_insertion_order) {
    using TP = TimingProfiler<true, detail::ConcurrencyPolicyThreadSafe, detail::OperationStoragePolicyInsertionOrder>;
    TP profiler;
    { TP::TimedOperation op("a", &profiler); }
    { TP::TimedOperation op("c", &profiler); }
    { TP::TimedOperation op("b", &profiler); }

    const auto &storage_policy = profiler.get_storage_policy();
    const auto &keys           = storage_policy.get_ordered_keys();
    ASSERT_EQ(size_t(3), keys.size());
    ASSERT_EQ("a", keys[0]);
    ASSERT_EQ("c", keys[1]);
    ASSERT_EQ("b", keys[2]);
}

TEST_F(TimingProfiler_GTest, test_lexical_order) {
    using TP = TimingProfiler<true, detail::ConcurrencyPolicyThreadSafe, detail::OperationStoragePolicyLexicalOrder>;
    TP profiler;
    { TP::TimedOperation op("a", &profiler); }
    { TP::TimedOperation op("c", &profiler); }
    { TP::TimedOperation op("b", &profiler); }

    const auto &storage_policy = profiler.get_storage_policy();
    const auto &keys           = storage_policy.get_ordered_keys();
    ASSERT_EQ(size_t(3), keys.size());
    ASSERT_EQ("a", keys[0]);
    ASSERT_EQ("b", keys[1]);
    ASSERT_EQ("c", keys[2]);
}

TEST_F(TimingProfiler_GTest, test_thread_safe) {
    TimingProfiler<true> profiler;
    std::thread t1([&profiler]() {
        for (int i = 0; i < 1000; ++i) {
            TimingProfiler<true>::TimedOperation op("test", &profiler);
            std::this_thread::sleep_for(std::chrono::microseconds(std::rand() % 1000));
        }
    });
    std::thread t2([&profiler]() {
        for (int i = 0; i < 1000; ++i) {
            TimingProfiler<true>::TimedOperation op("test", &profiler);
            std::this_thread::sleep_for(std::chrono::microseconds(std::rand() % 1000));
        }
    });

    t1.join();
    t2.join();

    const auto &storage_policy = profiler.get_storage_policy();
    const auto &keys           = storage_policy.get_ordered_keys();
    ASSERT_EQ(size_t(1), keys.size());
    bool found = false;
    storage_policy.get_time("test", &found);
    ASSERT_TRUE(found);
    found      = false;
    auto count = storage_policy.get_count("test", &found);
    ASSERT_TRUE(found);
    ASSERT_EQ(size_t(2000), count);
}

#if 0
// We cannot guarantee that the test does not crash
// Keep the code just for reference but do not activate it
TEST_F(TimingProfiler_GTest, test_thread_unsafe) {
    using TimingProfilerType = TimingProfiler<true, detail::ConcurrencyPolicyThreadUnsafe>;
    TimingProfilerType profiler;
    std::thread t1([&profiler]() {
        for (int i = 0; i < 1000; ++i) {
            TimingProfilerType::TimedOperation op("test", &profiler);
            std::this_thread::sleep_for(std::chrono::microseconds(std::rand() % 1000));
        }
    });
    std::thread t2([&profiler]() {
        for (int i = 0; i < 1000; ++i) {
            TimingProfilerType::TimedOperation op("test", &profiler);
            std::this_thread::sleep_for(std::chrono::microseconds(std::rand() % 1000));
        }
    });

    t1.join();
    t2.join();

    // A simple check that will most probably fail but also documents the fact that thread safety is
    // required if sharing a TimingProfiler.
    const auto &storage_policy = profiler.get_storage_policy();
    const auto &keys           = storage_policy.get_ordered_keys();
    if (keys.size() != size_t(1)) {
        std::cerr << "TimingProfiler : number of operations is incoherent which is expected since we are using "
                     "thread unsafe concurrency policy."
                  << std::endl;
    }
    bool found = false;
    storage_policy.get_time("test", &found);
    if (!found) {
        std::cerr << "TimingProfiler : operation not found which is expected since we are using "
                     "thread unsafe concurrency policy."
                  << std::endl;
    }
    found      = false;
    auto count = storage_policy.get_count("test", &found);
    if (!found) {
        std::cerr << "TimingProfiler : operation not found which is expected since we are using "
                     "thread unsafe concurrency policy."
                  << std::endl;
    }
    if (count < size_t(2000)) {
        std::cerr << "TimingProfiler : count of operation is incoherent which is expected since we are using "
                     "thread unsafe concurrency policy."
                  << std::endl;
    }

    SUCCEED();
}
#endif
