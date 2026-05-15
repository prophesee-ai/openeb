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

#include <thread>
#include <future>
#include <gtest/gtest.h>

#include "metavision/sdk/core/utils/concurrent_queue.h"

TEST(ConcurrentQueueTest, get_front_empty) {
    // GIVEN an empty concurrent queue
    Metavision::ConcurrentQueue<int> queue;

    // WHEN we try to get the front element in a non-blocking way
    // THEN the function returns a non-valid element
    ASSERT_EQ(queue.pop_front(false), std::nullopt);

    // WHEN we try to get the front element (from a separate thread)
    std::packaged_task<std::optional<int>()> task([&queue]() { return queue.pop_front(); });
    auto future_result = task.get_future();
    std::thread t(std::move(task));

    // (make sure the thread is running)
    while (!t.joinable())
        std::this_thread::yield();

    // (wait a bit to make sure the thread is blocked on the get_front_or_wait call)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // THEN the function blocks
    ASSERT_TRUE(t.joinable());

    // WHEN we close the queue
    queue.close();

    // THEN the function returns a non-valid element
    ASSERT_FALSE(future_result.get());

    t.join();
}

TEST(ConcurrentQueueTest, get_front_non_empty) {
    // GIVEN a concurrent queue with one element
    Metavision::ConcurrentQueue<int> queue;
    queue.emplace(1);

    // WHEN we try to get the front element
    // THEN the function returns a valid element and the front element is the one we added
    const auto front = queue.pop_front();
    ASSERT_TRUE(*front);
    ASSERT_EQ(1, *front);
}

TEST(ConcurrentQueueTest, push_when_opened) {
    // GIVEN an opened concurrent queue
    Metavision::ConcurrentQueue<int> queue;
    queue.open();

    // WHEN we try to push an element
    // THEN the function returns true
    ASSERT_TRUE(queue.emplace(1));
}

TEST(ConcurrentQueueTest, push_when_closed) {
    // GIVEN a closed concurrent queue
    Metavision::ConcurrentQueue<int> queue;
    queue.close();

    // WHEN we try to push an element
    // THEN the function returns false
    ASSERT_FALSE(queue.emplace(1));
}

TEST(ConcurrentQueueTest, push_when_opened_and_full) {
    // GIVEN a full opened concurrent queue
    Metavision::ConcurrentQueue<int> queue(1);
    queue.open();
    queue.emplace(1);

    // WHEN we try to push a new element (from a separate thread)
    std::packaged_task<bool()> task([&queue]() { return queue.emplace(2); });
    auto future_result = task.get_future();
    std::thread t(std::move(task));

    // (make sure the thread is running)
    while (!t.joinable())
        std::this_thread::yield();

    // (wait a bit to make sure the thread is blocked on the emplace call)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // THEN the function blocks
    ASSERT_TRUE(t.joinable());

    // WHEN we pop an element from the queue
    const auto front = queue.pop_front();

    // THEN
    // - the popped element is valid
    // - the emplace call succeeded
    ASSERT_TRUE(front);
    ASSERT_EQ(1, *front);
    ASSERT_TRUE(future_result.get());

    t.join();

    // WHEN we try to push a new element in a non-blocking way
    // THEN the function returns false
    ASSERT_FALSE(queue.emplace(3, false));
}