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
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <thread>

#include "metavision/sdk/base/utils/object_pool.h"

TEST(ObjectPool_GTest, default_constructible) {
    // WHEN creating a shared object pool with default constructor
    Metavision::SharedObjectPool<int> pool;

    // THEN size is not null
    ASSERT_NE(0, pool.size());

    // WHEN acquiring an object
    // THEN the pool does not stall (buffer available)
    auto object = pool.acquire();

    // THEN the object acquired is not null
    ASSERT_NE(nullptr, object.get());
}

TEST(ObjectPool_GTest, bounded) {
    // WHEN creating a bounded shared object pool with static builder
    auto pool = Metavision::SharedObjectPool<int>::make_bounded(10);

    // THEN the pool has the requested size
    ASSERT_EQ(10, pool.size());

    // WHEN acquiring an object
    // THEN the pool does not stall (buffer available)
    auto object = pool.acquire();

    // THEN the object acquired is not null
    ASSERT_NE(nullptr, object.get());

    // THEN the size of the pool has decreased by one
    ASSERT_EQ(9, pool.size());

    // WHEN releasing the object
    object.reset();

    // THEN the pool size is increased by 1
    ASSERT_EQ(10, pool.size());
}

TEST(ObjectPool_GTest, bounded_forward_param) {
    // WHEN creating a bounded shared object pool with static builder and we forward argument for object allocation
    auto pool = Metavision::SharedObjectPool<std::vector<int>>::make_bounded(10, 100, 5);

    // THEN the pool has the requested size
    ASSERT_EQ(10, pool.size());

    // WHEN acquiring an object
    // THEN the pool does not stall (buffer available)
    auto object = pool.acquire();

    // THEN the object acquired is not null
    ASSERT_NE(nullptr, object.get());

    // THEN the size of the pool has decreased by one
    ASSERT_EQ(9, pool.size());

    // THEN the object has the expected init parameters
    ASSERT_EQ(100, object->size());
    for (auto data : *object) {
        ASSERT_EQ(5, data);
    }

    // WHEN releasing the object
    object.reset();

    // THEN the pool size is increased by 1
    ASSERT_EQ(10, pool.size());
}

TEST(ObjectPool_GTest, bounded_overflow) {
    // WHEN creating a bounded shared object pool with static builder
    auto pool = Metavision::SharedObjectPool<int>::make_bounded(1);

    // THEN the pool has the requested size
    ASSERT_EQ(1, pool.size());

    // WHEN acquiring an object
    // THEN the pool does not stall (buffer available)
    auto object = pool.acquire();

    // THEN the object acquired is not null
    ASSERT_NE(nullptr, object.get());

    // THEN the size of the pool has decreased by one
    ASSERT_EQ(0, pool.size());

    // WHEN request acquisition of an object but the object pool is empty
    // THEN the method stall until a buffer is given back to the pool
    std::mutex acquire_success_mutex;
    std::condition_variable acquire_success_cond;
    bool acquire_success{false};
    std::thread release_thread([&]() {
        object = pool.acquire();

        std::lock_guard<std::mutex> lock(acquire_success_mutex);
        acquire_success = true;
        acquire_success_cond.notify_all();
    });

    while (!release_thread.joinable()) {}
    std::unique_lock<std::mutex> lock(acquire_success_mutex);
    auto ret = acquire_success_cond.wait_for(lock, std::chrono::milliseconds(1), [&]() { return acquire_success; });
    ASSERT_FALSE(ret);

    // WHEN a buffer is given back in the pool
    // THEN then the acquire method returns
    object.reset();
    acquire_success_cond.wait(lock, [&]() { return acquire_success; });
    ASSERT_TRUE(acquire_success); // redundant with above cond var returning
    ASSERT_NE(nullptr, object.get());

    // THEN the size of the pool is still 0
    ASSERT_EQ(0, pool.size());

    // WHEN releasing the object
    object.reset();

    // THEN the pool size is increased by 1
    ASSERT_EQ(1, pool.size());

    release_thread.join();
}

TEST(ObjectPool_GTest, unbounded) {
    // WHEN creating an unbounded shared object pool with static builder
    auto pool = Metavision::SharedObjectPool<int>::make_unbounded(10);

    // THEN the pool has the requested size
    ASSERT_EQ(10, pool.size());

    // WHEN acquiring an object
    // THEN the pool does not stall (buffer available)
    auto object = pool.acquire();

    // THEN the object acquired is not null
    ASSERT_NE(nullptr, object.get());

    // THEN the size of the pool has decreased by one
    ASSERT_EQ(9, pool.size());

    // WHEN releasing the object
    object.reset();

    // THEN the pool size is increased by 1
    ASSERT_EQ(10, pool.size());
}

TEST(ObjectPool_GTest, unbounded_forward_param) {
    // WHEN creating an unbounded shared object pool with static builder and we forward argument for object allocation
    auto pool = Metavision::SharedObjectPool<std::vector<int>>::make_unbounded(10, 100, 5);

    // THEN the pool has the requested size
    ASSERT_EQ(10, pool.size());

    // WHEN acquiring an object
    // THEN the pool does not stall (buffer available)
    auto object = pool.acquire();

    // THEN the object acquired is not null
    ASSERT_NE(nullptr, object.get());

    // THEN the size of the pool has decreased by one
    ASSERT_EQ(9, pool.size());

    // THEN the object has the expected init parameters
    ASSERT_EQ(100, object->size());
    for (auto data : *object) {
        ASSERT_EQ(5, data);
    }

    // WHEN releasing the object
    object.reset();

    // THEN the pool size is increased by 1
    ASSERT_EQ(10, pool.size());
}

TEST(ObjectPool_GTest, unbounded_overflow) {
    // WHEN creating an unbounded shared object pool with static builder
    auto pool = Metavision::SharedObjectPool<int>::make_unbounded(1);

    // THEN the pool has the requested size
    ASSERT_EQ(1, pool.size());

    // WHEN acquiring an object
    // THEN the pool does not stall (buffer available)
    auto object = pool.acquire();

    // THEN the object acquired is not null
    ASSERT_NE(nullptr, object.get());

    // THEN the size of the pool has decreased by one
    ASSERT_EQ(0, pool.size());

    // WHEN request acquisition of an object but the object pool is empty
    // THEN the method does not stall and allocate a new buffer
    auto new_object = pool.acquire();

    // THEN the object acquired is not null
    ASSERT_NE(nullptr, new_object.get());

    // THEN the size of the pool is still 0
    ASSERT_EQ(0, pool.size());

    // WHEN releasing the object
    object.reset();

    // THEN the pool size is increased by 1
    ASSERT_EQ(1, pool.size());
}

TEST(ObjectPool_GTest, move) {
    // WHEN creating an unbounded shared object pool with static builder
    auto pool = Metavision::SharedObjectPool<int>::make_unbounded(10);

    // THEN the pool has the requested size
    ASSERT_EQ(10, pool.size());

    // WHEN acquiring an object
    // THEN the pool does not stall (buffer available)
    auto object = pool.acquire();

    // THEN the size of the pool has decreased by one
    ASSERT_EQ(9, pool.size());

    // WHEN moving the object pool
    // THEN no crashed occur
    auto moved_pool = std::move(pool);

    // THEN the size of the moved pool has is the same
    ASSERT_EQ(9, moved_pool.size());

    // WHEN resetting the object
    object.reset();

    // THEN no crash occur: object is given back to the moved pool
    ASSERT_EQ(10, moved_pool.size());
}

TEST(ObjectPool_GTest, deleted_object_pool_with_object_in_the_wild) {
    // WHEN creating an unbounded shared object pool with static builder
    auto pool =
        std::make_unique<Metavision::SharedObjectPool<int>>(Metavision::SharedObjectPool<int>::make_unbounded(10));

    // THEN the pool has the requested size
    ASSERT_EQ(10, pool->size());

    // WHEN acquiring an object
    // THEN the pool does not stall (buffer available)
    auto object = pool->acquire();

    // THEN the size of the pool has decreased by one
    ASSERT_EQ(9, pool->size());

    // WHEN releasing the object pool
    // THEN no crashed occur
    pool.reset(nullptr);

    // WHEN reseting the object
    // THEN no crash occur: object is deleted instead of being brought back to the pool
    object.reset();
}

TEST(ObjectPool_GTest, should_arrange_object_pool_with_requested_size) {
    Metavision::ObjectPool<int> obj_pool = Metavision::ObjectPool<int>::make_unbounded(1, 42);

    EXPECT_EQ(obj_pool.size(), 1);

    // We ask to arrange 2 elements, and because the pool contains 1 object already,
    // Only 1 new element will be allocated.
    EXPECT_EQ(obj_pool.arrange(2, 43), 1);
    EXPECT_EQ(obj_pool.size(), 2);

    auto first_obj  = obj_pool.acquire();
    auto second_obj = obj_pool.acquire();

    EXPECT_EQ(*first_obj, 43);
    EXPECT_EQ(*second_obj, 42);
}

TEST(ObjectPool_GTest, should_do_nothing_when_arranging_object_pool_with_smaller_capacity) {
    Metavision::ObjectPool<int> obj_pool = Metavision::ObjectPool<int>::make_unbounded(10);

    EXPECT_EQ(obj_pool.size(), 10);
    EXPECT_EQ(obj_pool.arrange(2), 0);
    EXPECT_EQ(obj_pool.size(), 10);
}

TEST(ObjectPool_GTest, should_not_arrange_on_bounded_pool) {
    Metavision::ObjectPool<int> obj_pool = Metavision::ObjectPool<int>::make_bounded(10);

    EXPECT_EQ(obj_pool.size(), 10);
    EXPECT_EQ(obj_pool.arrange(2), 0);
    EXPECT_EQ(obj_pool.arrange(100), 0);
    EXPECT_EQ(obj_pool.size(), 10);
}
