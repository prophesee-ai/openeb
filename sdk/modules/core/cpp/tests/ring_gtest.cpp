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
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <gtest/gtest.h>

#include <vector>
#include <atomic>
#include <iostream>

#include "metavision/sdk/core/utils/detail/ring.h"
#include "metavision/sdk/base/utils/timestamp.h"

template<typename EVENT>
using RingTest = Metavision::detail::Ring<EVENT>;

class Ring_GTest : public ::testing::Test {};

namespace {
struct Event_Gtest {
    long t;
};

typedef std::vector<Event_Gtest> type_buffer;

} // namespace

TEST_F(Ring_GTest, test_ctr_empty) {
    RingTest<Event_Gtest> r;
    EXPECT_FALSE(r.data_available());
    EXPECT_EQ(size_t(0), r.size());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_last_time());
}

TEST_F(Ring_GTest, test_add1_one_elem) {
    RingTest<Event_Gtest> r;
    RingTest<Event_Gtest>::type_eventsadd vevents;
    vevents.push_back(Event_Gtest{1});
    r.add(vevents);
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(1), r.size());
    EXPECT_EQ(Metavision::timestamp(1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(1), r.get_last_time());
}

TEST_F(Ring_GTest, test_add1_not_empty) {
    RingTest<Event_Gtest> r;
    RingTest<Event_Gtest>::type_eventsadd vevents;
    vevents.push_back(Event_Gtest{1});
    vevents.push_back(Event_Gtest{10});
    vevents.push_back(Event_Gtest{20});
    vevents.push_back(Event_Gtest{100});
    r.add(vevents);
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(4), r.size());
    EXPECT_EQ(Metavision::timestamp(1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(100), r.get_last_time());
}
TEST_F(Ring_GTest, test_get_data_nodata_available) {
    RingTest<Event_Gtest> r;
    type_buffer buf;
    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to(inserter, 10);
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(size_t(0), r.size());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_last_time());
}
TEST_F(Ring_GTest, test_get_all_data) {
    RingTest<Event_Gtest> r;
    type_buffer buf;
    RingTest<Event_Gtest>::type_eventsadd vevents;
    vevents.push_back(Event_Gtest{1});
    vevents.push_back(Event_Gtest{10});
    vevents.push_back(Event_Gtest{20});
    vevents.push_back(Event_Gtest{100});
    r.add(vevents);
    EXPECT_EQ(Metavision::timestamp(1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(100), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to(inserter, 1000);
    EXPECT_EQ(static_cast<size_t>(4), buf.size());
    EXPECT_FALSE(r.data_available());
    EXPECT_EQ(size_t(0), r.size());
}
TEST_F(Ring_GTest, test_get_all_data_but1) {
    RingTest<Event_Gtest> r;
    type_buffer buf;
    RingTest<Event_Gtest>::type_eventsadd vevents;
    vevents.push_back(Event_Gtest{1});
    vevents.push_back(Event_Gtest{10});
    vevents.push_back(Event_Gtest{20});
    vevents.push_back(Event_Gtest{100});
    r.add(vevents);
    EXPECT_EQ(Metavision::timestamp(1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(100), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to(inserter, 50);
    EXPECT_EQ(static_cast<size_t>(3), buf.size());
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(1), r.size());
}
TEST_F(Ring_GTest, test_get_all_data_eq) {
    RingTest<Event_Gtest> r;
    type_buffer buf;
    RingTest<Event_Gtest>::type_eventsadd vevents;
    vevents.push_back(Event_Gtest{1});
    vevents.push_back(Event_Gtest{10});
    vevents.push_back(Event_Gtest{20});
    vevents.push_back(Event_Gtest{100});
    r.add(vevents);
    EXPECT_EQ(Metavision::timestamp(1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(100), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to(inserter, 100);
    EXPECT_EQ(static_cast<size_t>(3), buf.size());
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(1), r.size());
}
TEST_F(Ring_GTest, test_get_partial_data) {
    RingTest<Event_Gtest> r;
    type_buffer buf;
    RingTest<Event_Gtest>::type_eventsadd vevents;
    vevents.push_back(Event_Gtest{1});
    vevents.push_back(Event_Gtest{10});
    vevents.push_back(Event_Gtest{20});
    vevents.push_back(Event_Gtest{100});
    r.add(vevents);
    EXPECT_EQ(Metavision::timestamp(1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(100), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to(inserter, 30);
    EXPECT_EQ(static_cast<size_t>(3), buf.size());
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(1), r.size());
}
TEST_F(Ring_GTest, test_get_partial_data_eq) {
    RingTest<Event_Gtest> r;
    type_buffer buf;
    RingTest<Event_Gtest>::type_eventsadd vevents;
    vevents.push_back(Event_Gtest{1});
    vevents.push_back(Event_Gtest{10});
    vevents.push_back(Event_Gtest{20});
    vevents.push_back(Event_Gtest{100});
    r.add(vevents);
    EXPECT_EQ(Metavision::timestamp(1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(100), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to(inserter, 20);
    EXPECT_EQ(static_cast<size_t>(2), buf.size());
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(2), r.size());
}

TEST_F(Ring_GTest, test_2buf_get_partial_data) {
    RingTest<Event_Gtest> r;
    type_buffer buf;
    RingTest<Event_Gtest>::type_eventsadd vevents;
    vevents.push_back(Event_Gtest{1});
    vevents.push_back(Event_Gtest{10});
    vevents.push_back(Event_Gtest{20});
    vevents.push_back(Event_Gtest{100});
    r.add(vevents);
    EXPECT_EQ(Metavision::timestamp(1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(100), r.get_last_time());

    vevents.clear();
    vevents.push_back(Event_Gtest{200});
    vevents.push_back(Event_Gtest{300});
    r.add(vevents);

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to(inserter, 300);
    EXPECT_EQ(static_cast<size_t>(5), buf.size());
    EXPECT_TRUE(r.data_available());
    EXPECT_FALSE(r.data_available(1000));
    EXPECT_EQ(size_t(1), r.size());
}

TEST_F(Ring_GTest, test_get_sparse_data_not_after_TS) {
    RingTest<Event_Gtest> r;
    type_buffer buf;
    RingTest<Event_Gtest>::type_eventsadd vevents;
    vevents.push_back(Event_Gtest{99});
    r.add(vevents);
    EXPECT_EQ(Metavision::timestamp(99), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to(inserter, 100);
    EXPECT_EQ(static_cast<size_t>(1), buf.size());
    buf.clear();
    vevents.clear();
    vevents.push_back(Event_Gtest{199});
    r.add(vevents);
    r.fill_buffer_to(inserter, 2100);
    EXPECT_EQ(static_cast<size_t>(1), buf.size());
}

TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_nodata_available) {
    RingTest<Event_Gtest> r;
    type_buffer buf;
    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to_drop_max_events(inserter, 10, 1);
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(size_t(0), r.size());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_last_time());
}

TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_gt_ts_max5) {
    RingTest<Event_Gtest> r;
    type_buffer buf;
    RingTest<Event_Gtest>::type_eventsadd vevents;
    vevents.push_back(Event_Gtest{1});
    vevents.push_back(Event_Gtest{10});
    vevents.push_back(Event_Gtest{20});
    vevents.push_back(Event_Gtest{100});
    r.add(vevents);
    EXPECT_EQ(Metavision::timestamp(1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(100), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to_drop_max_events(inserter, 1000, 5);
    EXPECT_EQ(static_cast<size_t>(4), buf.size());
    EXPECT_FALSE(r.data_available());
    EXPECT_EQ(size_t(0), r.size());
}

TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_gt_ts_max2) {
    RingTest<Event_Gtest> r;
    type_buffer buf;
    RingTest<Event_Gtest>::type_eventsadd vevents;
    vevents.push_back(Event_Gtest{1});
    vevents.push_back(Event_Gtest{10});
    vevents.push_back(Event_Gtest{20});
    vevents.push_back(Event_Gtest{100});
    r.add(vevents);
    EXPECT_EQ(Metavision::timestamp(1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(100), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to_drop_max_events(inserter, 1000, 2);
    EXPECT_EQ(static_cast<size_t>(2), buf.size());
    EXPECT_FALSE(r.data_available());
    EXPECT_EQ(size_t(0), r.size());
}

TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_lt_ts_max5) {
    RingTest<Event_Gtest> r;
    type_buffer buf;
    RingTest<Event_Gtest>::type_eventsadd vevents;
    vevents.push_back(Event_Gtest{1});
    vevents.push_back(Event_Gtest{10});
    vevents.push_back(Event_Gtest{20});
    vevents.push_back(Event_Gtest{100});
    r.add(vevents);
    EXPECT_EQ(Metavision::timestamp(1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(100), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to_drop_max_events(inserter, 25, 5);
    EXPECT_EQ(static_cast<size_t>(3), buf.size());
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(1), r.size());
}

TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_lt_ts_max2) {
    RingTest<Event_Gtest> r;
    type_buffer buf;
    RingTest<Event_Gtest>::type_eventsadd vevents;
    vevents.push_back(Event_Gtest{1});
    vevents.push_back(Event_Gtest{10});
    vevents.push_back(Event_Gtest{20});
    vevents.push_back(Event_Gtest{100});
    r.add(vevents);
    EXPECT_EQ(Metavision::timestamp(1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(100), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to_drop_max_events(inserter, 25, 2);
    EXPECT_EQ(static_cast<size_t>(2), buf.size());
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(1), r.size());
}

TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_eq_ts_max5) {
    RingTest<Event_Gtest> r;
    type_buffer buf;
    RingTest<Event_Gtest>::type_eventsadd vevents;
    vevents.push_back(Event_Gtest{1});
    vevents.push_back(Event_Gtest{10});
    vevents.push_back(Event_Gtest{20});
    vevents.push_back(Event_Gtest{100});
    r.add(vevents);
    EXPECT_EQ(Metavision::timestamp(1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(100), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to_drop_max_events(inserter, 100, 5);
    EXPECT_EQ(static_cast<size_t>(3), buf.size());
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(1), r.size());
}

TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_eq_ts_max2) {
    RingTest<Event_Gtest> r;
    type_buffer buf;
    RingTest<Event_Gtest>::type_eventsadd vevents;
    vevents.push_back(Event_Gtest{1});
    vevents.push_back(Event_Gtest{10});
    vevents.push_back(Event_Gtest{20});
    vevents.push_back(Event_Gtest{100});
    r.add(vevents);
    EXPECT_EQ(Metavision::timestamp(1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(100), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to_drop_max_events(inserter, 100, 2);
    EXPECT_EQ(static_cast<size_t>(2), buf.size());
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(1), r.size());
}

TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_large_ring_max100) {
    RingTest<Event_Gtest> r;
    type_buffer buf;

    for (int i = 0; i < 100; ++i) {
        buf.push_back(Event_Gtest{i});
        if (i % 10 == 0) {
            r.add(buf);
            buf.clear();
        }
    }
    r.add(buf);
    buf.clear();
    EXPECT_EQ(Metavision::timestamp(0), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to_drop_max_events(inserter, 50, 100);
    EXPECT_EQ(50, buf.size());
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(50), r.size());
    EXPECT_EQ(Metavision::timestamp(50), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());
}

TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_large_ring_max10) {
    RingTest<Event_Gtest> r;
    type_buffer buf;

    for (int i = 0; i < 100; ++i) {
        buf.push_back(Event_Gtest{i});
        if (i % 10 == 0) {
            r.add(buf);
            buf.clear();
        }
    }
    r.add(buf);
    buf.clear();
    EXPECT_EQ(Metavision::timestamp(0), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to_drop_max_events(inserter, 50, 10);
    EXPECT_EQ(10, buf.size());
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(50), r.size());
    EXPECT_EQ(Metavision::timestamp(50), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());
}

// similar test to large_ring_max but we take some events of the last vector
TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_large_ring_ts99_max99) {
    RingTest<Event_Gtest> r;
    type_buffer buf;

    for (int i = 0; i < 100; ++i) {
        buf.push_back(Event_Gtest{i});
        if (i % 10 == 0) {
            r.add(buf);
            buf.clear();
        }
    }
    r.add(buf);
    buf.clear();
    EXPECT_EQ(Metavision::timestamp(0), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to_drop_max_events(inserter, 99, 99);
    EXPECT_EQ(99, buf.size());
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(1), r.size());
    EXPECT_EQ(Metavision::timestamp(99), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());
}

// similar test to large_ring_max but we take some events of the last vector
TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_large_ring_ts99_max98) {
    RingTest<Event_Gtest> r;
    type_buffer buf;

    for (int i = 0; i < 100; ++i) {
        buf.push_back(Event_Gtest{i});
        if (i % 10 == 0) {
            r.add(buf);
            buf.clear();
        }
    }
    r.add(buf);
    buf.clear();
    EXPECT_EQ(Metavision::timestamp(0), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to_drop_max_events(inserter, 99, 98);
    EXPECT_EQ(98, buf.size());
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(1), r.size());
    EXPECT_EQ(Metavision::timestamp(99), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());
}

// similar test to large_ring_max but we take some events of the last vector
TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_large_ring_ts98_max99) {
    RingTest<Event_Gtest> r;
    type_buffer buf;

    for (int i = 0; i < 100; ++i) {
        buf.push_back(Event_Gtest{i});
        if (i % 10 == 0) {
            r.add(buf);
            buf.clear();
        }
    }
    r.add(buf);
    buf.clear();
    EXPECT_EQ(Metavision::timestamp(0), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to_drop_max_events(inserter, 98, 99);
    EXPECT_EQ(98, buf.size());
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(2), r.size());
    EXPECT_EQ(Metavision::timestamp(98), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());
}

TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_large_ring_no_zero_idx) {
    RingTest<Event_Gtest> r;
    type_buffer buf;

    for (int i = 0; i < 100; ++i) {
        buf.push_back(Event_Gtest{i});
        if (i % 10 == 0) {
            r.add(buf);
            buf.clear();
        }
    }
    r.add(buf);
    buf.clear();
    EXPECT_EQ(Metavision::timestamp(0), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to(inserter, 25);
    buf.clear();

    r.fill_buffer_to_drop_max_events(inserter, 100, 100);
    EXPECT_EQ(75, buf.size());
    EXPECT_FALSE(r.data_available());
    EXPECT_EQ(size_t(0), r.size());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_last_time());
}

TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_large_ring_no_zero_idx_max_10) {
    RingTest<Event_Gtest> r;
    type_buffer buf;

    for (int i = 0; i < 100; ++i) {
        buf.push_back(Event_Gtest{i});
        if (i % 10 == 0) {
            r.add(buf);
            buf.clear();
        }
    }
    r.add(buf);
    buf.clear();
    EXPECT_EQ(Metavision::timestamp(0), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to(inserter, 25);
    buf.clear();

    r.fill_buffer_to_drop_max_events(inserter, 100, 10);
    EXPECT_EQ(10, buf.size());
    EXPECT_FALSE(r.data_available());
    EXPECT_EQ(size_t(0), r.size());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_last_time());
}

TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_large_ring_no_zero_idx_max_10_not_all) {
    RingTest<Event_Gtest> r;
    type_buffer buf;

    for (int i = 0; i < 100; ++i) {
        buf.push_back(Event_Gtest{i});
        if (i % 10 == 0) {
            r.add(buf);
            buf.clear();
        }
    }
    r.add(buf);
    buf.clear();
    EXPECT_EQ(Metavision::timestamp(0), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to(inserter, 25);
    buf.clear();

    r.fill_buffer_to_drop_max_events(inserter, 80, 10);
    EXPECT_EQ(10, buf.size());
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(20), r.size());
    EXPECT_EQ(Metavision::timestamp(80), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());
}

TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_single_buffer) {
    RingTest<Event_Gtest> r;
    type_buffer buf;

    for (int i = 0; i < 100; ++i) {
        buf.push_back(Event_Gtest{i});
    }
    r.add(buf);
    buf.clear();
    EXPECT_EQ(Metavision::timestamp(0), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());

    r.drop_max_events(50);

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to_drop_max_events(inserter, 100, 100);
    EXPECT_EQ(50, buf.size());
    EXPECT_FALSE(r.data_available());
    EXPECT_EQ(size_t(0), r.size());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_last_time());
}

TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_single_buffer_no_zero_idx) {
    RingTest<Event_Gtest> r;
    type_buffer buf;

    for (int i = 0; i < 100; ++i) {
        buf.push_back(Event_Gtest{i});
    }
    r.add(buf);
    buf.clear();
    EXPECT_EQ(Metavision::timestamp(0), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to(inserter, 25);
    buf.clear();

    r.fill_buffer_to_drop_max_events(inserter, 100, 100);
    EXPECT_EQ(75, buf.size());
    EXPECT_FALSE(r.data_available());
    EXPECT_EQ(size_t(0), r.size());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_last_time());
}

TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_single_buffer_no_zero_idx_max_10) {
    RingTest<Event_Gtest> r;
    type_buffer buf;

    for (int i = 0; i < 100; ++i) {
        buf.push_back(Event_Gtest{i});
    }
    r.add(buf);
    buf.clear();
    EXPECT_EQ(Metavision::timestamp(0), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to(inserter, 25);
    buf.clear();

    r.fill_buffer_to_drop_max_events(inserter, 100, 10);
    EXPECT_EQ(10, buf.size());
    EXPECT_FALSE(r.data_available());
    EXPECT_EQ(size_t(0), r.size());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_last_time());
}

TEST_F(Ring_GTest, test_fill_buffer_to_drop_max_events_single_buffer_no_zero_idx_max_10_not_all) {
    RingTest<Event_Gtest> r;
    type_buffer buf;

    for (int i = 0; i < 100; ++i) {
        buf.push_back(Event_Gtest{i});
    }
    r.add(buf);
    buf.clear();
    EXPECT_EQ(Metavision::timestamp(0), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());

    auto inserter = std::back_inserter(buf);
    r.fill_buffer_to(inserter, 25);
    buf.clear();

    r.fill_buffer_to_drop_max_events(inserter, 80, 10);
    EXPECT_EQ(10, buf.size());
    EXPECT_TRUE(r.data_available());
    EXPECT_EQ(size_t(20), r.size());
    EXPECT_EQ(Metavision::timestamp(80), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());
}

TEST_F(Ring_GTest, test_drop_max_events_large_ring) {
    RingTest<Event_Gtest> r;
    type_buffer buf;

    for (int i = 0; i < 100; ++i) {
        buf.push_back(Event_Gtest{i});
        if (i % 10 == 0) {
            r.add(buf);
            buf.clear();
        }
    }
    EXPECT_EQ(Metavision::timestamp(0), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(90), r.get_last_time());

    r.drop_max_events(50);

    type_buffer res;
    auto inserter = std::back_inserter(res);
    r.fill_buffer_remaining(inserter);
    EXPECT_EQ(50, res.size());
    EXPECT_EQ(size_t(0), r.size());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_last_time());
}

TEST_F(Ring_GTest, test_drop_max_events_large_ring_no_zero_idx) {
    RingTest<Event_Gtest> r;
    type_buffer buf;

    for (int i = 0; i < 100; ++i) {
        buf.push_back(Event_Gtest{i});
        if (i % 10 == 0) {
            r.add(buf);
            buf.clear();
        }
    }
    EXPECT_EQ(Metavision::timestamp(0), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(90), r.get_last_time());

    type_buffer res;

    auto inserter = std::back_inserter(res);
    r.fill_buffer_to(inserter, 25);
    res.clear();

    r.drop_max_events(50);

    r.fill_buffer_remaining(inserter);
    EXPECT_EQ(50, res.size());
    EXPECT_EQ(size_t(0), r.size());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_last_time());
}

TEST_F(Ring_GTest, test_drop_max_events_single_buffer) {
    RingTest<Event_Gtest> r;
    type_buffer buf;

    for (int i = 0; i < 100; ++i) {
        buf.push_back(Event_Gtest{i});
    }

    r.add(buf);
    EXPECT_EQ(Metavision::timestamp(0), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());

    r.drop_max_events(50);

    type_buffer res;
    auto inserter = std::back_inserter(res);
    r.fill_buffer_remaining(inserter);
    EXPECT_EQ(50, res.size());
    EXPECT_EQ(size_t(0), r.size());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_last_time());
}

TEST_F(Ring_GTest, test_drop_max_events_single_buffer_no_zero_idx) {
    RingTest<Event_Gtest> r;
    type_buffer buf;

    for (int i = 0; i < 100; ++i) {
        buf.push_back(Event_Gtest{i});
    }

    r.add(buf);
    EXPECT_EQ(Metavision::timestamp(0), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(99), r.get_last_time());

    type_buffer res;
    auto inserter = std::back_inserter(res);

    r.fill_buffer_to(inserter, 25);
    res.clear();

    r.drop_max_events(50);

    r.fill_buffer_remaining(inserter);
    EXPECT_EQ(50, res.size());
    EXPECT_EQ(size_t(0), r.size());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_first_time());
    EXPECT_EQ(Metavision::timestamp(-1), r.get_last_time());
}

namespace {
class TestThread {
public:
    TestThread(long stepproduce, long stepconsume) : stepproduce_(stepproduce), stepconsume_(stepconsume) {
        run_produce_.store(true);
        run_consume_.store(true);
    }
    void produce() {
        while (run_produce_.load()) {
            produce_only_one();
        }
    }
    void consume() {
        lastconsume_ = 1;
        while (run_consume_.load()) {
            consume_only_one();
        }
    }
    void produce_only_one() {
        //                std::cerr << "produce\n";
        RingTest<Event_Gtest>::type_eventsadd vevents;
        for (int i = 0; i < stepproduce_; ++i) {
            ++lastproduce_;
            vevents.push_back(Event_Gtest{lastproduce_});
        }
        r_.add(vevents);
        wait_cond_.notify_all();

        //            }
    }
    void consume_only_one() {
        //            std::cerr << "consume\n";
        type_buffer buf;
        {
            std::unique_lock<std::mutex> lock(ring_mut_);
            while (run_consume_ && !r_.data_available(lastconsume_)) {
                wait_cond_.wait(lock);
                if (!run_consume_)
                    return;
            }
        }
        auto inserter = std::back_inserter(buf);
        r_.fill_buffer_to(inserter, lastconsume_);
        lastconsume_ += stepconsume_;
    }
    void exit_produce() {
        run_produce_ = false;
    }
    void exit_consume() {
        std::unique_lock<std::mutex> lock(ring_mut_);
        run_consume_ = false;
        wait_cond_.notify_all();
    }

    long get_produced() const {
        return lastproduce_;
    }
    long get_next_to_consume() const {
        return lastconsume_;
    }

    void stat_ring() {
        r_.stat_ring(std::cout);
    }

private:
    std::atomic<bool> run_produce_;
    std::atomic<bool> run_consume_;
    RingTest<Event_Gtest> r_;
    long stepproduce_ = 1;
    long stepconsume_ = 1;

    long lastproduce_ = 0;
    long lastconsume_ = 1;
    std::mutex ring_mut_;

    std::condition_variable wait_cond_;
};
} // namespace

TEST_F(Ring_GTest, test_thread_notrhead) {
    TestThread threadtest(3, 1);
    threadtest.produce_only_one();
    threadtest.consume_only_one();
    threadtest.consume_only_one();
    EXPECT_EQ(static_cast<long>(3), threadtest.get_next_to_consume());
}

TEST_F(Ring_GTest, test_thread) {
    TestThread threadtest(1, 3);
    std::thread thread_produce([&] { threadtest.produce(); });
    std::thread thread_consume([&] { threadtest.consume(); });

    std::this_thread::sleep_for(std::chrono::seconds(1));
    threadtest.exit_produce();
    threadtest.exit_consume();

    thread_produce.join();
    thread_consume.join();
    std::cout << "prod " << threadtest.get_produced() << " cons " << threadtest.get_next_to_consume() << std::endl;
    threadtest.stat_ring();
}
