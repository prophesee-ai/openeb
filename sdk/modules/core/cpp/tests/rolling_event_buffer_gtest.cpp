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

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/core/utils/rolling_event_buffer.h"

struct FooEvent {
    Metavision::timestamp t;
};

using RollingEventBuffer       = Metavision::RollingEventBuffer<FooEvent>;
using RollingEventBufferConfig = Metavision::RollingEventBufferConfig;

void insert_events(RollingEventBuffer &buffer, const std::vector<FooEvent> &events) {
    buffer.insert_events(events.cbegin(), events.cend());
}

void check_rolling_buffer_content(const RollingEventBuffer &buffer, const std::vector<FooEvent> &gt) {
    // GIVEN a rolling buffer and its GT
    ASSERT_EQ(gt.size(), buffer.size());

    // WHEN we compare the buffer to the GT using direct access
    // THEN all the elements match
    for (size_t i = 0; i < buffer.size(); ++i) {
        ASSERT_EQ(buffer[i].t, gt[i].t);
    }

    // WHEN we compare the buffer to the GT using iterators
    // THEN all the elements match
    auto gt_it = gt.cbegin();
    for (const auto &ev : buffer) {
        ASSERT_EQ(ev.t, gt_it->t);
        ++gt_it;
    }
}

void check_iterators(const RollingEventBuffer &buffer) {
    // GIVEN a non-empty rolling buffer
    // WHEN we retrieve an iterator to the first and last element
    auto begin          = buffer.cbegin();
    auto end            = buffer.cend();
    auto last           = std::prev(end);
    const auto distance = std::distance(begin, last);

    // THEN
    // - the iterator pointing to the first element is ordered before the iterator pointing to the last element
    // - the distance between the two iterators is consistent with the size of the buffer
    // - incrementing the first iterator by this distance moves it to the second iterator
    // - decrementing the second iterator by this distance moves it to the first iterator
    ASSERT_TRUE(begin != end);
    ASSERT_TRUE(begin != last);
    ASSERT_TRUE(begin < last);
    ASSERT_TRUE(begin <= last);
    ASSERT_FALSE(begin > last);
    ASSERT_FALSE(begin >= last);
    ASSERT_EQ(buffer.size(), distance + 1);
    ASSERT_EQ(begin + distance, last);
    ASSERT_EQ(last - distance, begin);
}

TEST(RollingEventBuffer, n_events_empty_buffer) {
    // GIVEN an empty rolling buffer
    RollingEventBuffer buffer;

    // WHEN we check its size
    // THEN the result is 0
    ASSERT_TRUE(buffer.empty());
    ASSERT_EQ(0, buffer.size());
}

TEST(RollingEventBuffer, n_events_no_warp_up) {
    // GIVEN a rolling buffer configured to store at most 5 events
    RollingEventBuffer buffer(RollingEventBufferConfig::make_n_events(5));

    // WHEN we insert 5 events
    const std::vector<FooEvent> events_gt{{0}, {1}, {2}, {3}, {4}};
    insert_events(buffer, events_gt);

    // THEN
    // - the buffer is not empty
    // - its content matches the GT
    // - the iterators are consistent
    ASSERT_FALSE(buffer.empty());
    check_rolling_buffer_content(buffer, events_gt);
    check_iterators(buffer);
}

TEST(RollingEventBuffer, n_events_warp_up) {
    // GIVEN a rolling buffer configured to store at most 5 events
    RollingEventBuffer buffer(RollingEventBufferConfig::make_n_events(5));

    // WHEN we insert 8 events
    insert_events(buffer, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}});

    // THEN
    // - the buffer is not empty
    // - its content matches the GT
    // - the iterators are consistent
    ASSERT_FALSE(buffer.empty());
    check_rolling_buffer_content(buffer, {{3}, {4}, {5}, {6}, {7}});
    check_iterators(buffer);
}

TEST(RollingEventBuffer, n_us_empty_buffer) {
    // GIVEN an empty rolling buffer configured to store 10us of events
    RollingEventBuffer buffer(RollingEventBufferConfig::make_n_us(10));

    // WHEN we check its size
    // THEN the result is 0
    ASSERT_TRUE(buffer.empty());
    ASSERT_EQ(0, buffer.size());
}

TEST(RollingEventBuffer, n_us_no_warp_up) {
    // GIVEN an empty rolling buffer configured to store 10us of events
    RollingEventBuffer buffer(RollingEventBufferConfig::make_n_us(10));

    // WHEN we insert 5 events corresponding to a slice of 10us
    const std::vector<FooEvent> events_gt{{0}, {3}, {6}, {9}, {10}};

    insert_events(buffer, events_gt);

    // THEN
    // - the buffer is not empty
    // - its content matches the GT
    // - the iterators are consistent
    ASSERT_FALSE(buffer.empty());
    check_rolling_buffer_content(buffer, events_gt);
    check_iterators(buffer);
}

TEST(RollingEventBuffer, n_us_warp_up) {
    // GIVEN an empty rolling buffer configured to store 10us of events
    RollingEventBuffer buffer(RollingEventBufferConfig::make_n_us(10));

    // WHEN we first insert 5 events corresponding to a slice of 10us and then 2 more events corresponding to 5us of
    // data (probable memory layout [14, 15, 6, 9, 10])
    const std::vector<FooEvent> events_gt{{0}, {3}, {6}, {9}, {10}};
    insert_events(buffer, events_gt);
    insert_events(buffer, {{14}, {15}});

    // THEN
    // - no reallocation is done
    // - only the 5 events corresponding to the last 10us of data are kept
    // - the iterators are consistent
    ASSERT_EQ(5, buffer.capacity());
    check_rolling_buffer_content(buffer, {{6}, {9}, {10}, {14}, {15}});
    check_iterators(buffer);

    // WHEN we insert 5 events corresponding to a slice of 10us (probable memory layout [10, 14, 15, 16, 17, 18, 19,
    // 20])
    insert_events(buffer, {{16}, {17}, {18}, {19}, {20}});

    // THEN
    // - a reallocation is done
    // - only the 8 events corresponding to the last 10us of data are kept
    // - the iterators are consistent
    ASSERT_EQ(8, buffer.capacity());
    check_rolling_buffer_content(buffer, {{10}, {14}, {15}, {16}, {17}, {18}, {19}, {20}});
    check_iterators(buffer);

    // WHEN we insert 3 events corresponding to a slice of 6us (probable memory layout [21, 22, 26, 16, 17, 18, 19, 20])
    insert_events(buffer, {{21}, {22}, {26}});

    // THEN
    // - no reallocation is done
    // - only the 8 events corresponding to the last 10us of data are kept
    // - the iterators are consistent
    check_rolling_buffer_content(buffer, {{16}, {17}, {18}, {19}, {20}, {21}, {22}, {26}});
    check_iterators(buffer);

    // WHEN we insert 3 events corresponding to a slice of 11us (probable memory layout [X, X, X, 35, 36, 37, X, X]
    insert_events(buffer, {{35}, {36}, {37}});

    // THEN
    // - no reallocation is done
    // - only the 3 events corresponding to the last 10us of data are kept
    // - the iterators are consistent
    ASSERT_EQ(8, buffer.capacity());
    check_rolling_buffer_content(buffer, {{35}, {36}, {37}});
    check_iterators(buffer);

    // WHEN we insert 2 events corresponding to a slice of 3us (probable memory layout [X, X, X, 35, 36, 37, 38, 39]
    insert_events(buffer, {{38}, {39}});

    // THEN
    // - no reallocation is done
    // - only the 5 events corresponding to the last 10us of data are kept
    // - the iterators are consistent
    ASSERT_EQ(8, buffer.capacity());
    check_rolling_buffer_content(buffer, {{35}, {36}, {37}, {38}, {39}});
    check_iterators(buffer);
}