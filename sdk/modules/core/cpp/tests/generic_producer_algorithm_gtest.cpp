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
#include <future>
#include <chrono>
#include <iterator>
#include <stdexcept>

#include "metavision/sdk/core/algorithms/generic_producer_algorithm.h"
#include "metavision/sdk/base/events/event2d.h"
#include "metavision/sdk/base/utils/timestamp.h"

using namespace Metavision;

class GenericProducerAlgorithm_GTest : public ::testing::Test {
protected:
    virtual void SetUp() override {}

    virtual void TearDown() override {}

    detail::Ring<Event2d> &get_ring() {
        return producer_algo_.ring_event_;
    }

    void assert_ring_size(size_t max_size) {
        std::unique_lock<std::mutex> lock(producer_algo_.underfilled_wait_mut_);
        std::unique_lock<std::mutex> lock2(producer_algo_.overfilled_wait_mut_);
        if (producer_algo_.ring_event_.data_available())
            ASSERT_GE(size_t(5), producer_algo_.ring_event_.get_last_time() - get_ring().get_first_time());
    }
    GenericProducerAlgorithm<Event2d> producer_algo_;
};

#ifndef ANDROID
// created bug issue TEAM-6380
TEST_F(GenericProducerAlgorithm_GTest, register_nothing) {
    std::vector<Event2d> events;
    ASSERT_EXIT((this->producer_algo_.register_new_event_buffer(events.begin(), events.end()), exit(0)),
                ::testing::ExitedWithCode(0), ".*");
}
#endif

TEST_F(GenericProducerAlgorithm_GTest, call_callback_and_produce_output) {
    std::vector<Event2d> events(10);
    int i = 0;
    for (auto &event : events) {
        event.t = i++;
    }
    this->producer_algo_.register_new_event_buffer(events.begin(), events.end());

    std::vector<Event2d> buffer;
    auto inserter = std::back_inserter(buffer);
    this->producer_algo_.process_events(5, inserter);

    ASSERT_EQ(5, buffer.size());
    i = 0;
    for (auto it = buffer.begin(); it != buffer.end(); ++it, ++i) {
        ASSERT_EQ(i, it->t);
    }

    buffer.clear();
    this->producer_algo_.process_events(9, inserter);

    ASSERT_EQ(4, buffer.size());

    for (auto it = buffer.begin(); it != buffer.end(); ++it, ++i) {
        ASSERT_EQ(i, it->t);
    }
}

TEST_F(GenericProducerAlgorithm_GTest, test_wait) {
    std::vector<Event2d> events(10);
    int i = 0;
    for (auto &event : events) {
        event.t = i++;
    }
    auto mid = std::lower_bound(events.begin(), events.end(), Event2d(0, 0, 0, 5),
                                [=](const Event2d &ev1, const Event2d &ev2) { return ev1.t < ev2.t; });
    // Put only half of the events, this means events up to time 4
    this->producer_algo_.register_new_event_buffer(events.begin(), mid);

    std::vector<Event2d> buffer;
    auto inserter = std::back_inserter(buffer);
    auto finished = std::async(std::launch::async, [&inserter, this] {
        // This should wait until event 9 is received
        this->producer_algo_.process_events(9, inserter);
        return true;
    });
    auto status   = finished.wait_for(std::chrono::seconds(1));
    ASSERT_EQ(std::future_status::timeout, status);

    // Put the rest of the events (up to time 9)
    this->producer_algo_.register_new_event_buffer(mid, events.end());

    status = finished.wait_for(std::chrono::seconds(1));
    ASSERT_EQ(std::future_status::ready, status);

    ASSERT_EQ(9, buffer.size());
}

TEST_F(GenericProducerAlgorithm_GTest, test_timeout) {
    this->producer_algo_.set_timeout(100000);

    std::vector<Event2d> events(10);
    int i = 0;
    for (auto &event : events) {
        event.t = i++;
    }
    this->producer_algo_.register_new_event_buffer(events.begin(), events.end());

    std::vector<Event2d> buffer;
    auto inserter = std::back_inserter(buffer);
    auto finished = std::async(std::launch::async, [&inserter, this] {
        // This should wait until event 9 is received
        this->producer_algo_.process_events(10, inserter);
        return true;
    });

    auto status = finished.wait_for(std::chrono::seconds(1));
    ASSERT_EQ(std::future_status::ready, status);

    ASSERT_EQ(10, buffer.size());
}

TEST_F(GenericProducerAlgorithm_GTest, test_is_done) {
    std::vector<Event2d> events(10);
    int i = 0;
    for (auto &event : events) {
        event.t = i++;
    }
    this->producer_algo_.register_new_event_buffer(events.begin(), events.end());

    std::vector<Event2d> buffer;
    auto inserter = std::back_inserter(buffer);
    this->producer_algo_.set_source_as_done();

    ASSERT_FALSE(this->producer_algo_.is_done());

    // This call should get 9 out of 10 events from the buffer
    this->producer_algo_.process_events(9, inserter);

    // One event is still left in the buffer
    ASSERT_FALSE(this->producer_algo_.is_done());

    // To get the last event we need to set a timeout such that
    // The last event can be retrieved.
    this->producer_algo_.process_events(10, inserter);

    ASSERT_TRUE(this->producer_algo_.is_done());
}

TEST_F(GenericProducerAlgorithm_GTest, test_max_events_per_second) {
    this->producer_algo_.set_max_events_per_second(500000);
    std::vector<Event2d> events(10);
    int i = 0;
    for (auto &event : events) {
        event.t = i++;
    }
    this->producer_algo_.register_new_event_buffer(events.begin(), events.end());

    std::vector<Event2d> buffer;
    auto inserter = std::back_inserter(buffer);
    this->producer_algo_.set_source_as_done();

    this->producer_algo_.process_events(10, inserter);

    ASSERT_EQ(5, buffer.size());
}
TEST_F(GenericProducerAlgorithm_GTest, test_no_max_duration) {
    this->producer_algo_.set_max_duration_stored(std::numeric_limits<timestamp>::max());
    std::vector<Event2d> events(10);
    int i = 0;
    for (auto &event : events) {
        event.t = i++;
    }
    this->producer_algo_.register_new_event_buffer(events.begin(), events.end());
    ASSERT_EQ(timestamp(0), get_ring().get_first_time());
    ASSERT_EQ(timestamp(9), get_ring().get_last_time());
}

TEST_F(GenericProducerAlgorithm_GTest, test_max_duration_drop_range_too_big) {
    this->producer_algo_.set_max_duration_stored(5);
    this->producer_algo_.set_allow_drop_when_overfilled(true);
    std::vector<Event2d> events(10);
    int i = 0;
    for (auto &event : events) {
        event.t = i++;
    }
    this->producer_algo_.register_new_event_buffer(events.begin(), events.end());
    ASSERT_EQ(timestamp(4), get_ring().get_first_time());
    ASSERT_EQ(timestamp(9), get_ring().get_last_time());
}

TEST_F(GenericProducerAlgorithm_GTest, test_max_duration_no_drop_throw) {
    this->producer_algo_.set_max_duration_stored(5);
    this->producer_algo_.set_allow_drop_when_overfilled(false);
    std::vector<Event2d> events(10);
    int i = 0;
    for (auto &event : events) {
        event.t = i++;
    }
    // try to insert events from t = 0 to t = 9
    // this should throw as the producer has a fixed capacity (5) and can not drop
    ASSERT_THROW(this->producer_algo_.register_new_event_buffer(events.begin(), events.end()), std::runtime_error);
}

TEST_F(GenericProducerAlgorithm_GTest, test_max_duration_no_drop_stop) {
    this->producer_algo_.set_max_duration_stored(8);
    this->producer_algo_.set_allow_drop_when_overfilled(false);
    std::vector<Event2d> events(10);
    int i = 0;
    for (auto &event : events) {
        event.t = i++;
    }
    this->producer_algo_.register_new_event_buffer(events.begin(), events.begin() + 3);

    // try to insert events from t = 3 to t = 9 on a separarate thread
    // this should block even if the capacity (8) is enough to insert the (7) events from
    // because the producer can not drop, and some events were already added
    auto finished_reg = std::async(std::launch::async, [this, &events] {
        this->producer_algo_.register_new_event_buffer(events.begin() + 3, events.end());
    });

    // so even if we wait a long time (1s), the other thread is blocked
    auto status = finished_reg.wait_for(std::chrono::seconds(1));
    ASSERT_EQ(std::future_status::timeout, status);

    this->producer_algo_.set_source_as_done();

    // ... the thread should have been unblocked
    status = finished_reg.wait_for(std::chrono::seconds(1));
    ASSERT_EQ(std::future_status::ready, status);
}

TEST_F(GenericProducerAlgorithm_GTest, test_max_duration_no_drop_range_ok_ring_empty) {
    this->producer_algo_.set_max_duration_stored(9);
    this->producer_algo_.set_allow_drop_when_overfilled(false);
    std::vector<Event2d> events(10);
    int i = 0;
    for (auto &event : events) {
        event.t = i++;
    }

    // try to insert events from t = 0 to t = 9
    // this should not block since the capacity (9) is enough
    this->producer_algo_.register_new_event_buffer(events.begin(), events.end());

    ASSERT_EQ(timestamp(0), get_ring().get_first_time());
    ASSERT_EQ(timestamp(9), get_ring().get_last_time());
}

TEST_F(GenericProducerAlgorithm_GTest, test_max_duration_no_drop_range_ok_ring_already_filled) {
    this->producer_algo_.set_max_duration_stored(8);
    this->producer_algo_.set_allow_drop_when_overfilled(false);
    std::vector<Event2d> events(10);
    int i = 0;
    for (auto &event : events) {
        event.t = i++;
    }
    this->producer_algo_.register_new_event_buffer(events.begin(), events.begin() + 3);

    // try to insert events from t = 3 to t = 9 on a separarate thread
    // this should block even if the capacity (8) is enough to insert the (7) events from
    // because the producer can not drop, and some events were already added
    auto finished_reg = std::async(std::launch::async, [this, &events] {
        this->producer_algo_.register_new_event_buffer(events.begin() + 3, events.end());
    });

    // so even if we wait a long time (1s), the other thread is blocked
    auto status = finished_reg.wait_for(std::chrono::seconds(1));
    ASSERT_EQ(std::future_status::timeout, status);

    // process up to t=4 ...
    std::vector<Event2d> evts;
    this->producer_algo_.process_events(2, std::back_inserter(evts));

    // ... the thread should have been unblocked
    status = finished_reg.wait_for(std::chrono::seconds(1));
    ASSERT_EQ(std::future_status::ready, status);
    ASSERT_EQ(timestamp(2), get_ring().get_first_time());
    ASSERT_EQ(timestamp(9), get_ring().get_last_time());
}

TEST_F(GenericProducerAlgorithm_GTest, test_max_duration_drop_big) {
    this->producer_algo_.set_max_duration_stored(5);
    this->producer_algo_.set_allow_drop_when_overfilled(true);

    std::vector<Event2d> events(1001), events2;
    int i = 0;
    for (auto &event : events) {
        event.t = i++;
    }

    this->producer_algo_.register_new_event_buffer(events.begin(), events.end());
    this->producer_algo_.process_events(1000, std::back_inserter(events2));

    ASSERT_EQ(timestamp(995), events2.front().t);
    ASSERT_EQ(timestamp(999), events2.back().t);
    ASSERT_EQ(timestamp(1000), get_ring().get_first_time());
    ASSERT_EQ(timestamp(1000), get_ring().get_last_time());
}

TEST_F(GenericProducerAlgorithm_GTest, test_max_duration_no_drop_big) {
    this->producer_algo_.set_max_duration_stored(5);
    this->producer_algo_.set_allow_drop_when_overfilled(false);

    std::vector<Event2d> events(1000), events2;
    int i = 0;
    for (auto &event : events) {
        event.t = i++;
    }

    std::atomic<bool> t1_done{false}, t2_done{false};
    std::thread t1([&events, &t1_done, this] {
        for (int i = 0; i <= 1000 - 5; i += 5) {
            this->producer_algo_.register_new_event_buffer(events.begin() + i, events.begin() + i + 5);
        }
        t1_done = true;
    });
    std::thread t2([&events2, &t2_done, this] {
        for (timestamp t = 1; t < 1000; ++t) {
            this->producer_algo_.process_events(t, std::back_inserter(events2));
        }
        t2_done = true;
    });
    while (!t1_done || !t2_done) {
        assert_ring_size(size_t(5));
    }
    t1.join();
    t2.join();
    ASSERT_EQ(timestamp(0), events2.front().t);
    ASSERT_EQ(timestamp(998), events2.back().t);
    ASSERT_EQ(timestamp(999), get_ring().get_first_time());
    ASSERT_EQ(timestamp(999), get_ring().get_last_time());
}

// test avec ring empty pour tester start != end, range OK,
