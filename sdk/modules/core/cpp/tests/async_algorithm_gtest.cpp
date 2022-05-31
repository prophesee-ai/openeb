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
#include <functional>
#include <gtest/gtest.h>

#include "metavision/sdk/core/algorithms/async_algorithm.h"
#include "metavision/sdk/base/events/event_cd.h"

using namespace Metavision;

using SizeType = std::vector<size_t>::size_type;

class AsyncAlgorithmImpl : public AsyncAlgorithm<AsyncAlgorithmImpl> {
public:
    using State  = std::vector<Event2d>;
    using TestCb = std::function<void(const State &)>;

    AsyncAlgorithmImpl() {}

    /// @brief Function to process events online
    template<class InputIt>
    inline void process_online(InputIt it_begin, InputIt it_end) {
        events_.insert(events_.cend(), it_begin, it_end);
    }

    /// @brief Function to process the state that is called every n_events or n_us
    void process_async(const timestamp processing_ts, const size_t n_processed_events) {
        current_processing_ts_us_ = processing_ts;
        processed_events_count_   = n_processed_events;
        test_cb_(events_);
        events_.clear();
    }

    void set_callback(const TestCb &test_cb) {
        test_cb_ = test_cb;
    }

    void on_init(const timestamp processing_ts) {
        current_processing_ts_us_ = processing_ts;
    }

    timestamp current_processing_ts_us_{0};
    size_t processed_events_count_{0};

private:
    State events_;
    TestCb test_cb_;
};

class AsyncAlgorithm_GTest : public ::testing::Test {
public:
    AsyncAlgorithm_GTest() {}

protected:
    AsyncAlgorithmImpl algo_;
};

TEST_F(AsyncAlgorithm_GTest, synch) {
    // GIVEN 10 consecutive events with increasing timestamps from t = 0 to 9
    const int n_events = 10;
    std::vector<Event2d> events;
    for (int i = 0; i < n_events; ++i)
        events.push_back(Event2d(0, 0, 0, i));

    // WHEN we process in SYNC mode the full buffer of events
    std::vector<size_t> buffer_sizes;
    const auto cb = [&](const std::vector<Event2d> &buffer) {
        buffer_sizes.push_back(buffer.size());
        ASSERT_EQ(10, algo_.processed_events_count_);
    };
    algo_.set_callback(cb);
    algo_.process_events(events.cbegin(), events.cend());

    // THEN we get only one output buffer, containing the n events corresponding to the time slice [0, 10[
    ASSERT_EQ(1, buffer_sizes.size());
    ASSERT_EQ(n_events, buffer_sizes[0]);
    ASSERT_EQ(events.back().t + 1, algo_.current_processing_ts_us_);
}

TEST_F(AsyncAlgorithm_GTest, step_1) {
    // GIVEN 10 consecutive events with increasing timestamps from t = 0 to 9
    const int n_events = 10;
    std::vector<Event2d> events;
    for (int i = 0; i < n_events; ++i)
        events.push_back(Event2d(0, 0, 0, i));

    // WHEN we process in N_EVENTS mode the full buffer of events
    // with a step of 1 event
    const int step = 1;
    algo_.set_processing_n_events(step);

    int num_async_processes = 0, sum_ev = 0;
    const auto cb = [&](const std::vector<Event2d> &buffer) {
        num_async_processes++;
        sum_ev += buffer.size();
        ASSERT_EQ(step, algo_.processed_events_count_);
    };

    algo_.set_callback(cb);
    algo_.process_events(events.cbegin(), events.cend());

    // THEN we get 10 output buffers, each containing 1 event
    ASSERT_EQ(n_events, sum_ev);
    ASSERT_EQ(10, num_async_processes);
    ASSERT_EQ(events.back().t, algo_.current_processing_ts_us_);
}

TEST_F(AsyncAlgorithm_GTest, step_10) {
    // GIVEN 50 consecutive events with increasing timestamps from t = 0 to 49
    const int n_events = 50;
    std::vector<Event2d> events;
    for (int i = 0; i < n_events; ++i)
        events.push_back(Event2d(0, 0, 0, i));

    // WHEN we process in N_EVENTS mode with a step of 10 events
    // first the 25 first events and then the 25 remaining ones
    const int step = 10, middle = n_events / 2;
    algo_.set_processing_n_events(step);
    int sum_ev = 0;
    std::vector<int> buffer_sizes_0, buffer_sizes_1;
    const auto cb_0 = [&](const std::vector<Event2d> &buffer) {
        buffer_sizes_0.push_back(buffer.size());
        sum_ev += buffer.size();
        ASSERT_EQ(step, algo_.processed_events_count_);
    };
    const auto cb_1 = [&](const std::vector<Event2d> &buffer) {
        buffer_sizes_1.push_back(buffer.size());
        sum_ev += buffer.size();
        ASSERT_EQ(step, algo_.processed_events_count_);
    };
    algo_.set_callback(cb_0);
    algo_.process_events(events.cbegin(), events.cbegin() + middle);
    algo_.set_callback(cb_1);
    algo_.process_events(events.cbegin() + middle, events.cend());

    // THEN we get 5 output buffers (2 and 3 for the first and second process),
    // each containing 10 events
    ASSERT_EQ(n_events, sum_ev);
    ASSERT_EQ(events.back().t, algo_.current_processing_ts_us_);

    ASSERT_EQ(2, buffer_sizes_0.size());
    for (const auto &s : buffer_sizes_0)
        ASSERT_EQ(step, s);

    ASSERT_EQ(3, buffer_sizes_1.size());
    for (const auto &s : buffer_sizes_1)
        ASSERT_EQ(step, s);
}

TEST_F(AsyncAlgorithm_GTest, us_1) {
    // GIVEN 50 consecutive events with increasing timestamps from t = 0 to 49
    const int n_events = 50;
    std::vector<Event2d> events;
    for (int i = 0; i < n_events; ++i)
        events.push_back(Event2d(0, 0, 0, i));

    // WHEN we process in N_US mode with a step of 1us the full buffer of events
    const int step_us = 1;
    algo_.set_processing_n_us(step_us);
    std::vector<size_t> buffer_sizes;
    std::vector<timestamp> timestamps;
    int sum_ev    = 0;
    const auto cb = [&](const std::vector<Event2d> &buffer) {
        buffer_sizes.push_back(buffer.size());
        timestamps.push_back(algo_.current_processing_ts_us_);
        sum_ev += buffer.size();
        ASSERT_EQ(1, algo_.processed_events_count_);
    };
    algo_.set_callback(cb);
    algo_.process_events(events.cbegin(), events.cend());

    // THEN we get 49 output buffers, each containing 1 event and the last event is waiting for new one to trigger the
    // asynchronous process.
    // Note about AsyncAlgo n_us policy -> process async is called if and only if the timestamp of the last event
    // processed is >= to the processing ts.
    ASSERT_EQ(49, sum_ev);
    ASSERT_EQ(49, buffer_sizes.size());
    ASSERT_EQ(49, timestamps.back());
    ASSERT_EQ(49, algo_.current_processing_ts_us_);

    for (SizeType i = 0; i < buffer_sizes.size(); ++i)
        ASSERT_EQ(1, buffer_sizes[i]);

    ASSERT_EQ(49, timestamps.size());
    int t = 1;
    for (auto it = timestamps.cbegin(); it != timestamps.cend(); it++, t++)
        ASSERT_EQ(t, *it);

    // WHEN flushing, process async is called to process all remaining events and so the last timeslice (50) is
    // processed
    algo_.flush();
    ASSERT_EQ(50, sum_ev);
    ASSERT_EQ(50, timestamps.size());
    ASSERT_EQ(50, timestamps.back());
    ASSERT_EQ(50, algo_.current_processing_ts_us_);
}

TEST_F(AsyncAlgorithm_GTest, us_10) {
    // GIVEN 50 consecutive events with increasing timestamps from t = 0 to 49
    const int n_events = 50;
    std::vector<Event2d> events;
    for (int i = 0; i < n_events; ++i)
        events.push_back(Event2d(0, 0, 0, i));

    // WHEN we process in N_US mode with a step of 10us the full buffer of events
    const int step_us = 10;
    algo_.set_processing_n_us(step_us);
    std::vector<size_t> buffer_sizes;
    std::vector<timestamp> timestamps;
    int sum_ev    = 0;
    const auto cb = [&](const std::vector<Event2d> &buffer) {
        buffer_sizes.push_back(buffer.size());
        timestamps.push_back(algo_.current_processing_ts_us_);
        sum_ev += buffer.size();
        ASSERT_EQ(10, algo_.processed_events_count_);
    };
    algo_.set_callback(cb);
    algo_.process_events(events.cbegin(), events.cend());

    // THEN we get 4 output buffers, each containing 10 events
    // The algo waiting for an event with ts > 49 to call the async process
    ASSERT_EQ(40, sum_ev);
    ASSERT_EQ(4, buffer_sizes.size());
    ASSERT_EQ(40, timestamps.back());
    ASSERT_EQ(40, algo_.current_processing_ts_us_);

    for (SizeType i = 0; i < buffer_sizes.size(); ++i)
        ASSERT_EQ(10, buffer_sizes[i]);

    ASSERT_EQ(4, timestamps.size());
    int t = 10;
    for (auto it = timestamps.cbegin(); it != timestamps.cend(); it++, t += step_us)
        ASSERT_EQ(t, *it);

    // WHEN flushing, process async is called to process all remaining events and so the last timeslice (50) is
    // processed
    algo_.flush();
    ASSERT_EQ(50, sum_ev);
    ASSERT_EQ(5, timestamps.size());
    ASSERT_EQ(50, timestamps.back());
    ASSERT_EQ(50, algo_.current_processing_ts_us_);
}

TEST_F(AsyncAlgorithm_GTest, mixed) {
    // GIVEN 50 consecutive events with increasing timestamps from t = 0 to 49,
    // followed by 50 events, the timestamps of which form an arithmetic sequence
    // of common difference 4 (50, 54, ... 242, 246)
    const int n_events = 50, delta_us = 4;
    std::vector<Event2d> events;
    for (int i = 0; i < n_events; ++i)
        events.push_back(Event2d(0, 0, 0, i));
    for (int i = 0; i < n_events; ++i)
        events.push_back(Event2d(0, 0, 0, n_events + delta_us * i));

    // WHEN we process in N_MIXED mode with steps of 10events and 20us
    const int step = 10, step_us = 20, middle = 35;
    const int n_step = n_events / step;
    algo_.set_processing_mixed(step, step_us);
    std::vector<int> buffer_sizes_0, buffer_sizes_1;
    std::vector<timestamp> timestamps_0, timestamps_1;
    int sum_ev                       = 0;
    int n                            = 0;
    timestamp last_step_condition_ts = 0;
    const auto cb                    = [&, n_step](const std::vector<Event2d> &buffer) {
        sum_ev += buffer.size();
        if (n < n_step) {
            // N_EVENTS behaviour: here, we have met the event count condition (i.e. event processed since last async
            // call == step) before the time slice one. The time slice must correspond to the last event processed
            // timestamp and the size of the buffer must be the size of step.
            buffer_sizes_0.push_back(buffer.size());
            timestamps_0.push_back(buffer.empty() ? -1 : buffer.back().t);
            ASSERT_EQ(step, algo_.processed_events_count_);
            ASSERT_EQ(timestamps_0.back(), algo_.current_processing_ts_us_);
            last_step_condition_ts = algo_.current_processing_ts_us_;
        } else {
            // N_US behaviour: here, we have met the time slice condition before the event count one. The time slice
            // must be n times * step_us away from the last event buffer timestamp that has met the event count
            // condition.
            buffer_sizes_1.push_back(buffer.size());
            timestamps_1.push_back(buffer.empty() ? -1 : buffer.back().t);
            ASSERT_EQ(buffer.size(), algo_.processed_events_count_);
            ASSERT_LT(buffer.size(), step);
            ASSERT_EQ(last_step_condition_ts + (n + 1 - n_step) * step_us, algo_.current_processing_ts_us_);
        }
        n++;
    };
    algo_.set_callback(cb);
    algo_.process_events(events.cbegin(), events.cbegin() + middle);
    algo_.process_events(events.cbegin() + middle, events.cend());

    // THEN we first get 5 output buffers, each containing 10 events,
    // and then 9 output buffers, each containing 5 events
    ASSERT_EQ(5, buffer_sizes_0.size());
    for (const auto &s : buffer_sizes_0)
        ASSERT_EQ(step, s);

    ASSERT_EQ(5, timestamps_0.size());
    int t0 = step - 1;
    for (auto it = timestamps_0.cbegin(); it != timestamps_0.cend(); it++, t0 += step)
        ASSERT_EQ(t0, *it);

    ASSERT_EQ(9, buffer_sizes_1.size());
    const int buffer_size_us = step_us / delta_us;
    for (const auto &s : buffer_sizes_1)
        ASSERT_EQ(buffer_size_us, s);

    ASSERT_EQ(9, timestamps_1.size());
    int t1 = n_events + step_us - delta_us; // 66
    for (auto it = timestamps_1.cbegin(); it != timestamps_1.cend(); it++, t1 += step_us)
        ASSERT_EQ(t1, *it);
};

TEST_F(AsyncAlgorithm_GTest, external) {
    // GIVEN 50 consecutive events with increasing timestamps from t = 0 to 49
    const int n_events = 50;
    std::vector<Event2d> events;
    for (int i = 0; i < n_events; ++i)
        events.push_back(Event2d(0, 0, 0, i));

    // WHEN we process in EXTERNAL mode
    const int middle = n_events / 2;
    algo_.set_processing_external();
    std::vector<int> buffer_sizes;
    const auto cb = [&](const std::vector<Event2d> &buffer) {
        buffer_sizes.push_back(buffer.size());
        ASSERT_EQ(buffer.size(), algo_.processed_events_count_);
    };
    algo_.set_callback(cb);

    // THEN ts and event count follow what have been sent to the processing but process async is never called
    algo_.process_events(events.cbegin(), events.cbegin() + middle);
    ASSERT_EQ(0, buffer_sizes.size());
    ASSERT_EQ(0, algo_.current_processing_ts_us_);
    ASSERT_EQ(0, algo_.processed_events_count_);

    algo_.process_events(events.cbegin() + middle, events.cend());
    ASSERT_EQ(0, buffer_sizes.size());
    ASSERT_EQ(0, algo_.current_processing_ts_us_);
    ASSERT_EQ(0, algo_.processed_events_count_);

    // WHEN we flush, process async is called.
    algo_.flush();

    // THEN we get a single call to process async corresponding to the time slice [0, 50[
    ASSERT_EQ(1, buffer_sizes.size());
    ASSERT_EQ(events.back().t + 1, algo_.current_processing_ts_us_);
    ASSERT_EQ(events.size(), algo_.processed_events_count_);
};

TEST_F(AsyncAlgorithm_GTest, timeshift_round_ts) {
    // GIVEN a large timeshift of 10s and 50 consecutive events with increasing timestamps
    // from t = 10000000 to 10000049
    const int n_events = 50, timeshift = 10e6;
    std::vector<Event2d> events;
    for (int i = timeshift; i < timeshift + n_events; ++i)
        events.push_back(Event2d(0, 0, 0, i));

    // WHEN we process in N_US mode with a step of 10us the full buffer of events
    const int step_us = 10;
    algo_.set_processing_n_us(step_us);
    std::vector<size_t> buffer_sizes;
    std::vector<timestamp> timestamps;
    int sum_ev    = 0;
    const auto cb = [&](const std::vector<Event2d> &buffer) {
        buffer_sizes.push_back(buffer.size());
        timestamps.push_back(algo_.current_processing_ts_us_);
        sum_ev += buffer.size();
    };
    algo_.set_callback(cb);
    algo_.process_events(events.cbegin(), events.cend());

    // THEN we get 4 output buffers, each containing 10 events. The first buffer ts being timeshift + step. The last
    // buffer not being pcreated because async algo wait an events we greater timestamp to trig the process online
    ASSERT_EQ(40, sum_ev);
    ASSERT_EQ(4, buffer_sizes.size());
    for (SizeType i = 0; i < buffer_sizes.size(); ++i)
        ASSERT_EQ(10, buffer_sizes[i]);

    ASSERT_EQ(4, timestamps.size());
    int t = timeshift + step_us;
    for (auto it = timestamps.cbegin(); it != timestamps.cend(); it++, t += step_us)
        ASSERT_EQ(t, *it);
}

TEST_F(AsyncAlgorithm_GTest, timeshift_not_round_ts) {
    // GIVEN a large time shift of 10s, a first timestamp at 3us and 50 consecutive events with increasing timestamps
    // from t = 10000003 to 10000052
    const int n_events = 50, timeshift = 10e6, ts_first = 3;
    std::vector<Event2d> events;
    const int shifted_ts_first = timeshift + ts_first;
    for (int i = 0; i < n_events; ++i)
        events.push_back(Event2d(0, 0, 0, shifted_ts_first + i));

    // WHEN we process in N_US mode with a step of 10us the full buffer of events
    const int step_us = 10;
    algo_.set_processing_n_us(step_us);
    std::vector<size_t> buffer_sizes;
    std::vector<timestamp> timestamps;
    int sum_ev    = 0;
    const auto cb = [&](const std::vector<Event2d> &buffer) {
        buffer_sizes.push_back(buffer.size());
        timestamps.push_back(algo_.current_processing_ts_us_);
        sum_ev += buffer.size();
    };
    algo_.set_callback(cb);
    algo_.process_events(events.cbegin(), events.cend());

    // THEN the first processing timestamp is not set to the timestamp of the first event, but it's set to the next
    // multiple of step_us. Number of events processed is the 7 first (from 3 to 9 in the first slice, then 4 full
    // time slices of 10 events, the last times slice being incomplete)
    ASSERT_EQ(7 + 4 * 10, sum_ev);

    ASSERT_EQ(5, buffer_sizes.size());
    ASSERT_EQ(7, buffer_sizes[0]);
    for (SizeType i = 1; i < buffer_sizes.size(); ++i)
        ASSERT_EQ(10, buffer_sizes[i]);

    ASSERT_EQ(5, timestamps.size());
    int t = timeshift + step_us;
    for (auto it = timestamps.cbegin(); it != timestamps.cend(); it++, t += step_us)
        ASSERT_EQ(t, *it);

    algo_.flush();
    ASSERT_EQ(50, sum_ev);
    ASSERT_EQ(6, buffer_sizes.size());
    ASSERT_EQ(3, buffer_sizes.back());
    ASSERT_EQ(6, timestamps.size());
    ASSERT_EQ(events.back().t + 1, timestamps.back());
}

TEST_F(AsyncAlgorithm_GTest, explicit_end_ts) {
    // GIVEN 6 events distributed over 10us time intervals such that:
    //  0-10us : Empty          50-60us  : 1 event
    // 10-20us : Empty          60-70us  : Empty
    // 20-30us : Empty          70-80us  : 2 events
    // 30-40us : Empty          80-90us  : Empty
    // 40-50us : Empty          90-100us : 3 events
    std::vector<Event2d> events = {Event2d(0, 0, 0, 51),
                                   // Empty slice.
                                   Event2d(0, 0, 0, 71), Event2d(0, 0, 0, 72),
                                   // Empt slice.
                                   Event2d(0, 0, 0, 91), Event2d(0, 0, 0, 92), Event2d(0, 0, 0, 99)};

    // WHEN we process in N_US mode with a step of 10us, set the first processing timestamp to 0
    // and use the variant of the process_events method that requires the timestamp
    const int step_us = 10;
    algo_.set_processing_n_us(step_us);
    std::vector<size_t> buffer_sizes;
    std::vector<timestamp> timestamps;
    int sum_ev    = 0;
    const auto cb = [&](const std::vector<Event2d> &buffer) {
        buffer_sizes.push_back(buffer.size());
        timestamps.push_back(algo_.current_processing_ts_us_);
        sum_ev += buffer.size();
    };
    algo_.set_callback(cb);
    algo_.process_events(events.cbegin(), events.cend());

    // THEN the first processing timestamp is not set to the timestamp of the first event, but it's set to the next
    // multiple of step_us. 3 buffers are output with 1 event each. Events with ts from 91 to 99 belong to the same
    // slice. This slice waits for a ts >= 100 to trigger the async pocess.
    ASSERT_EQ(3, sum_ev);

    ASSERT_EQ(4, buffer_sizes.size());
    ASSERT_EQ(1, buffer_sizes[0]);
    ASSERT_EQ(0, buffer_sizes[1]);
    ASSERT_EQ(2, buffer_sizes[2]);
    ASSERT_EQ(0, buffer_sizes[3]);

    ASSERT_EQ(4, timestamps.size());
    ASSERT_EQ(60, timestamps[0]);
    ASSERT_EQ(70, timestamps[1]);
    ASSERT_EQ(80, timestamps[2]);
    ASSERT_EQ(90, timestamps[3]);

    algo_.flush();
    ASSERT_EQ(6, sum_ev);
    ASSERT_EQ(5, buffer_sizes.size());
    ASSERT_EQ(5, timestamps.size());
    ASSERT_EQ(3, buffer_sizes.back());
    ASSERT_EQ(100, timestamps.back());
}
