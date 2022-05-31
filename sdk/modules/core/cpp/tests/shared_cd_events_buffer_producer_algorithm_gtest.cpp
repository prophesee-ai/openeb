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

#include <random>
#include <vector>

#include "metavision/sdk/core/algorithms/shared_cd_events_buffer_producer_algorithm.h"

struct SharedCdBufferEvent {
    Metavision::timestamp t;
    Metavision::SharedCdEventsBufferProducerAlgorithm::SharedEventsBuffer data_;
};

class SharedCdEventsBufferProducer_Gtest : public ::testing::Test {
protected:
    virtual void SetUp() {}

    virtual void TearDown() {}
};

TEST_F(SharedCdEventsBufferProducer_Gtest, shared_cd_buffer_producer_0_size_fails) {
    // GIVEN A buffer pool of size 0
    Metavision::SharedEventsBufferProducerParameters params;
    params.buffers_pool_size_ = 0;

    // WHEN trying to allocate
    // THEN it throws
    ASSERT_THROW(
        Metavision::SharedCdEventsBufferProducerAlgorithm(params, [&](Metavision::timestamp ts, const auto &ev) {}),
        std::invalid_argument);
}

TEST_F(SharedCdEventsBufferProducer_Gtest, unbounded_shared_cd_buffer_producer_0_size_allocates_new_buffer) {
    // GIVEN A buffer pool of size 10 with pre-allocation of the buffers to size 10 using n_events policy of 1
    Metavision::SharedEventsBufferProducerParameters params;
    params.buffers_pool_size_          = 10;
    params.buffers_preallocation_size_ = 10;
    params.bounded_memory_pool_        = false;

    std::vector<SharedCdBufferEvent> produced;

    // WHEN a buffer is produced
    Metavision::SharedCdEventsBufferProducerAlgorithm producer(params, [&](Metavision::timestamp ts, const auto &ev) {
        produced.push_back({ts, ev});
    });

    producer.set_processing_n_events(1);

    std::vector<Metavision::Event2d> data{{0, 0, 0, 0}, {0, 0, 0, 2}, {0, 0, 0, 4}, {0, 0, 0, 5}};
    producer.process_events(data.cbegin(), data.cend());
    ASSERT_EQ(4, produced.size());
    // THEN the capacity of the produced vector is equal to the one set as input (10) even though the number of
    // events within is less than the capacity
    for (const auto &buffer : produced) {
        ASSERT_GE(10, buffer.data_->capacity());
        ASSERT_EQ(1, buffer.data_->size());
    }
}

TEST_F(SharedCdEventsBufferProducer_Gtest, shared_cd_buffer_producer_prealloc) {
    // GIVEN A buffer pool of size 10 with pre-allocation of the buffers to size 10 using n_events policy of 1
    Metavision::SharedEventsBufferProducerParameters params;
    params.buffers_pool_size_          = 10;
    params.buffers_preallocation_size_ = 10;

    std::vector<SharedCdBufferEvent> produced;

    // WHEN a buffer is produced
    Metavision::SharedCdEventsBufferProducerAlgorithm producer(params, [&](Metavision::timestamp ts, const auto &ev) {
        produced.push_back({ts, ev});
    });

    producer.set_processing_n_events(1);

    std::vector<Metavision::Event2d> data{{0, 0, 0, 0}, {0, 0, 0, 2}, {0, 0, 0, 4}, {0, 0, 0, 5}};
    producer.process_events(data.cbegin(), data.cend());
    ASSERT_EQ(4, produced.size());
    // THEN the capacity of the produced vector is equal to the one set as input (10) even though the number of
    // events within is less than the capacity
    for (const auto &buffer : produced) {
        ASSERT_EQ(10, buffer.data_->capacity());
        ASSERT_EQ(1, buffer.data_->size());
    }
}

TEST_F(SharedCdEventsBufferProducer_Gtest, shared_cd_buffer_producer_time_slice_policy) {
    // GIVEN A buffer pool of size 10 with pre-allocation of the buffers to size 10 and a time slice producing policy
    // (5ms of events/buffer)
    Metavision::SharedEventsBufferProducerParameters params;
    params.buffers_pool_size_          = 10;
    params.buffers_preallocation_size_ = 10;
    params.buffers_time_slice_us_      = 5000;
    params.buffers_events_count_       = 0;

    // WHEN processing a full buffer of events containing 2 full time slices and one not full (5000, 10000)
    const std::vector<Metavision::Event2d> data{{0, 0, 0, 0},    {0, 0, 0, 4999},  {0, 0, 0, 5000}, {0, 0, 0, 5001},
                                                {0, 0, 0, 9999}, {0, 0, 0, 10000}, {0, 0, 0, 10001}};
    std::vector<SharedCdBufferEvent> produced;
    Metavision::SharedCdEventsBufferProducerAlgorithm producer(params, [&](Metavision::timestamp ts, const auto &ev) {
        produced.push_back({ts, ev});
    });

    producer.process_events(data.cbegin(), data.cend());

    // THEN 2 buffers are produced with all the events that belong to the expected time slice
    ASSERT_EQ(2, produced.size());

    // 5000 time slice -> events from 0 to 4999
    ASSERT_EQ(5000, produced[0].t);
    ASSERT_EQ(2, produced[0].data_->size());
    ASSERT_EQ(0, produced[0].data_->front().t);
    ASSERT_EQ(4999, produced[0].data_->back().t);

    // 10000 time slice -> events from 5000 to 9999
    ASSERT_EQ(10000, produced[1].t);
    ASSERT_EQ(3, produced[1].data_->size());
    ASSERT_EQ(5000, produced[1].data_->front().t);
    ASSERT_EQ(9999, produced[1].data_->back().t);

    // Flush -> process async on remaining events which corresponds to the following truncated time slice
    // [10000, last event's timestamp + 1[
    producer.flush();
    ASSERT_EQ(3, produced.size());

    ASSERT_EQ(10002, produced[2].t);
    ASSERT_EQ(2, produced[2].data_->size());
    ASSERT_EQ(10000, produced[2].data_->front().t);
    ASSERT_EQ(10001, produced[2].data_->back().t);
}

TEST_F(SharedCdEventsBufferProducer_Gtest, shared_cd_buffer_producer_count_policy) {
    // GIVEN A buffer pool of size 10 with pre-allocation of the buffers to size 10 and a event count producing policy
    // (2 events/buffer)
    Metavision::SharedEventsBufferProducerParameters params;
    params.buffers_pool_size_          = 10;
    params.buffers_preallocation_size_ = 10;
    params.buffers_time_slice_us_      = 0;
    params.buffers_events_count_       = 2;

    // WHEN processing a full buffer of events containing 5 events
    const std::vector<Metavision::Event2d> data{
        {0, 0, 0, 0}, {0, 0, 0, 4999}, {0, 0, 0, 5000}, {0, 0, 0, 5001}, {0, 0, 0, 9999}};
    std::vector<SharedCdBufferEvent> produced;
    Metavision::SharedCdEventsBufferProducerAlgorithm producer(params, [&](Metavision::timestamp ts, const auto &ev) {
        produced.push_back({ts, ev});
    });

    producer.process_events(data.cbegin(), data.cend());

    // THEN 2 buffers of size 2 are produced
    ASSERT_EQ(2, produced.size());

    // 1 st buffer
    ASSERT_EQ(4999, produced[0].t);
    ASSERT_EQ(2, produced[0].data_->size());
    ASSERT_EQ(0, produced[0].data_->front().t);
    ASSERT_EQ(4999, produced[0].data_->back().t);

    ASSERT_EQ(5001, produced[1].t);
    ASSERT_EQ(2, produced[1].data_->size());
    ASSERT_EQ(5000, produced[1].data_->front().t);
    ASSERT_EQ(5001, produced[1].data_->back().t);

    // Flush -> process async on remaining events
    // 1 remaining event that belongs to a third buffer created
    producer.flush();
    ASSERT_EQ(3, produced.size());

    ASSERT_EQ(data.back().t + 1, produced[2].t);
    ASSERT_EQ(1, produced[2].data_->size());
    ASSERT_EQ(9999, produced[2].data_->front().t);
}

TEST_F(SharedCdEventsBufferProducer_Gtest, shared_cd_buffer_producer_mixed_policy) {
    // GIVEN A buffer pool of size 10 with pre-allocation of the buffers to size 10 and a mixed producing policy
    // (3 events/buffer & 5ms/buffer)
    Metavision::SharedEventsBufferProducerParameters params;
    params.buffers_pool_size_          = 10;
    params.buffers_preallocation_size_ = 10;
    params.buffers_time_slice_us_      = 5000;
    params.buffers_events_count_       = 3;

    // WHEN processing a full buffer of events containing 5 events with 2 full time slices
    std::vector<Metavision::Event2d> data{
        {0, 0, 0, 0}, {0, 0, 0, 4998}, {0, 0, 0, 4999}, {0, 0, 0, 5001}, {0, 0, 0, 10000}};
    std::vector<SharedCdBufferEvent> produced;
    Metavision::SharedCdEventsBufferProducerAlgorithm producer(params, [&](Metavision::timestamp ts, const auto &ev) {
        produced.push_back({ts, ev});
    });

    producer.process_events(data.cbegin(), data.cend());

    // THEN 2 buffers are produced, one with 3 events (count policy takes over), one with 1 event (time slice policy
    // takes over). The remaining event do not match count or n_us condition.
    ASSERT_EQ(2, produced.size());

    ASSERT_EQ(4999, produced[0].t);
    ASSERT_EQ(3, produced[0].data_->size());
    ASSERT_EQ(0, produced[0].data_->front().t);
    ASSERT_EQ(4999, produced[0].data_->back().t);

    ASSERT_EQ(9999, produced[1].t);
    ASSERT_EQ(1, produced[1].data_->size());
    ASSERT_EQ(5001, produced[1].data_->front().t);

    // Flush -> process async on remaining events
    // 1 remaining event that belongs to a third buffer
    producer.flush();
    ASSERT_EQ(3, produced.size());
    ASSERT_EQ(data.back().t + 1, produced[2].t);
    ASSERT_EQ(1, produced[2].data_->size());
    ASSERT_EQ(10000, produced[2].data_->front().t);
}

TEST_F(SharedCdEventsBufferProducer_Gtest, shared_cd_buffer_producer_none_policy) {
    // GIVEN A buffer pool of size 10 with pre-allocation of the buffers to size 10 and no producing policy applied
    Metavision::SharedEventsBufferProducerParameters params;
    params.buffers_pool_size_          = 10;
    params.buffers_preallocation_size_ = 10;
    params.buffers_time_slice_us_      = 0;
    params.buffers_events_count_       = 0;

    // WHEN processing a full buffer of events
    std::vector<Metavision::Event2d> data{
        {0, 0, 0, 0}, {0, 0, 0, 4998}, {0, 0, 0, 4999}, {0, 0, 0, 5001}, {0, 0, 0, 10000}};
    std::vector<SharedCdBufferEvent> produced;
    Metavision::SharedCdEventsBufferProducerAlgorithm producer(params, [&](Metavision::timestamp ts, const auto &ev) {
        produced.push_back({ts, ev});
    });

    // THEN buffers are produced only if flush is called

    // None policy -> no buffer created as process_async is not called, only events inserted
    producer.process_events(data.cbegin(), data.cbegin() + 3);
    ASSERT_EQ(0, produced.size());
    // None policy -> no buffer created as process_async is not called, only events inserted
    producer.process_events(data.cbegin() + 3, data.cend());
    ASSERT_EQ(0, produced.size());

    producer.flush();
    ASSERT_EQ(1, produced.size()); // None policy -> flush calls process async
    ASSERT_EQ(data.back().t + 1, produced[0].t);
    ASSERT_EQ(data.size(), produced[0].data_->size());
}
