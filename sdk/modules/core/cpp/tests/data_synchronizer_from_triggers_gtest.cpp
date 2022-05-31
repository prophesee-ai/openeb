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
#include <vector>
#include <atomic>
#include <thread>
#include <functional>

#include "metavision/sdk/core/utils/data_synchronizer_from_triggers.h"
#include "metavision/sdk/base/events/event2d.h"

namespace Metavision {

using SizeType = std::vector<uint32_t>::size_type;

struct Event2dIndex : public Event2d {
    static uint32_t index_accessor(const Event2dIndex &ev) {
        return ev.index;
    }
    static Metavision::timestamp &timestamp_accessor(Event2dIndex &ev) {
        return ev.t;
    }
    uint32_t index;
};

template<typename T>
std::vector<Metavision::EventExtTrigger> create_trigger_buffer(const std::vector<T> &indices, const uint32_t period_us,
                                                               const timestamp ts_offset_us = 0,
                                                               const timestamp noise        = 0) {
    std::vector<Metavision::EventExtTrigger> triggers;
    for (auto it_idx = indices.cbegin(), it_idx_end = indices.cend(); it_idx != it_idx_end; ++it_idx) {
        Metavision::EventExtTrigger ev_up, ev_down;
        ev_up.t    = ts_offset_us + (period_us * (*it_idx)) + (noise > 0 ? (rand() % noise) : 0);
        ev_up.p    = 1;
        ev_up.id   = 0;
        ev_down.t  = ev_up.t;
        ev_down.p  = 0;
        ev_down.id = 0;
        triggers.push_back(ev_up);
        triggers.push_back(ev_down);
    }

    return triggers;
}

class DataSynchronizerFromTriggers_GTest : public ::testing::Test {
public:
    DataSynchronizerFromTriggers_GTest() {}

    virtual ~DataSynchronizerFromTriggers_GTest() {}
};

TEST_F(DataSynchronizerFromTriggers_GTest, ThrowIfNullPeriod) {
    // Check that instantiating parameters with a period of value 0 is invalid and results in an exception being thrown

    uint32_t period_us = 0;
    ASSERT_THROW({ DataSynchronizerFromTriggers::Parameters params(period_us); }, std::invalid_argument);
}

TEST_F(DataSynchronizerFromTriggers_GTest, CanNotIndexTriggerConditions) {
    // Check that the indexing method returns false whenever we don't index any trigger or index a duplicate

    std::vector<EventExtTrigger> triggers_indexed;

    uint32_t period_us = 20000;
    DataSynchronizerFromTriggers::Parameters param(period_us);
    param.index_offset_                 = 0;
    param.reference_polarity_           = 0;
    param.periodicity_tolerance_factor_ = 0.1;
    param.to_discard_count_             = 0;
    DataSynchronizerFromTriggers sync(param);

    const Metavision::EventExtTrigger failed_to_index_polarity(!param.reference_polarity_, 0,
                                                               0); // polarity 1 are not indexed
    const Metavision::EventExtTrigger successfully_indexed(param.reference_polarity_, 0,
                                                           0);               // polarity 0 are indexed
    const Metavision::EventExtTrigger failed_to_index(successfully_indexed); // Same as the previous one
    const Metavision::EventExtTrigger successfully_indexed_bis(param.reference_polarity_, period_us,
                                                               0); // okay trigger

    ASSERT_FALSE(sync.index_triggers(&failed_to_index_polarity, &failed_to_index_polarity + 1));
    ASSERT_TRUE(sync.index_triggers(&successfully_indexed, &successfully_indexed + 1));
    ASSERT_FALSE(sync.index_triggers(&failed_to_index, &failed_to_index + 1)); // Same as successfully_indexed
    ASSERT_TRUE(sync.index_triggers(&successfully_indexed_bis, &successfully_indexed_bis + 1));
    ASSERT_FALSE(sync.index_triggers(&failed_to_index,
                                     &failed_to_index +
                                         1)); // In the past (has t = 0 when successfully_indexed_bis has t = period)
}

TEST_F(DataSynchronizerFromTriggers_GTest, Nominal) {
    // Verify that with a basic set of triggers and indexed data, we associate the correct timestamps.
    // All triggers + all frames = each frame has the timestamp of its corresponding trigger

    const int32_t period_us = 20000;
    DataSynchronizerFromTriggers::Parameters param(period_us);
    param.index_offset_                 = 0;
    param.reference_polarity_           = 0;
    param.periodicity_tolerance_factor_ = 0.1;
    param.to_discard_count_             = 0;
    DataSynchronizerFromTriggers sync(param);

    const std::vector<uint32_t> indices_data_input    = {0, 1, 2, 3, 4, 5};
    const std::vector<uint32_t> indices_trigger_input = {0, 1, 2, 3, 4, 5};
    std::vector<Event2dIndex> to_index(indices_data_input.size());
    for (SizeType i = 0; i < indices_data_input.size(); ++i) {
        to_index[i].index = indices_data_input[i];
    }

    auto trigger_buffer = create_trigger_buffer(indices_trigger_input, period_us);
    ASSERT_TRUE(sync.index_triggers(trigger_buffer.cbegin(), trigger_buffer.cend()));
    ASSERT_EQ(indices_data_input.size(),
              sync.synchronize_data_from_triggers(to_index.begin(), to_index.end(), &Event2dIndex::timestamp_accessor,
                                                  &Event2dIndex::index_accessor));

    for (SizeType i = 0; i < indices_data_input.size(); ++i) {
        ASSERT_EQ(trigger_buffer[2 * i + !param.reference_polarity_].t, to_index[i].t);
    }
}

TEST_F(DataSynchronizerFromTriggers_GTest, NominalNoisy) {
    // Verify that with a set of triggers with almost perfectly periodic ts and indexed data, we associate the correct
    // timestamps. All triggers + all frames = each frames has the timestamp of its corresponding trigger

    const int32_t period_us = 20000;
    DataSynchronizerFromTriggers::Parameters param(period_us);
    param.index_offset_                 = 0;
    param.reference_polarity_           = 1;
    param.periodicity_tolerance_factor_ = 0.1;
    param.to_discard_count_             = 0;
    DataSynchronizerFromTriggers sync(param);

    const std::vector<uint32_t> indices_data_input    = {0, 1, 2, 3, 4, 5};
    const std::vector<uint32_t> indices_trigger_input = {0, 1, 2, 3, 4, 5};
    std::vector<Event2dIndex> to_index(indices_data_input.size());
    for (SizeType i = 0; i < indices_data_input.size(); ++i) {
        to_index[i].index = indices_data_input[i];
    }

    auto trigger_buffer =
        create_trigger_buffer(indices_trigger_input, period_us, 0, param.periodicity_tolerance_factor_ * period_us);
    ASSERT_TRUE(sync.index_triggers(trigger_buffer.cbegin(), trigger_buffer.cend()));
    ASSERT_EQ(indices_data_input.size(),
              sync.synchronize_data_from_triggers(to_index.begin(), to_index.end(), &Event2dIndex::timestamp_accessor,
                                                  &Event2dIndex::index_accessor));

    for (SizeType i = 0; i < indices_data_input.size(); ++i) {
        ASSERT_EQ(trigger_buffer[2 * i + !param.reference_polarity_].t, to_index[i].t);
    }
}

TEST_F(DataSynchronizerFromTriggers_GTest, LostTriggers) {
    // Verify that with a set of triggers with some missing, and a set of indexed data, we associate the correct
    // timestamps. If a trigger is missing, the timestamp must be interpolated correctly and must be between the
    // previous and the following one

    const int32_t period_us = 20000;
    DataSynchronizerFromTriggers::Parameters param(period_us);
    param.index_offset_                 = 0;
    param.reference_polarity_           = 0;
    param.periodicity_tolerance_factor_ = 0.1;
    param.to_discard_count_             = 0;
    DataSynchronizerFromTriggers sync(param);

    const std::vector<uint32_t> indices_data_input    = {0, 1, 2, 3, 4, 5};
    const std::vector<uint32_t> indices_trigger_input = {0, 1, 4, 5};
    std::vector<Event2dIndex> to_index(indices_data_input.size());
    for (SizeType i = 0; i < indices_data_input.size(); ++i) {
        to_index[i].index = indices_data_input[i];
    }

    auto trigger_buffer =
        create_trigger_buffer(indices_trigger_input, period_us, 0, param.periodicity_tolerance_factor_ * period_us);

    ASSERT_TRUE(sync.index_triggers(trigger_buffer.cbegin(), trigger_buffer.cend()));
    ASSERT_EQ(indices_data_input.size(),
              sync.synchronize_data_from_triggers(to_index.begin(), to_index.end(), &Event2dIndex::timestamp_accessor,
                                                  &Event2dIndex::index_accessor));

    ASSERT_EQ(trigger_buffer[!param.reference_polarity_].t, to_index[0].t);
    ASSERT_EQ(trigger_buffer[2 + !param.reference_polarity_].t, to_index[1].t);
    ASSERT_EQ(trigger_buffer[2 + !param.reference_polarity_].t + period_us,
              to_index[2].t); // missing trigger that was interpolated
    ASSERT_EQ(trigger_buffer[2 + !param.reference_polarity_].t + 2 * period_us,
              to_index[3].t); // missing trigger that was interpolated
    ASSERT_EQ(trigger_buffer[4 + !param.reference_polarity_].t, to_index[4].t);
    ASSERT_EQ(trigger_buffer[6 + !param.reference_polarity_].t, to_index[5].t);
}

TEST_F(DataSynchronizerFromTriggers_GTest, LostDataAndTriggers) {
    // Verify that with a set of triggers with some missing, and a set of indexed data with some missing, we associate
    // the correct timestamps.

    const int32_t period_us = 20000;
    DataSynchronizerFromTriggers::Parameters param(period_us);
    param.index_offset_                 = 0;
    param.reference_polarity_           = 0;
    param.periodicity_tolerance_factor_ = 0.1;
    param.to_discard_count_             = 0;
    DataSynchronizerFromTriggers sync(param);

    const std::vector<uint32_t> indices_data_input    = {0, 2, 3, 5};
    const std::vector<uint32_t> indices_trigger_input = {0, 1, 4, 5};
    std::vector<Event2dIndex> to_index(indices_data_input.size());
    for (SizeType i = 0; i < indices_data_input.size(); ++i) {
        to_index[i].index = indices_data_input[i];
    }

    auto trigger_buffer =
        create_trigger_buffer(indices_trigger_input, period_us, 0, param.periodicity_tolerance_factor_ * period_us);

    std::vector<EventExtTrigger> indexed;
    sync.index_triggers(trigger_buffer.cbegin(), trigger_buffer.cend(), std::back_inserter(indexed));
    ASSERT_EQ(6, indexed.size()); // Trigger with index from 0 to 5 to be present/interpolated if missing to synchronize
                                  // frame with sequence count from 0 to 5 --> 6 triggers must be indexed.
    ASSERT_EQ(indices_data_input.size(),
              sync.synchronize_data_from_triggers(to_index.begin(), to_index.end(), &Event2dIndex::timestamp_accessor,
                                                  &Event2dIndex::index_accessor));

    ASSERT_EQ(trigger_buffer[!param.reference_polarity_].t, to_index[0].t);
    ASSERT_EQ(trigger_buffer[2 + !param.reference_polarity_].t + period_us,
              to_index[1].t); // missing trigger that was interpolated to missing frame
    ASSERT_EQ(trigger_buffer[2 + !param.reference_polarity_].t + 2 * period_us,
              to_index[2].t); // missing trigger that was interpolated
    ASSERT_EQ(trigger_buffer[6 + !param.reference_polarity_].t, to_index[3].t);
}

TEST_F(DataSynchronizerFromTriggers_GTest, NotEnoughTriggers) {
    // In case we have more data than triggers, checks that only the right amount is timestamped and that we leave the
    // function correctly (avoid deadlock)

    const int32_t period_us = 20000;
    DataSynchronizerFromTriggers::Parameters param(period_us);
    param.index_offset_                 = 0;
    param.reference_polarity_           = 0;
    param.periodicity_tolerance_factor_ = 0.1;
    param.to_discard_count_             = 0;
    DataSynchronizerFromTriggers sync(param);

    const std::vector<uint32_t> indices_data_input    = {0, 1, 2, 3, 4, 5};
    const std::vector<uint32_t> indices_trigger_input = {0, 1, 2, 3};
    std::vector<Event2dIndex> to_index(indices_data_input.size());
    for (SizeType i = 0; i < indices_data_input.size(); ++i) {
        to_index[i].index = indices_data_input[i];
    }

    auto trigger_buffer =
        create_trigger_buffer(indices_trigger_input, period_us, 0, param.periodicity_tolerance_factor_ * period_us);

    ASSERT_TRUE(sync.index_triggers(trigger_buffer.cbegin(), trigger_buffer.cend()));
    sync.set_synchronization_as_done();
    ASSERT_EQ(indices_trigger_input.size(),
              sync.synchronize_data_from_triggers(to_index.begin(), to_index.end(), &Event2dIndex::timestamp_accessor,
                                                  &Event2dIndex::index_accessor));

    using SizeType = std::vector<uint32_t>::size_type;
    for (SizeType i = 0; i < indices_trigger_input.size(); ++i) {
        ASSERT_EQ(trigger_buffer[2 * i + !param.reference_polarity_].t, to_index[i].t);
    }
}

TEST_F(DataSynchronizerFromTriggers_GTest, NoTriggerToSyncFrame) {
    // In case we have more data than triggers, checks that only the right amount is timestamped and that we leave the
    // function correctly (avoid deadlock)

    const int32_t period_us = 20000;
    DataSynchronizerFromTriggers::Parameters param(period_us);
    param.index_offset_                 = 3;
    param.reference_polarity_           = 0;
    param.periodicity_tolerance_factor_ = 0.1;
    param.to_discard_count_             = 0;
    DataSynchronizerFromTriggers sync(param);

    const std::vector<uint32_t> indices_data_input    = {1, 2, 3, 4, 5};
    const std::vector<uint32_t> indices_trigger_input = {3, 4, 5};
    std::vector<Event2dIndex> to_index(indices_data_input.size());
    for (SizeType i = 0; i < indices_data_input.size(); ++i) {
        to_index[i].index = indices_data_input[i];
    }

    auto trigger_buffer =
        create_trigger_buffer(indices_trigger_input, period_us, 0, param.periodicity_tolerance_factor_ * period_us);

    ASSERT_TRUE(sync.index_triggers(trigger_buffer.cbegin(), trigger_buffer.cend()));
    sync.set_synchronization_as_done();
    ASSERT_EQ(indices_data_input.size(),
              sync.synchronize_data_from_triggers(to_index.begin(), to_index.end(), &Event2dIndex::timestamp_accessor,
                                                  &Event2dIndex::index_accessor));

    ASSERT_EQ(trigger_buffer[0 + !param.reference_polarity_].t - 2 * period_us, to_index[0].t);
    ASSERT_EQ(trigger_buffer[0 + !param.reference_polarity_].t - period_us, to_index[1].t);
    ASSERT_EQ(trigger_buffer[0 + !param.reference_polarity_].t, to_index[2].t);
    ASSERT_EQ(trigger_buffer[2 + !param.reference_polarity_].t, to_index[3].t);
    ASSERT_EQ(trigger_buffer[4 + !param.reference_polarity_].t, to_index[4].t);
}

TEST_F(DataSynchronizerFromTriggers_GTest, TriggerDiscarded) {
    // Checks first triggers to be discarded is taken into consideration

    const int32_t period_us = 20000;
    DataSynchronizerFromTriggers::Parameters param(period_us);
    param.index_offset_                 = 0;
    param.reference_polarity_           = 1;
    param.periodicity_tolerance_factor_ = 0.1;
    param.to_discard_count_             = 2;
    DataSynchronizerFromTriggers sync(param);

    const std::vector<uint32_t> indices_data_input    = {0, 1, 2, 3, 4, 5};
    const std::vector<uint32_t> indices_trigger_input = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<Event2dIndex> to_index(indices_data_input.size());
    for (SizeType i = 0; i < indices_data_input.size(); ++i) {
        to_index[i].index = indices_data_input[i];
    }

    auto trigger_buffer =
        create_trigger_buffer(indices_trigger_input, period_us, 0, param.periodicity_tolerance_factor_ * period_us);

    ASSERT_TRUE(sync.index_triggers(trigger_buffer.cbegin(), trigger_buffer.cend()));
    ASSERT_EQ(indices_data_input.size(),
              sync.synchronize_data_from_triggers(to_index.begin(), to_index.end(), &Event2dIndex::timestamp_accessor,
                                                  &Event2dIndex::index_accessor));

    for (SizeType i = 0; i < indices_data_input.size(); ++i) {
        ASSERT_EQ(trigger_buffer[2 * (i + param.to_discard_count_) + !param.reference_polarity_].t, to_index[i].t);
    }
}

TEST_F(DataSynchronizerFromTriggers_GTest, NominalThreaded) {
    // Checks that we synchronize correctly data in nominal case in threaded context

    const int32_t period_us = 20000;
    DataSynchronizerFromTriggers::Parameters param(period_us);
    param.index_offset_                 = 0;
    param.reference_polarity_           = 1;
    param.periodicity_tolerance_factor_ = 0.1;
    param.to_discard_count_             = 0;
    DataSynchronizerFromTriggers sync(param);

    const std::vector<uint32_t> indices_data_input    = {0, 1, 2, 3, 4, 5};
    const std::vector<uint32_t> indices_trigger_input = {0, 1, 2, 3, 4, 5};
    std::vector<Event2dIndex> to_index(indices_data_input.size());
    for (SizeType i = 0; i < indices_data_input.size(); ++i) {
        to_index[i].index = indices_data_input[i];
    }

    auto trigger_buffer = create_trigger_buffer(indices_trigger_input, period_us, 0);

    std::thread trigger_thread([&]() {
        for (SizeType i = 0; i < trigger_buffer.size() / 2; ++i) {
            std::this_thread::sleep_for(std::chrono::microseconds(period_us));
            EXPECT_TRUE(sync.index_triggers(trigger_buffer.cbegin() + 2 * i, trigger_buffer.cbegin() + 2 * (i + 1)));
        }
    });

    EXPECT_EQ(indices_data_input.size(),
              sync.synchronize_data_from_triggers(to_index.begin(), to_index.end(), &Event2dIndex::timestamp_accessor,
                                                  &Event2dIndex::index_accessor));

    for (SizeType i = 0; i < indices_data_input.size(); ++i) {
        EXPECT_EQ(trigger_buffer[2 * (i + param.to_discard_count_) + !param.reference_polarity_].t, to_index[i].t);
    }

    trigger_thread.join();
}

TEST_F(DataSynchronizerFromTriggers_GTest, LostTriggersThreaded) {
    // Checks that we synchronize correctly data in threaded context even if triggers are missing

    const int32_t period_us = 20000;
    DataSynchronizerFromTriggers::Parameters param(period_us);
    param.index_offset_                 = 0;
    param.reference_polarity_           = 1;
    param.periodicity_tolerance_factor_ = 0.1;
    param.to_discard_count_             = 0;
    DataSynchronizerFromTriggers sync(param);

    const std::vector<uint32_t> indices_data_input    = {0, 1, 2, 3, 4, 5};
    const std::vector<uint32_t> indices_trigger_input = {0, 1, 5};
    std::vector<Event2dIndex> to_index(indices_data_input.size());
    for (SizeType i = 0; i < indices_data_input.size(); ++i) {
        to_index[i].index = indices_data_input[i];
    }

    auto trigger_buffer = create_trigger_buffer(indices_trigger_input, period_us, 0);

    std::thread trigger_thread([&]() {
        std::this_thread::sleep_for(std::chrono::microseconds(period_us));
        EXPECT_TRUE(sync.index_triggers(trigger_buffer.cbegin(), trigger_buffer.cbegin() + 2));

        std::this_thread::sleep_for(std::chrono::microseconds(period_us));
        EXPECT_TRUE(sync.index_triggers(trigger_buffer.cbegin() + 2, trigger_buffer.cbegin() + 4));

        std::this_thread::sleep_for(std::chrono::microseconds(period_us)); // Missing 2, 3, and 4th trigger
        std::this_thread::sleep_for(std::chrono::microseconds(period_us));
        std::this_thread::sleep_for(std::chrono::microseconds(period_us));

        std::this_thread::sleep_for(std::chrono::microseconds(period_us));
        EXPECT_TRUE(sync.index_triggers(trigger_buffer.cbegin() + 4, trigger_buffer.cbegin() + 6));
    });

    const auto tnow = std::chrono::system_clock::now();
    EXPECT_EQ(indices_data_input.size(),
              sync.synchronize_data_from_triggers(to_index.begin(), to_index.end(), &Event2dIndex::timestamp_accessor,
                                                  &Event2dIndex::index_accessor));
    const auto tdiff =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - tnow).count();
    EXPECT_GE(tdiff, period_us * indices_data_input.size());

    EXPECT_EQ(trigger_buffer[!param.reference_polarity_].t, to_index[0].t);
    EXPECT_EQ(trigger_buffer[2 + !param.reference_polarity_].t, to_index[1].t);
    EXPECT_EQ(trigger_buffer[2 + !param.reference_polarity_].t + period_us,
              to_index[2].t); // missing trigger that was interpolated
    EXPECT_EQ(trigger_buffer[2 + !param.reference_polarity_].t + 2 * period_us,
              to_index[3].t); // missing trigger that was interpolated
    EXPECT_EQ(trigger_buffer[2 + !param.reference_polarity_].t + 3 * period_us,
              to_index[4].t); // missing trigger that was interpolated
    EXPECT_EQ(trigger_buffer[4 + !param.reference_polarity_].t, to_index[5].t);

    trigger_thread.join();
}

} // namespace Metavision
