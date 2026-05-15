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
#include <gtest/gtest.h>

#include "metavision/sdk/core/algorithms/event_buffer_reslicer_algorithm.h"
#include "metavision/sdk/base/events/event_cd.h"

using namespace Metavision;

template<bool enable_interruptions>
struct TestParams {
    using ReslicerAlgorithmType = EventBufferReslicerAlgorithmT<enable_interruptions>;
};

template<typename Params>
class EventBufferReslicerAlgorithmT_GTest : public ::testing::Test {
public:
    using ReslicerAlgorithmType = typename Params::ReslicerAlgorithmType;

    EventBufferReslicerAlgorithmT_GTest() {}
};

using TestingTypes = ::testing::Types<TestParams<false>, TestParams<true>>;

TYPED_TEST_CASE(EventBufferReslicerAlgorithmT_GTest, TestingTypes);

TYPED_TEST(EventBufferReslicerAlgorithmT_GTest, identity_mode) {
    using EvContainer              = std::vector<EventCD>;
    using EvIterator               = std::vector<EventCD>::const_iterator;
    using ReslicerAlgorithmType    = typename TypeParam::ReslicerAlgorithmType;
    using ReslicingConditionStatus = typename ReslicerAlgorithmType::ConditionStatus;
    using ReslicingCondition       = typename ReslicerAlgorithmType::Condition;

    // GIVEN 10 consecutive input events with increasing timestamps from t = 0 to 9
    const std::size_t nevents = 10;
    EvContainer input;
    for (std::size_t i = 0; i < nevents; ++i)
        input.push_back(EventCD(0, 0, 0, i));

    ///////////
    // GIVEN an instance of ReslicerAlgorithmType configured in IDENTITY mode
    std::vector<timestamp> ts_slices1;
    std::vector<EvContainer> output1(1);
    ReslicerAlgorithmType reslicer1(
        [&](ReslicingConditionStatus s, timestamp slice_ts_upper_bound, std::size_t slice_nevents) {
            ASSERT_EQ(s, ReslicingConditionStatus::MET_AUTOMATIC);
            for (const auto &ev : output1.back()) {
                ASSERT_LE(ev.t, slice_ts_upper_bound);
            }
            ASSERT_EQ(slice_nevents, output1.back().size());
            ts_slices1.push_back(slice_ts_upper_bound);
            output1.push_back(EvContainer());
        },
        ReslicingCondition::make_identity());

    // WHEN we process the input event buffer in 1 pass
    auto on_ev_cb1 = [&](EvIterator it_beg, EvIterator it_end) {
        auto &container = output1.back();
        container.insert(container.end(), it_beg, it_end);
        for (auto it = it_beg; it != it_end && !ts_slices1.empty(); ++it) {
            ASSERT_LE(ts_slices1.back(), it->t);
        }
    };
    reslicer1.process_events(input.cbegin(), input.cend(), on_ev_cb1);

    // THEN we get 2 output buffers, the first one with the expected events, the second one empty (since process_events
    // sliced just before returning)
    ASSERT_EQ(output1.size(), 2);
    ASSERT_EQ(output1[0].size(), nevents);
    for (std::size_t i = 0; i < nevents; ++i) {
        ASSERT_EQ(output1[0][i].t, i);
    }
    ASSERT_EQ(output1[1].size(), 0);

    ///////////
    // GIVEN an instance of ReslicerAlgorithmType configured in IDENTITY mode
    std::vector<timestamp> ts_slices2;
    std::vector<EvContainer> output2(1);
    ReslicerAlgorithmType reslicer2(
        [&](ReslicingConditionStatus s, timestamp slice_ts_upper_bound, std::size_t slice_nevents) {
            ASSERT_EQ(s, ReslicingConditionStatus::MET_AUTOMATIC);
            for (const auto &ev : output2.back()) {
                ASSERT_LE(ev.t, slice_ts_upper_bound);
            }
            ASSERT_EQ(slice_nevents, output2.back().size());
            ts_slices2.push_back(slice_ts_upper_bound);
            output2.push_back(EvContainer());
        },
        ReslicingCondition::make_identity());

    // WHEN we process the input event buffer in 2 passes
    auto on_ev_cb2 = [&](EvIterator it_beg, EvIterator it_end) {
        auto &container = output2.back();
        container.insert(container.end(), it_beg, it_end);
        for (auto it = it_beg; it != it_end && !ts_slices2.empty(); ++it) {
            ASSERT_LE(ts_slices2.back(), it->t);
        }
    };
    reslicer2.process_events(input.cbegin(), input.cbegin() + nevents / 2, on_ev_cb2);
    reslicer2.process_events(input.cbegin() + nevents / 2, input.cend(), on_ev_cb2);

    // THEN we get 3 output buffers, the first two with the expected events, the third one empty
    ASSERT_EQ(output2.size(), 3);
    ASSERT_EQ(output2[0].size(), nevents / 2);
    for (std::size_t i = 0; i < nevents / 2; ++i) {
        ASSERT_EQ(output2[0][i].t, i);
    }
    ASSERT_EQ(output2[1].size(), nevents / 2);
    for (std::size_t i = 0; i < nevents / 2; ++i) {
        ASSERT_EQ(output2[1][i].t, nevents / 2 + i);
    }
    ASSERT_EQ(output2[2].size(), 0);
}

TYPED_TEST(EventBufferReslicerAlgorithmT_GTest, n_events_mode) {
    using EvContainer              = std::vector<EventCD>;
    using EvIterator               = std::vector<EventCD>::const_iterator;
    using ReslicerAlgorithmType    = typename TypeParam::ReslicerAlgorithmType;
    using ReslicingConditionStatus = typename ReslicerAlgorithmType::ConditionStatus;
    using ReslicingCondition       = typename ReslicerAlgorithmType::Condition;

    // GIVEN 10 consecutive input events with increasing doubled timestamps from t = 0 to 9
    const std::size_t nevents = 18;
    EvContainer input;
    for (std::size_t i = 0; i < nevents / 2; ++i) {
        input.push_back(EventCD(0, 0, 0, i));
        input.push_back(EventCD(0, 0, 0, i));
    }

    // GIVEN an instance of ReslicerAlgorithmType configured in N_EVENTS mode
    const std::size_t slice_size = 3;
    std::vector<timestamp> ts_slices;
    std::vector<EvContainer> output(1);
    ReslicerAlgorithmType reslicer(
        [&](ReslicingConditionStatus s, timestamp slice_ts_upper_bound, std::size_t slice_nevents) {
            ASSERT_EQ(s, ReslicingConditionStatus::MET_N_EVENTS);
            for (const auto &ev : output.back()) {
                ASSERT_LE(ev.t, slice_ts_upper_bound);
            }
            ASSERT_EQ(slice_nevents, output.back().size());
            ts_slices.push_back(slice_ts_upper_bound);
            output.push_back(EvContainer());
        },
        ReslicingCondition::make_n_events(slice_size));

    // WHEN we process the input event buffer in 2 passes
    auto on_ev_cb = [&](EvIterator it_beg, EvIterator it_end) {
        auto &container = output.back();
        container.insert(container.end(), it_beg, it_end);
        for (auto it = it_beg; it != it_end && !ts_slices.empty(); ++it) {
            ASSERT_LE(ts_slices.back(), it->t);
        }
    };
    reslicer.process_events(input.cbegin(), input.cend(), on_ev_cb);

    // THEN we get 4 output buffers, with the expected events
    ASSERT_EQ(output.size(), 1 + nevents / 3);
    for (std::size_t i = 0; i < nevents / 6; ++i) {
        ASSERT_EQ(output[2 * i + 0].size(), 3);
        ASSERT_EQ(output[2 * i + 0][0].t, 3 * i + 0);
        ASSERT_EQ(output[2 * i + 0][1].t, 3 * i + 0);
        ASSERT_EQ(output[2 * i + 0][2].t, 3 * i + 1);
        ASSERT_EQ(output[2 * i + 1].size(), 3);
        ASSERT_EQ(output[2 * i + 1][0].t, 3 * i + 1);
        ASSERT_EQ(output[2 * i + 1][1].t, 3 * i + 2);
        ASSERT_EQ(output[2 * i + 1][2].t, 3 * i + 2);
    }
}

TYPED_TEST(EventBufferReslicerAlgorithmT_GTest, n_events_mode_perfect_size_match) {
    using EvContainer              = std::vector<EventCD>;
    using EvIterator               = std::vector<EventCD>::const_iterator;
    using ReslicerAlgorithmType    = typename TypeParam::ReslicerAlgorithmType;
    using ReslicingConditionStatus = typename ReslicerAlgorithmType::ConditionStatus;
    using ReslicingCondition       = typename ReslicerAlgorithmType::Condition;

    // GIVEN 10 consecutive input events with increasing timestamps from t = 0 to 9
    const std::size_t nevents = 10;
    EvContainer input;
    for (std::size_t i = 0; i < nevents; ++i)
        input.push_back(EventCD(0, 0, 0, i));

    // GIVEN an instance of ReslicerAlgorithmType configured in N_EVENTS mode, with slice ending exactly at the
    // end of the input buffer
    const std::size_t slice_size = nevents;
    std::vector<timestamp> ts_slices;
    std::vector<EvContainer> output(1);
    ReslicerAlgorithmType reslicer(
        [&](ReslicingConditionStatus s, timestamp slice_ts_upper_bound, std::size_t slice_nevents) {
            ASSERT_EQ(s, ReslicingConditionStatus::MET_N_EVENTS);
            for (const auto &ev : output.back()) {
                ASSERT_LE(ev.t, slice_ts_upper_bound);
            }
            ASSERT_EQ(slice_nevents, output.back().size());
            ts_slices.push_back(slice_ts_upper_bound);
            output.push_back(EvContainer());
        },
        ReslicingCondition::make_n_events(slice_size));

    // WHEN we process the input event buffer in 1 pass
    auto on_ev_cb = [&](EvIterator it_beg, EvIterator it_end) {
        auto &container = output.back();
        container.insert(container.end(), it_beg, it_end);
        for (auto it = it_beg; it != it_end && !ts_slices.empty(); ++it) {
            ASSERT_LE(ts_slices.back(), it->t);
        }
    };
    reslicer.process_events(input.cbegin(), input.cend(), on_ev_cb);

    // THEN we get 2 output buffers, the first one with the expected events, the second one empty (since the buffer was
    // sliced right at the end before returning from process_events)
    ASSERT_EQ(output.size(), 2);
    ASSERT_EQ(output[0].size(), nevents);
    for (std::size_t i = 0; i < nevents; ++i) {
        ASSERT_EQ(output[0][i].t, i);
    }
    ASSERT_EQ(output[1].size(), 0);
}

TYPED_TEST(EventBufferReslicerAlgorithmT_GTest, n_us_mode) {
    using EvContainer              = std::vector<EventCD>;
    using EvIterator               = std::vector<EventCD>::const_iterator;
    using ReslicerAlgorithmType    = typename TypeParam::ReslicerAlgorithmType;
    using ReslicingConditionStatus = typename ReslicerAlgorithmType::ConditionStatus;
    using ReslicingCondition       = typename ReslicerAlgorithmType::Condition;

    // GIVEN 10 consecutive input events with increasing timestamps from t = 0 to 9
    const std::size_t nevents = 10;
    EvContainer input;
    for (std::size_t i = 0; i < nevents; ++i)
        input.push_back(EventCD(0, 0, 0, i));

    // GIVEN an instance of ReslicerAlgorithmType configured in N_US mode
    const timestamp slice_duration = 3;
    std::vector<timestamp> ts_slices;
    std::vector<EvContainer> output(1);
    ReslicerAlgorithmType reslicer(
        [&](ReslicingConditionStatus s, timestamp slice_ts_upper_bound, std::size_t slice_nevents) {
            ASSERT_EQ(s, ReslicingConditionStatus::MET_N_US);
            for (const auto &ev : output.back()) {
                ASSERT_LE(ev.t, slice_ts_upper_bound);
            }
            ASSERT_EQ(slice_nevents, output.back().size());
            ts_slices.push_back(slice_ts_upper_bound);
            output.push_back(EvContainer());
        },
        ReslicingCondition::make_n_us(slice_duration));

    // WHEN we process the input event buffer in 2 passes
    auto on_ev_cb = [&](EvIterator it_beg, EvIterator it_end) {
        auto &container = output.back();
        container.insert(container.end(), it_beg, it_end);
        for (auto it = it_beg; it != it_end && !ts_slices.empty(); ++it) {
            ASSERT_LE(ts_slices.back(), it->t);
        }
    };
    reslicer.process_events(input.cbegin(), input.cbegin() + nevents / 2, on_ev_cb);
    reslicer.process_events(input.cbegin() + nevents / 2, input.cend(), on_ev_cb);

    // THEN we get 4 output buffers, with the expected events
    ASSERT_EQ(output.size(), 4);
    ASSERT_EQ(output[0].size(), 3);
    ASSERT_EQ(output[0][0].t, 0);
    ASSERT_EQ(output[0][1].t, 1);
    ASSERT_EQ(output[0][2].t, 2);
    ASSERT_EQ(output[1].size(), 3);
    ASSERT_EQ(output[1][0].t, 3);
    ASSERT_EQ(output[1][1].t, 4);
    ASSERT_EQ(output[1][2].t, 5);
    ASSERT_EQ(output[2].size(), 3);
    ASSERT_EQ(output[2][0].t, 6);
    ASSERT_EQ(output[2][1].t, 7);
    ASSERT_EQ(output[2][2].t, 8);
    ASSERT_EQ(output[3].size(), 1);
    ASSERT_EQ(output[3][0].t, 9);
}

TYPED_TEST(EventBufferReslicerAlgorithmT_GTest, n_us_mode_perfect_size_match) {
    using EvContainer              = std::vector<EventCD>;
    using EvIterator               = std::vector<EventCD>::const_iterator;
    using ReslicerAlgorithmType    = typename TypeParam::ReslicerAlgorithmType;
    using ReslicingConditionStatus = typename ReslicerAlgorithmType::ConditionStatus;
    using ReslicingCondition       = typename ReslicerAlgorithmType::Condition;

    // GIVEN 10 consecutive input events with increasing timestamps from t = 0 to 9
    const std::size_t nevents = 10;
    EvContainer input;
    for (std::size_t i = 0; i < nevents; ++i)
        input.push_back(EventCD(0, 0, 0, i));

    // GIVEN an instance of ReslicerAlgorithmType configured in N_US mode
    const timestamp slice_duration = input.back().t + 1;
    std::vector<timestamp> ts_slices;
    std::vector<EvContainer> output(1);
    ReslicerAlgorithmType reslicer(
        [&](ReslicingConditionStatus s, timestamp slice_ts_upper_bound, std::size_t slice_nevents) {
            ASSERT_EQ(s, ReslicingConditionStatus::MET_N_US);
            for (const auto &ev : output.back()) {
                ASSERT_LE(ev.t, slice_ts_upper_bound);
            }
            ASSERT_EQ(slice_nevents, output.back().size());
            ts_slices.push_back(slice_ts_upper_bound);
            output.push_back(EvContainer());
        },
        ReslicingCondition::make_n_us(slice_duration));

    // WHEN we process the input event buffer in 1 pass
    auto on_ev_cb = [&](EvIterator it_beg, EvIterator it_end) {
        auto &container = output.back();
        container.insert(container.end(), it_beg, it_end);
        for (auto it = it_beg; it != it_end && !ts_slices.empty(); ++it) {
            ASSERT_LE(ts_slices.back(), it->t);
        }
    };
    reslicer.process_events(input.cbegin(), input.cend(), on_ev_cb);

    // THEN we get 1 output buffers (not 2 as in the N_EVENTS case, since here we cannot decide to slice before getting
    // the next input buffer), with the expected events
    ASSERT_EQ(output.size(), 1);
    ASSERT_EQ(output[0].size(), input.size());
    for (std::size_t i = 0; i < output[0].size(); ++i)
        ASSERT_EQ(output[0][i].t, i);
}

TYPED_TEST(EventBufferReslicerAlgorithmT_GTest, mixed_mode) {
    using EvContainer              = std::vector<EventCD>;
    using EvIterator               = std::vector<EventCD>::const_iterator;
    using ReslicerAlgorithmType    = typename TypeParam::ReslicerAlgorithmType;
    using ReslicingConditionStatus = typename ReslicerAlgorithmType::ConditionStatus;
    using ReslicingCondition       = typename ReslicerAlgorithmType::Condition;

    // GIVEN 10 consecutive input events with increasing timestamps from t = 0 to 9, followed by 10 events with
    // duplicated timestamps from t = 10 to 15
    const std::size_t nevents = 20;
    EvContainer input;
    for (std::size_t i = 0; i < nevents / 2; ++i)
        input.push_back(EventCD(0, 0, 0, i));
    for (std::size_t i = 0; i < nevents / 4; ++i) {
        input.push_back(EventCD(0, 0, 0, nevents / 2 + i));
        input.push_back(EventCD(0, 0, 0, nevents / 2 + i));
    }

    // GIVEN an instance of ReslicerAlgorithmType configured in MIXED mode
    const timestamp slice_duration = 3;
    const timestamp slice_max_size = 4;
    std::vector<timestamp> ts_slices;
    std::vector<EvContainer> output(1);
    ReslicerAlgorithmType reslicer(
        [&](ReslicingConditionStatus s, timestamp slice_ts_upper_bound, std::size_t slice_nevents) {
            if (slice_ts_upper_bound <= 9) {
                ASSERT_EQ(s, ReslicingConditionStatus::MET_N_US);
            } else {
                ASSERT_EQ(s, ReslicingConditionStatus::MET_N_EVENTS);
            }
            for (const auto &ev : output.back()) {
                ASSERT_LE(ev.t, slice_ts_upper_bound);
            }
            ASSERT_EQ(slice_nevents, output.back().size());
            ts_slices.push_back(slice_ts_upper_bound);
            output.push_back(EvContainer());
        },
        ReslicingCondition::make_mixed(slice_duration, slice_max_size));

    // WHEN we process the input event buffer in 1 pass
    auto on_ev_cb = [&](EvIterator it_beg, EvIterator it_end) {
        auto &container = output.back();
        container.insert(container.end(), it_beg, it_end);
        for (auto it = it_beg; it != it_end && !ts_slices.empty(); ++it) {
            ASSERT_LE(ts_slices.back(), it->t);
        }
    };
    reslicer.process_events(input.cbegin(), input.cend(), on_ev_cb);

    // THEN we get the following output buffers
    ASSERT_EQ(output.size(), 6);
    ASSERT_EQ(output[0].size(), 3);
    ASSERT_EQ(output[0][0].t, 0);
    ASSERT_EQ(output[0][1].t, 1);
    ASSERT_EQ(output[0][2].t, 2);
    ASSERT_EQ(output[1].size(), 3);
    ASSERT_EQ(output[1][0].t, 3);
    ASSERT_EQ(output[1][1].t, 4);
    ASSERT_EQ(output[1][2].t, 5);
    ASSERT_EQ(output[2].size(), 3);
    ASSERT_EQ(output[2][0].t, 6);
    ASSERT_EQ(output[2][1].t, 7);
    ASSERT_EQ(output[2][2].t, 8);
    ASSERT_EQ(output[3].size(), 4);
    ASSERT_EQ(output[3][0].t, 9);
    ASSERT_EQ(output[3][1].t, 10);
    ASSERT_EQ(output[3][2].t, 10);
    ASSERT_EQ(output[3][3].t, 11);
    ASSERT_EQ(output[4].size(), 4);
    ASSERT_EQ(output[4][0].t, 11);
    ASSERT_EQ(output[4][1].t, 12);
    ASSERT_EQ(output[4][2].t, 12);
    ASSERT_EQ(output[4][3].t, 13);
    ASSERT_EQ(output[5].size(), 3);
    ASSERT_EQ(output[5][0].t, 13);
    ASSERT_EQ(output[5][1].t, 14);
    ASSERT_EQ(output[5][2].t, 14);
}

TYPED_TEST(EventBufferReslicerAlgorithmT_GTest, mixed_mode_events_gap) {
    using EvContainer              = std::vector<EventCD>;
    using EvIterator               = std::vector<EventCD>::const_iterator;
    using ReslicerAlgorithmType    = typename TypeParam::ReslicerAlgorithmType;
    using ReslicingConditionStatus = typename ReslicerAlgorithmType::ConditionStatus;
    using ReslicingCondition       = typename ReslicerAlgorithmType::Condition;

    // GIVEN 10 consecutive input events with increasing timestamps from t = 0 to 9, followed by a gap of 5 time units,
    // finlally followed by 10 events with duplicated timestamps from t = 10 to 15
    const std::size_t nevents = 20;
    EvContainer input;
    for (std::size_t i = 0; i < nevents / 2; ++i)
        input.push_back(EventCD(0, 0, 0, 6 + i));
    for (std::size_t i = 0; i < nevents / 4; ++i) {
        input.push_back(EventCD(0, 0, 0, 6 + 3 * nevents / 4 + i));
        input.push_back(EventCD(0, 0, 0, 6 + 3 * nevents / 4 + i));
    }

    // GIVEN an instance of ReslicerAlgorithmType configured in MIXED mode
    const timestamp slice_duration = 3;
    const timestamp slice_max_size = 4;
    std::vector<timestamp> ts_slices;
    std::vector<EvContainer> output(1);
    ReslicerAlgorithmType reslicer(
        [&](ReslicingConditionStatus s, timestamp slice_ts_upper_bound, std::size_t slice_nevents) {
            if (slice_ts_upper_bound <= 6 + 15) {
                ASSERT_EQ(s, ReslicingConditionStatus::MET_N_US);
            } else {
                ASSERT_EQ(s, ReslicingConditionStatus::MET_N_EVENTS);
            }
            for (const auto &ev : output.back()) {
                ASSERT_LE(ev.t, slice_ts_upper_bound);
            }
            ASSERT_EQ(slice_nevents, output.back().size());
            ts_slices.push_back(slice_ts_upper_bound);
            output.push_back(EvContainer());
        },
        ReslicingCondition::make_mixed(slice_duration, slice_max_size));

    // WHEN we process the input event buffer in 3 passes
    auto on_ev_cb = [&](EvIterator it_beg, EvIterator it_end) {
        auto &container = output.back();
        container.insert(container.end(), it_beg, it_end);
        for (auto it = it_beg; it != it_end && !ts_slices.empty(); ++it) {
            ASSERT_LE(ts_slices.back(), it->t);
        }
    };
    reslicer.notify_elapsed_time(6);
    reslicer.process_events(input.cbegin(), input.cbegin() + nevents / 2, on_ev_cb);
    reslicer.notify_elapsed_time(6 + 3 * nevents / 4);
    reslicer.process_events(input.cbegin() + nevents / 2, input.cend(), on_ev_cb);

    // THEN we get the following output buffers
    ASSERT_EQ(output.size(), 10);
    ASSERT_EQ(output[0].size(), 0);
    ASSERT_EQ(output[1].size(), 0);
    ASSERT_EQ(output[2].size(), 3);
    ASSERT_EQ(output[2][0].t, 6);
    ASSERT_EQ(output[2][1].t, 7);
    ASSERT_EQ(output[2][2].t, 8);
    ASSERT_EQ(output[3].size(), 3);
    ASSERT_EQ(output[3][0].t, 9);
    ASSERT_EQ(output[3][1].t, 10);
    ASSERT_EQ(output[3][2].t, 11);
    ASSERT_EQ(output[4].size(), 3);
    ASSERT_EQ(output[4][0].t, 12);
    ASSERT_EQ(output[4][1].t, 13);
    ASSERT_EQ(output[4][2].t, 14);
    ASSERT_EQ(output[5].size(), 1);
    ASSERT_EQ(output[5][0].t, 15);
    ASSERT_EQ(output[6].size(), 0);
    ASSERT_EQ(output[7].size(), 4);
    ASSERT_EQ(output[7][0].t, 21);
    ASSERT_EQ(output[7][1].t, 21);
    ASSERT_EQ(output[7][2].t, 22);
    ASSERT_EQ(output[7][3].t, 22);
    ASSERT_EQ(output[8].size(), 4);
    ASSERT_EQ(output[8][0].t, 23);
    ASSERT_EQ(output[8][1].t, 23);
    ASSERT_EQ(output[8][2].t, 24);
    ASSERT_EQ(output[8][3].t, 24);
    ASSERT_EQ(output[9].size(), 2);
    ASSERT_EQ(output[9][0].t, 25);
    ASSERT_EQ(output[9][1].t, 25);
}

TEST(InterruptibleEventBufferReslicerAlgorithm_GTest, interruption_before_end_reached) {
    using EvContainer = std::vector<EventCD>;
    using EvIterator  = std::vector<EventCD>::const_iterator;

    // GIVEN 10 consecutive input events with increasing timestamps from t = 0 to 9
    const std::size_t nevents = 10;
    EvContainer input;
    for (std::size_t i = 0; i < nevents; ++i)
        input.push_back(EventCD(0, 0, 0, i));

    ///////////
    // GIVEN an instance of InterruptibleEventBufferReslicerAlgorithm configured in N_EVENTS mode, with 1 event per
    // slice and a slicing callback doing a sleep of 50ms
    const std::size_t slice_size = 1;
    std::vector<EvContainer> output(1);
    InterruptibleEventBufferReslicerAlgorithm reslicer(
        [&](InterruptibleEventBufferReslicerAlgorithm::ConditionStatus s, timestamp slice_ts_upper_bound,
            std::size_t slice_nevents) {
            output.push_back(EvContainer());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        },
        InterruptibleEventBufferReslicerAlgorithm::Condition::make_n_events(slice_size));

    // WHEN we start a lengthy event buffer processing and interrupt it asynchronously
    auto interrupting_thread = std::thread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        reslicer.interrupt();
    });
    auto on_ev_cb            = [&](EvIterator it_beg, EvIterator it_end) {
        auto &container = output.back();
        container.insert(container.end(), it_beg, it_end);
    };
    reslicer.process_events(input.cbegin(), input.cend(), on_ev_cb);
    interrupting_thread.join();

    // THEN some events slices were processed but less than we should have without interruption
    ASSERT_GT(output.size(), 2);
    ASSERT_LT(output.size(), nevents);

    ///////////
    // GIVEN the same instance of InterruptibleEventBufferReslicerAlgorithm after reset and with a normal slicing
    // callback
    reslicer.reset();
    std::vector<timestamp> ts_slices2;
    std::vector<EvContainer> output2(1);
    reslicer.set_on_new_slice_callback([&](InterruptibleEventBufferReslicerAlgorithm::ConditionStatus s,
                                           timestamp slice_ts_upper_bound, std::size_t slice_nevents) {
        ASSERT_EQ(s, InterruptibleEventBufferReslicerAlgorithm::ConditionStatus::MET_N_EVENTS);
        for (const auto &ev : output2.back()) {
            ASSERT_LE(ev.t, slice_ts_upper_bound);
        }
        ASSERT_EQ(slice_nevents, output2.back().size());
        ts_slices2.push_back(slice_ts_upper_bound);
        output2.push_back(EvContainer());
    });

    // WHEN we process the input event buffer
    auto on_ev_cb2 = [&](EvIterator it_beg, EvIterator it_end) {
        auto &container = output2.back();
        container.insert(container.end(), it_beg, it_end);
        for (auto it = it_beg; it != it_end && !ts_slices2.empty(); ++it) {
            ASSERT_LE(ts_slices2.back(), it->t);
        }
    };
    reslicer.process_events(input.cbegin(), input.cend(), on_ev_cb2);

    // THEN we get the expected output buffers with the expected events
    ASSERT_EQ(output2.size(), 1 + nevents);
    for (std::size_t i = 0; i < nevents; ++i) {
        ASSERT_EQ(output2[i].size(), 1);
        ASSERT_EQ(output2[i][0].t, i);
    }
    ASSERT_TRUE(output2[nevents].empty());
}
