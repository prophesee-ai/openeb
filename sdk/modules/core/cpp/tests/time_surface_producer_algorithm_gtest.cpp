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
#include <metavision/sdk/base/events/event_cd.h>

#include "metavision/sdk/core/algorithms/time_surface_producer_algorithm.h"

class TimesurfaceProducerAlgorithmGTest : public ::testing::Test {
public:
    TimesurfaceProducerAlgorithmGTest()          = default;
    virtual ~TimesurfaceProducerAlgorithmGTest() = default;
};

TEST_F(TimesurfaceProducerAlgorithmGTest, test_output_n_positive_events) {
    Metavision::TimeSurfaceProducerAlgorithm<1> producer(3, 3);
    Metavision::timestamp ts;
    Metavision::MostRecentTimestampBuffer timesurface;

    producer.set_output_callback([&ts, &timesurface](Metavision::timestamp output_ts,
                                                     const Metavision::MostRecentTimestampBuffer &output_timesurface) {
        ts          = output_ts;
        timesurface = output_timesurface;
    });

    // GIVEN
    // - a producer that produces a one-channel time surface every 5 events, and
    // - a buffer of 6 positive events
    producer.set_processing_n_events(5);

    std::vector<Metavision::EventCD> events = {{0, 0, 1, 0}, {1, 0, 1, 1}, {2, 0, 1, 2},
                                               {0, 1, 1, 3}, {1, 1, 1, 4}, {2, 1, 1, 5}};

    // WHEN
    // We process the events
    producer.process_events(events.cbegin(), events.cend());

    // THEN
    // One time surface is produced and
    //                |0 1 2|
    // timesurface =  |3 4 0|
    //                |0 0 0|
    ASSERT_EQ(ts, 4);
    ASSERT_EQ(timesurface.at(0, 0), 0);
    ASSERT_EQ(timesurface.at(0, 1), 1);
    ASSERT_EQ(timesurface.at(0, 2), 2);
    ASSERT_EQ(timesurface.at(1, 0), 3);
    ASSERT_EQ(timesurface.at(1, 1), 4);
    ASSERT_EQ(timesurface.at(1, 2), 0);
    ASSERT_EQ(timesurface.at(2, 0), 0);
    ASSERT_EQ(timesurface.at(2, 1), 0);
    ASSERT_EQ(timesurface.at(2, 2), 0);
}

TEST_F(TimesurfaceProducerAlgorithmGTest, test_output_n_negative_events) {
    Metavision::TimeSurfaceProducerAlgorithm<1> producer(3, 3);
    Metavision::timestamp ts;
    Metavision::MostRecentTimestampBuffer timesurface;

    producer.set_output_callback([&ts, &timesurface](Metavision::timestamp output_ts,
                                                     const Metavision::MostRecentTimestampBuffer &output_timesurface) {
        ts          = output_ts;
        timesurface = output_timesurface;
    });

    // GIVEN
    // - a producer that produces a one-channel time surface every 5 events, and
    // - a buffer of 6 negative events
    producer.set_processing_n_events(5);

    std::vector<Metavision::EventCD> events = {{0, 0, 0, 0}, {1, 0, 0, 1}, {2, 0, 0, 2},
                                               {0, 1, 0, 3}, {1, 1, 0, 4}, {2, 1, 0, 5}};

    // WHEN
    // We process the events
    producer.process_events(events.cbegin(), events.cend());

    // THEN
    // One time surface is produced and
    //                |0 1 2|
    // timesurface =  |3 4 0|
    //                |0 0 0|
    ASSERT_EQ(ts, 4);
    ASSERT_EQ(timesurface.at(0, 0), 0);
    ASSERT_EQ(timesurface.at(0, 1), 1);
    ASSERT_EQ(timesurface.at(0, 2), 2);
    ASSERT_EQ(timesurface.at(1, 0), 3);
    ASSERT_EQ(timesurface.at(1, 1), 4);
    ASSERT_EQ(timesurface.at(1, 2), 0);
    ASSERT_EQ(timesurface.at(2, 0), 0);
    ASSERT_EQ(timesurface.at(2, 1), 0);
    ASSERT_EQ(timesurface.at(2, 2), 0);
}

TEST_F(TimesurfaceProducerAlgorithmGTest, test_output_n_mixed_positive_and_negative_events) {
    Metavision::TimeSurfaceProducerAlgorithm<1> producer(3, 3);
    Metavision::timestamp ts;
    Metavision::MostRecentTimestampBuffer timesurface;

    producer.set_output_callback([&ts, &timesurface](Metavision::timestamp output_ts,
                                                     const Metavision::MostRecentTimestampBuffer &output_timesurface) {
        ts          = output_ts;
        timesurface = output_timesurface;
    });

    // GIVEN
    // - a producer that produces a one-channel time surface every 5 events, and
    // - a buffer of 6 events (mix of positive and negative)
    producer.set_processing_n_events(5);

    std::vector<Metavision::EventCD> events = {{0, 0, 1, 0}, {1, 0, 0, 1}, {2, 0, 1, 2},
                                               {0, 1, 1, 3}, {1, 1, 0, 4}, {2, 1, 0, 5}};

    // WHEN
    // We process the events
    producer.process_events(events.cbegin(), events.cend());

    // THEN
    // One time surface is produced and
    //                |0 1 2|
    // timesurface =  |3 4 0|
    //                |0 0 0|
    ASSERT_EQ(ts, 4);
    ASSERT_EQ(timesurface.at(0, 0), 0);
    ASSERT_EQ(timesurface.at(0, 1), 1);
    ASSERT_EQ(timesurface.at(0, 2), 2);
    ASSERT_EQ(timesurface.at(1, 0), 3);
    ASSERT_EQ(timesurface.at(1, 1), 4);
    ASSERT_EQ(timesurface.at(1, 2), 0);
    ASSERT_EQ(timesurface.at(2, 0), 0);
    ASSERT_EQ(timesurface.at(2, 1), 0);
    ASSERT_EQ(timesurface.at(2, 2), 0);
}

TEST_F(TimesurfaceProducerAlgorithmGTest, test_output_n_mixed_events_two_channels) {
    Metavision::TimeSurfaceProducerAlgorithm<2> producer(3, 3);
    Metavision::timestamp ts;
    Metavision::MostRecentTimestampBuffer timesurface;

    producer.set_output_callback([&ts, &timesurface](Metavision::timestamp output_ts,
                                                     const Metavision::MostRecentTimestampBuffer &output_timesurface) {
        ts          = output_ts;
        timesurface = output_timesurface;
    });

    // GIVEN
    // - a producer that produces a two-channels time surface every 5 events, and
    // - a buffer of 6 events (mix of positive and negative)
    producer.set_processing_n_events(5);

    std::vector<Metavision::EventCD> events = {{0, 0, 1, 0}, {1, 0, 0, 1}, {2, 0, 1, 2},
                                               {0, 1, 1, 3}, {1, 1, 0, 4}, {2, 1, 0, 5}};

    // WHEN
    // We process the events
    producer.process_events(events.cbegin(), events.cend());

    // THEN
    // Two time surfaces are produced (one for each channel)
    //                                    |0 1 0|
    // timesurface for positive events =  |0 4 0|
    //                                    |0 0 0|
    //
    //                                    |0 0 2|
    // timesurface for negative events =  |3 0 0|
    //                                    |0 0 0|
    ASSERT_EQ(ts, 4);

    ASSERT_EQ(timesurface.at(0, 0, 0), 0);
    ASSERT_EQ(timesurface.at(0, 1, 0), 1);
    ASSERT_EQ(timesurface.at(0, 2, 0), 0);
    ASSERT_EQ(timesurface.at(1, 0, 0), 0);
    ASSERT_EQ(timesurface.at(1, 1, 0), 4);
    ASSERT_EQ(timesurface.at(1, 2, 0), 0);
    ASSERT_EQ(timesurface.at(2, 0, 0), 0);
    ASSERT_EQ(timesurface.at(2, 1, 0), 0);
    ASSERT_EQ(timesurface.at(2, 2, 0), 0);

    ASSERT_EQ(timesurface.at(0, 0, 1), 0);
    ASSERT_EQ(timesurface.at(0, 1, 1), 0);
    ASSERT_EQ(timesurface.at(0, 2, 1), 2);
    ASSERT_EQ(timesurface.at(1, 0, 1), 3);
    ASSERT_EQ(timesurface.at(1, 1, 1), 0);
    ASSERT_EQ(timesurface.at(1, 2, 1), 0);
    ASSERT_EQ(timesurface.at(2, 0, 1), 0);
    ASSERT_EQ(timesurface.at(2, 1, 1), 0);
    ASSERT_EQ(timesurface.at(2, 2, 1), 0);
}

TEST_F(TimesurfaceProducerAlgorithmGTest, test_output_n_us) {
    Metavision::TimeSurfaceProducerAlgorithm<1> producer(3, 3);
    Metavision::timestamp ts;
    Metavision::MostRecentTimestampBuffer timesurface;

    producer.set_output_callback([&ts, &timesurface](Metavision::timestamp output_ts,
                                                     const Metavision::MostRecentTimestampBuffer &output_timesurface) {
        ts          = output_ts;
        timesurface = output_timesurface;
    });

    // GIVEN
    // - a producer that produces a time surface every 5us, and
    // - a buffer of 5us of events
    producer.set_processing_n_us(5);

    std::vector<Metavision::EventCD> events = {{0, 0, 1, 0}, {1, 0, 0, 1}, {2, 0, 1, 2},
                                               {0, 1, 1, 3}, {1, 1, 0, 4}, {2, 1, 1, 5}};

    // WHEN
    // We process the events
    producer.process_events(events.cbegin(), events.cend());

    // THEN
    // One time surface is produced and
    //               |0 1 2|
    // timesurface = |3 4 0|
    //               |0 0 0|
    ASSERT_EQ(ts, 5);
    ASSERT_EQ(timesurface.at(0, 0), 0);
    ASSERT_EQ(timesurface.at(0, 1), 1);
    ASSERT_EQ(timesurface.at(0, 2), 2);
    ASSERT_EQ(timesurface.at(1, 0), 3);
    ASSERT_EQ(timesurface.at(1, 1), 4);
    ASSERT_EQ(timesurface.at(1, 2), 0);
    ASSERT_EQ(timesurface.at(2, 0), 0);
    ASSERT_EQ(timesurface.at(2, 1), 0);
    ASSERT_EQ(timesurface.at(2, 2), 0);
}

TEST_F(TimesurfaceProducerAlgorithmGTest, test_keeping_history) {
    Metavision::TimeSurfaceProducerAlgorithm<1> producer(3, 3);
    Metavision::timestamp ts;
    Metavision::MostRecentTimestampBuffer timesurface;

    producer.set_output_callback([&ts, &timesurface](Metavision::timestamp output_ts,
                                                     const Metavision::MostRecentTimestampBuffer &output_timesurface) {
        ts          = output_ts;
        timesurface = output_timesurface;
    });

    // GIVEN
    // - a producer that produces a time surface every 4 events, and
    // - a buffer of 9 events
    producer.set_processing_n_events(4);

    // clang-format off
    std::vector<Metavision::EventCD> events = {{0, 0, 0, 0}, {1, 0, 0, 1}, {2, 0, 1, 2},
                                               {0, 1, 1, 3}, {1, 1, 1, 4}, {2, 1, 0, 5},
                                               {0, 2, 1, 6}, {1, 2, 0, 7}, {2, 2, 1, 8}};
    // clang-format on

    // WHEN
    // We process the events
    producer.process_events(events.cbegin(), events.cend());

    // THEN
    // Two copies of the internal time surface are done and the last
    //               |0 1 2|         |0 0 0|
    // timesurface = |3 4 5| and not |0 4 5|
    //               |6 7 0|         |6 7 0|
    ASSERT_EQ(ts, 7);
    ASSERT_EQ(timesurface.at(0, 0), 0);
    ASSERT_EQ(timesurface.at(0, 1), 1);
    ASSERT_EQ(timesurface.at(0, 2), 2);
    ASSERT_EQ(timesurface.at(1, 0), 3);
    ASSERT_EQ(timesurface.at(1, 1), 4);
    ASSERT_EQ(timesurface.at(1, 2), 5);
    ASSERT_EQ(timesurface.at(2, 0), 6);
    ASSERT_EQ(timesurface.at(2, 1), 7);
    ASSERT_EQ(timesurface.at(2, 2), 0);
}