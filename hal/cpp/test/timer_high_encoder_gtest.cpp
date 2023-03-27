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

#include "metavision/hal/decoders/base/event_base.h"
#include "timer_high_encoder.h"
#include "evt2_raw_format.h"
#include "encoding_policies.h"

using namespace Metavision;

class TimerHighEncoder_GTest : public ::testing::Test {
protected:
    virtual void SetUp() override {}

    virtual void TearDown() override {}
};

TEST_F(TimerHighEncoder_GTest, evt2_get_size_encoded) {
    TimerHighEncoder<Evt2RawFormat, TimerHighRedundancyNone> t_encoder;
    EXPECT_EQ(4, t_encoder.get_size_encoded());
}

TEST_F(TimerHighEncoder_GTest, evt2_no_redundancy_initialize) {
    // EVT1 TH STEP is 64

    TimerHighEncoder<Evt2RawFormat, TimerHighRedundancyNone> t_encoder;
    EXPECT_EQ(0, t_encoder.get_current_time_high());
    EXPECT_EQ(0, t_encoder.get_next_timestamp_to_encode());

    t_encoder.initialize(50);
    EXPECT_EQ(0, t_encoder.get_current_time_high());
    EXPECT_EQ(0, t_encoder.get_next_timestamp_to_encode());

    t_encoder.initialize(102);
    EXPECT_EQ(64, t_encoder.get_current_time_high());
    EXPECT_EQ(64, t_encoder.get_next_timestamp_to_encode());

    t_encoder.initialize(353);
    EXPECT_EQ(320, t_encoder.get_current_time_high());
    EXPECT_EQ(320, t_encoder.get_next_timestamp_to_encode());

    t_encoder.initialize(6413);
    EXPECT_EQ(6400, t_encoder.get_current_time_high());
    EXPECT_EQ(6400, t_encoder.get_next_timestamp_to_encode());
}

TEST_F(TimerHighEncoder_GTest, evt2_redundancy_default_initialize) {
    // EVT1 TH STEP is 64, repeated every 16 us

    TimerHighEncoder<Evt2RawFormat, TimerHighRedundancyEvt2Default> t_encoder;
    EXPECT_EQ(0, t_encoder.get_current_time_high());
    EXPECT_EQ(0, t_encoder.get_next_timestamp_to_encode());

    t_encoder.initialize(1626);
    EXPECT_EQ(1600, t_encoder.get_current_time_high());
    EXPECT_EQ(1616, t_encoder.get_next_timestamp_to_encode());

    t_encoder.initialize(2048);
    EXPECT_EQ(2048, t_encoder.get_current_time_high());
    EXPECT_EQ(2048, t_encoder.get_next_timestamp_to_encode());

    t_encoder.initialize(4030);
    EXPECT_EQ(3968, t_encoder.get_current_time_high());
    EXPECT_EQ(4016, t_encoder.get_next_timestamp_to_encode());

    t_encoder.initialize(5800);
    EXPECT_EQ(5760, t_encoder.get_current_time_high());
    EXPECT_EQ(5792, t_encoder.get_next_timestamp_to_encode());
}

TEST_F(TimerHighEncoder_GTest, evt2_no_redundancy_encode) {
    TimerHighEncoder<Evt2RawFormat, TimerHighRedundancyNone> t_encoder;
    t_encoder.initialize(6413);

    ASSERT_EQ(sizeof(EventBase::RawEvent), t_encoder.get_size_encoded());
    EventBase::RawEvent th;

    t_encoder.encode_next_event(reinterpret_cast<uint8_t *>(&th));
    EXPECT_EQ(8, th.type);
    EXPECT_EQ(6400, (th.trail) << 6); // i.e EXPECT_EQ(100, th.trail);
    EXPECT_EQ(6400, t_encoder.get_current_time_high());
    EXPECT_EQ(6464, t_encoder.get_next_timestamp_to_encode());
}

TEST_F(TimerHighEncoder_GTest, evt2_redundancy_default_encode) {
    TimerHighEncoder<Evt2RawFormat, TimerHighRedundancyEvt2Default> t_encoder;
    t_encoder.initialize(6430);

    ASSERT_EQ(sizeof(EventBase::RawEvent), t_encoder.get_size_encoded());
    EventBase::RawEvent th;

    // Check we encode 3 times 6400 (because 6430is between the 2nd and 3rd repetition)
    t_encoder.encode_next_event(reinterpret_cast<uint8_t *>(&th));
    EXPECT_EQ(8, th.type);
    EXPECT_EQ(6400, (th.trail) << 6); // i.e EXPECT_EQ(100, th.trail);
    EXPECT_EQ(6400, t_encoder.get_current_time_high());
    EXPECT_EQ(6432, t_encoder.get_next_timestamp_to_encode());

    t_encoder.encode_next_event(reinterpret_cast<uint8_t *>(&th));
    EXPECT_EQ(8, th.type);
    EXPECT_EQ(6400, (th.trail) << 6); // i.e EXPECT_EQ(100, th.trail);
    EXPECT_EQ(6400, t_encoder.get_current_time_high());
    EXPECT_EQ(6448, t_encoder.get_next_timestamp_to_encode());
    t_encoder.encode_next_event(reinterpret_cast<uint8_t *>(&th));

    EXPECT_EQ(8, th.type);
    EXPECT_EQ(6400, (th.trail) << 6); // i.e EXPECT_EQ(100, th.trail);
    EXPECT_EQ(6400, t_encoder.get_current_time_high());
    EXPECT_EQ(6464, t_encoder.get_next_timestamp_to_encode());

    // Check we encode 4 times the same :

    t_encoder.encode_next_event(reinterpret_cast<uint8_t *>(&th));
    EXPECT_EQ(8, th.type);
    EXPECT_EQ(6464, (th.trail) << 6); // i.e EXPECT_EQ(101, th.trail);
    EXPECT_EQ(6464, t_encoder.get_current_time_high());
    EXPECT_EQ(6480, t_encoder.get_next_timestamp_to_encode());

    t_encoder.encode_next_event(reinterpret_cast<uint8_t *>(&th));
    EXPECT_EQ(8, th.type);
    EXPECT_EQ(6464, (th.trail) << 6); // i.e EXPECT_EQ(101, th.trail);
    EXPECT_EQ(6464, t_encoder.get_current_time_high());
    EXPECT_EQ(6496, t_encoder.get_next_timestamp_to_encode());

    t_encoder.encode_next_event(reinterpret_cast<uint8_t *>(&th));
    EXPECT_EQ(8, th.type);
    EXPECT_EQ(6464, (th.trail) << 6); // i.e EXPECT_EQ(101, th.trail);
    EXPECT_EQ(6464, t_encoder.get_current_time_high());
    EXPECT_EQ(6512, t_encoder.get_next_timestamp_to_encode());

    t_encoder.encode_next_event(reinterpret_cast<uint8_t *>(&th));
    EXPECT_EQ(8, th.type);
    EXPECT_EQ(6464, (th.trail) << 6); // i.e EXPECT_EQ(101, th.trail);
    EXPECT_EQ(6464, t_encoder.get_current_time_high());
    EXPECT_EQ(6528, t_encoder.get_next_timestamp_to_encode());

    // Check we update timestamp next time we encode :
    t_encoder.encode_next_event(reinterpret_cast<uint8_t *>(&th));
    EXPECT_EQ(8, th.type);
    EXPECT_EQ(6528, (th.trail) << 6); // i.e EXPECT_EQ(102, th.trail);
    EXPECT_EQ(6528, t_encoder.get_current_time_high());
    EXPECT_EQ(6544, t_encoder.get_next_timestamp_to_encode());
}
