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
#include <array>

#include "metavision/sdk/base/events/event_cd.h"
#include "event_encoder.h"
#include "evt2_raw_format.h"

using namespace Metavision;

class EventEncoders_GTest : public ::testing::Test {
protected:
    virtual void SetUp() override {}

    virtual void TearDown() override {}
};

TEST_F(EventEncoders_GTest, get_size_encoded) {
    // Event2d Cd
    EXPECT_EQ(4, BatchEventEncoder<EventCD *>::get_size_encoded<Evt2RawFormat>());

    // Ext Triggers
    EXPECT_EQ(4, BatchEventEncoder<EventExtTrigger *>::get_size_encoded<Evt2RawFormat>());
}

TEST_F(EventEncoders_GTest, register_empty_buffer) {
    std::vector<EventCD> v;
    auto it = v.begin(), it_end = v.end();
    BatchEventEncoder<decltype(it)> event_encoder;
    event_encoder.register_buffer(it, it_end);
    EXPECT_EQ(std::numeric_limits<timestamp>::max(), event_encoder.get_next_timestamp_to_encode());
    EXPECT_TRUE(event_encoder.is_done());
}

TEST_F(EventEncoders_GTest, register_buffer) {
    std::vector<EventCD> v = {{1, 2, 0, 153}, {456, 275, 1, 207}, {45, 180, 1, 510}};
    auto it = v.begin(), it_end = v.end();
    BatchEventEncoder<decltype(it)> event_encoder;
    event_encoder.register_buffer(it, it_end);
    EXPECT_EQ(153, event_encoder.get_next_timestamp_to_encode());
    EXPECT_FALSE(event_encoder.is_done());
}

TEST_F(EventEncoders_GTest, evt2_encode_cd) {
    using Format  = Evt2RawFormat;
    int time_mask = 0x3F; // Take only 6 lower bits

    std::vector<EventCD> v = {{1, 2, 0, 153}, {456, 255, 1, 2407}, {45, 180, 1, 10010}};
    auto it = v.begin(), it_end = v.end();
    BatchEventEncoder<decltype(it)> event_encoder;
    event_encoder.register_buffer(it, it_end);

    ASSERT_EQ(sizeof(EVT2Event2D), event_encoder.get_size_encoded<Format>());
    EVT2Event2D encoded_ev;

    event_encoder.encode_next_event<Format>(reinterpret_cast<uint8_t *>(&encoded_ev));
    EXPECT_EQ(1, encoded_ev.x);
    EXPECT_EQ(2, encoded_ev.y);
    EXPECT_EQ(0, encoded_ev.type); // Polarity 0
    EXPECT_EQ(153 & time_mask, encoded_ev.timestamp);

    EXPECT_EQ(2407, event_encoder.get_next_timestamp_to_encode());

    event_encoder.encode_next_event<Format>(reinterpret_cast<uint8_t *>(&encoded_ev));
    EXPECT_EQ(456, encoded_ev.x);
    EXPECT_EQ(255, encoded_ev.y);
    EXPECT_EQ(1, encoded_ev.type); // Polarity 1
    EXPECT_EQ(2407 & time_mask, encoded_ev.timestamp);

    EXPECT_EQ(10010, event_encoder.get_next_timestamp_to_encode());

    event_encoder.encode_next_event<Format>(reinterpret_cast<uint8_t *>(&encoded_ev));
    EXPECT_EQ(45, encoded_ev.x);
    EXPECT_EQ(180, encoded_ev.y);
    EXPECT_EQ(1, encoded_ev.type); // Polarity 1
    EXPECT_EQ(10010 & time_mask, encoded_ev.timestamp);
}

TEST_F(EventEncoders_GTest, evt2_encode_ext_triggers) {
    using Format  = Evt2RawFormat;
    int time_mask = 0x3F; // Take only 6 lower bits

    std::vector<EventExtTrigger> v = {EventExtTrigger(1, 3000, 2), EventExtTrigger(0, 6000, 1),
                                      EventExtTrigger(1, 9000, 3)};
    auto it = v.begin(), it_end = v.end();
    BatchEventEncoder<decltype(it)> event_encoder;
    event_encoder.register_buffer(it, it_end);

    ASSERT_EQ(sizeof(EVT2EventExtTrigger), event_encoder.get_size_encoded<Format>());
    EVT2EventExtTrigger encoded_ev;

    event_encoder.encode_next_event<Format>(reinterpret_cast<uint8_t *>(&encoded_ev));
    EXPECT_EQ(1, encoded_ev.value);
    EXPECT_EQ(2, encoded_ev.id);
    EXPECT_EQ(10, encoded_ev.type); // Polarity 0
    EXPECT_EQ(3000 & time_mask, encoded_ev.timestamp);

    EXPECT_EQ(6000, event_encoder.get_next_timestamp_to_encode());

    event_encoder.encode_next_event<Format>(reinterpret_cast<uint8_t *>(&encoded_ev));
    EXPECT_EQ(0, encoded_ev.value);
    EXPECT_EQ(1, encoded_ev.id);
    EXPECT_EQ(10, encoded_ev.type); // Polarity 1
    EXPECT_EQ(6000 & time_mask, encoded_ev.timestamp);

    EXPECT_EQ(9000, event_encoder.get_next_timestamp_to_encode());

    event_encoder.encode_next_event<Format>(reinterpret_cast<uint8_t *>(&encoded_ev));
    EXPECT_EQ(1, encoded_ev.value);
    EXPECT_EQ(3, encoded_ev.id);
    EXPECT_EQ(10, encoded_ev.type); // Polarity 1
    EXPECT_EQ(9000 & time_mask, encoded_ev.timestamp);
}
