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

#include "metavision/hal/decoders/evt3/evt3_decoder.h"
#include "metavision/sdk/base/events/event_cd.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

using namespace Metavision;
using namespace ::testing;

using EventCdDecoder  = I_EventDecoder<EventCD>;
using EventExtDecoder = I_EventDecoder<EventExtTrigger>;
using EventErcDecoder = I_EventDecoder<EventERCCounter>;
using EventMonitoringDecoder = I_EventDecoder<EventMonitoring>;

using EventCdBuffer  = std::vector<EventCD>;
using EventExtBuffer = std::vector<EventExtTrigger>;
using EventErcBuffer = std::vector<EventERCCounter>;
using EventMonitoringBuffer = std::vector<EventMonitoring>;

using DataBuffer = std::vector<uint16_t>;

const I_Decoder::RawData *const cbegin(const DataBuffer &buff) {
    return reinterpret_cast<const I_Decoder::RawData *>(buff.data());
}

const I_Decoder::RawData *const cend(const DataBuffer &buff) {
    return reinterpret_cast<const I_Decoder::RawData *>(buff.data() + buff.size());
}

uint16_t time_high(uint16_t time = 0) {
    Evt3Raw::Event_Time e_time{time, uint8_t(Evt3EventTypes_4bits::EVT_TIME_HIGH)};
    return *reinterpret_cast<uint16_t *>(&e_time);
}

uint16_t addr_x(uint8_t x = 0, bool pol = 0) {
    Evt3Raw::Event_PosX e_x{x, pol, uint8_t(Evt3EventTypes_4bits::EVT_ADDR_X)};
    return *reinterpret_cast<uint16_t *>(&e_x);
}

uint16_t addr_y(uint8_t y = 0, bool orig = 0) {
    Evt3Raw::Event_Y e_y{y, orig, uint8_t(Evt3EventTypes_4bits::EVT_ADDR_Y)};
    return *reinterpret_cast<uint16_t *>(&e_y);
}

uint16_t xbase(uint8_t x = 0, bool pol = 0) {
    Evt3Raw::Event_XBase xbase{x, pol, uint8_t(Evt3EventTypes_4bits::VECT_BASE_X)};
    return *reinterpret_cast<uint16_t *>(&xbase);
}

uint16_t raw_event(Evt3EventTypes_4bits raw_event_type, uint16_t value) {
    Evt3Raw::RawEvent raw{value, uint8_t(raw_event_type)};
    return *reinterpret_cast<uint16_t *>(&raw);
}

uint16_t vect12(uint16_t value) {
    return raw_event(Metavision::Evt3EventTypes_4bits::VECT_12, value);
}

uint16_t vect8(uint16_t value) {
    return raw_event(Metavision::Evt3EventTypes_4bits::VECT_8, value);
}

namespace std {
template<class T>
ostream &operator<<(ostream &o, const vector<T> &evt) {
    std::copy(evt.cbegin(), evt.cend(), ostream_iterator<T>(o, "\n"));
    return o;
}
} // namespace std

using DecodedBuffers = std::tuple<EventCdBuffer, EventExtBuffer, EventErcBuffer, EventMonitoringBuffer>;

DecodedBuffers decode_buffer(const DataBuffer &data, I_Decoder &decoder, EventCdDecoder &event_cd_decoder,
                             EventExtDecoder &event_ext_decoder, EventErcDecoder &event_erc_decoder,
                             EventMonitoringDecoder &event_monitoring_decoder) {
    std::vector<EventCD> event_cd_buffer;
    auto cd_cb_id = event_cd_decoder.add_event_buffer_callback(
        [&](auto beg, auto end) { std::copy(beg, end, std::back_inserter(event_cd_buffer)); });

    std::vector<EventExtTrigger> event_ext_buffer;
    auto ext_cb_id = event_ext_decoder.add_event_buffer_callback(
        [&](auto beg, auto end) { std::copy(beg, end, std::back_inserter(event_ext_buffer)); });

    std::vector<EventERCCounter> event_erc_buffer;
    auto erc_cb_id = event_erc_decoder.add_event_buffer_callback(
        [&](auto beg, auto end) { std::copy(beg, end, std::back_inserter(event_erc_buffer)); });

    std::vector<EventMonitoring> event_monitoring_buffer;
    auto monitoring_cb_id = event_monitoring_decoder.add_event_buffer_callback(
        [&](auto beg, auto end) { std::copy(beg, end, std::back_inserter(event_monitoring_buffer)); });

    decoder.decode(cbegin(data), cend(data));

    event_cd_decoder.remove_callback(cd_cb_id);
    event_ext_decoder.remove_callback(ext_cb_id);
    event_erc_decoder.remove_callback(erc_cb_id);
    event_monitoring_decoder.remove_callback(monitoring_cb_id);

    return std::make_tuple(event_cd_buffer, event_ext_buffer, event_erc_buffer, event_monitoring_buffer);
}

TEST(Evt3_decoder, should_construct_evt3_decoder) {
    EXPECT_NO_THROW((EVT3Decoder{false, 100, 100}));
}

struct Evt3DecoderTest : public ::testing::Test {
    std::shared_ptr<EventCdDecoder> event_cd_decoder   = std::make_shared<EventCdDecoder>();
    std::shared_ptr<EventExtDecoder> event_ext_decoder = std::make_shared<EventExtDecoder>();
    std::shared_ptr<EventErcDecoder> event_erc_decoder = std::make_shared<EventErcDecoder>();
    std::shared_ptr<EventMonitoringDecoder> event_monitoring_decoder = std::make_shared<EventMonitoringDecoder>();

    EVT3Decoder decoder{false, 100, 100, event_cd_decoder, event_ext_decoder, event_erc_decoder,
        event_monitoring_decoder};

    template<class T>
    using StrictMockFunction = StrictMock<MockFunction<T>>;
    StrictMockFunction<void(DecoderProtocolViolation)> mock_protocol_violation;

    Evt3DecoderTest() {
        decoder.add_protocol_violation_callback(mock_protocol_violation.AsStdFunction());
    }

    DecodedBuffers decode(DataBuffer &&data) {
        return decode_buffer(data, decoder, *event_cd_decoder, *event_ext_decoder, *event_erc_decoder,
                             *event_monitoring_decoder);
    }

    template<class T>
    T decode(DataBuffer &&data) {
        auto events_buffers = decode_buffer(data, decoder, *event_cd_decoder, *event_ext_decoder, *event_erc_decoder,
                                            *event_monitoring_decoder);
        return std::get<T>(events_buffers);
    }
};

TEST_F(Evt3DecoderTest, should_decode_empty_evt3_stream) {
    auto events = decode<EventCdBuffer>({});
    EXPECT_EQ(events.size(), 0);
}

TEST_F(Evt3DecoderTest, should_decode_basic_evt3_stream) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        addr_y(1),
        addr_x(2, false),
        addr_x(3, true),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {2, 1, 0, 0},
        {3, 1, 1, 0},
    };
    EXPECT_THAT(events, ContainerEq(expected_events)) << "-- Actual events: \n"
                                                      << events << "-- Expected events: \n"
                                                      << expected_events;
}

TEST_F(Evt3DecoderTest, should_drop_events_before_1st_timehigh) {
    auto events = decode<EventCdBuffer>({
        addr_y(1),
        addr_x(2),
        time_high(3),
        addr_y(4),
        addr_x(5),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, 3 << 12},
    };
    EXPECT_THAT(events, ContainerEq(expected_events)) << "-- Actual events: \n"
                                                      << events << "-- Expected events: \n"
                                                      << expected_events;
}

TEST_F(Evt3DecoderTest, should_decode_non_monotonic_timehigh_by_default) {
    EXPECT_CALL(mock_protocol_violation, Call(DecoderProtocolViolation::NonMonotonicTimeHigh)).Times(1);

    auto events = decode<EventCdBuffer>({
        time_high(0),
        addr_y(1),
        addr_x(2),
        time_high(1),
        addr_x(3),
        time_high(0), // HERE We go back in time
        addr_x(4),
        time_high(1),
        addr_x(5),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {2, 1, 0, 0},
        {3, 1, 0, 1 << 12},
        {4, 1, 0, 0},
        {5, 1, 0, 1 << 12},
    };
    EXPECT_THAT(events, ContainerEq(expected_events)) << "-- Actual events: \n"
                                                      << events << "-- Expected events: \n"
                                                      << expected_events;
}

TEST_F(Evt3DecoderTest, should_decode_timehigh_count_jump) {
    auto events                                = decode<EventCdBuffer>({
        time_high(0),
        addr_y(1),
        addr_x(2),
        time_high(2), // <-- time high count jump
        addr_x(3),
        time_high(3),
        addr_x(4),
    });
    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {2, 1, 0, 0 << 12},
        {3, 1, 0, 2 << 12},
        {4, 1, 0, 3 << 12},
    };
    EXPECT_THAT(events, ContainerEq(expected_events)) << "-- Actual events: \n"
                                                      << events << "-- Expected events: \n"
                                                      << expected_events;
}

TEST_F(Evt3DecoderTest, should_not_raise_error_with_timehigh_gap_on_overflow) {
    DataBuffer data;

    // We send Time high from 0 until 4093 (0xFFD)
    for (int i = 0; i <= 0xFFD; i++) {
        data.emplace_back(time_high(i));
    }
    data.insert(data.end(), {
                                addr_y(1),
                                addr_x(2),
                                time_high(1), // Flip around time_high counter
                                addr_x(3),    // perfeclyt valid event
                                time_high(2),
                                addr_x(4),
                            });
    auto events = decode<EventCdBuffer>(std::move(data));

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {2, 1, 0, 0xFFD << 12},
        {3, 1, 0, 0x1001 << 12},
        {4, 1, 0, 0x1002 << 12},
    };
    EXPECT_THAT(events, ContainerEq(expected_events)) << "-- Actual events: \n"
                                                      << events << "-- Expected events: \n"
                                                      << expected_events;
}

TEST_F(Evt3DecoderTest, should_raise_error_with_timehigh_gap_on_overflow_larger_than_epsilon) {
    EXPECT_CALL(mock_protocol_violation, Call(DecoderProtocolViolation::NonMonotonicTimeHigh)).Times(1);

    DataBuffer data;

    // We send Time high from 0 until 4093 (0xFFD)
    for (int i = 0; i <= 0xFFF; i++) {
        data.emplace_back(time_high(i));
    }
    data.insert(data.end(),
                {
                    addr_y(1), addr_x(2), time_high(EVT3Decoder::ValidatorType::LOOSE_TIME_HIGH_OVERFLOW_EPSILON),
                    addr_x(3), // perfeclyt valid event
                });
    auto events = decode<EventCdBuffer>(std::move(data));
}

TEST_F(Evt3DecoderTest, should_decode_vect_12_12_8_words) {
    auto events = decode<EventCdBuffer>({
        time_high(0), addr_y(1), xbase(1),
        vect12(0xFFFF), // 12 events
        vect12(0xFFFF), // 12 events
        vect8(0xFF),    // 8 events
    });

    EXPECT_EQ(events.size(), 12 + 12 + 8);
}

TEST_F(Evt3DecoderTest, should_decode_partial_vect_12_12_8_words) {
    auto events = decode<EventCdBuffer>({
        time_high(0), addr_y(1), xbase(1),
        vect12(0xFFFF), // 12 events
        addr_x(2),      // 1 event
        vect8(0xFF),    // 8 events
    });

    EXPECT_EQ(events.size(), 12 + 1 + 8);
}

TEST_F(Evt3DecoderTest, should_not_decode_vect8_unique_word) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        addr_y(1),
        xbase(1),
        vect8(0xFF),
        addr_x(2),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {2, 1, 0, 0},
    };
    EXPECT_THAT(events, ContainerEq(expected_events)) << "-- Actual events: \n"
                                                      << events << "-- Expected events: \n"
                                                      << expected_events;
}

TEST_F(Evt3DecoderTest, should_decode_vect_12_12_8_accross_multiple_calls) {
    auto events_1 = decode<EventCdBuffer>({
        time_high(0), addr_y(1), xbase(1),
        vect12(0xFFFF), // 12 events
    });
    EXPECT_EQ(events_1.size(), 0);

    auto events_2 = decode<EventCdBuffer>({
        vect12(0xFFFF), // 12 events
        vect8(0xFF),    // 8 events
    });
    EXPECT_EQ(events_2.size(), 12 + 12 + 8);
}

TEST_F(Evt3DecoderTest, should_decode_continue_12_12_4_words) {
    auto events = decode<EventErcBuffer>({
        time_high(0),
        raw_event(Evt3EventTypes_4bits::OTHERS, uint16_t(Evt3MasterEventTypes::MASTER_IN_CD_EVENT_COUNT)),
        raw_event(Evt3EventTypes_4bits::CONTINUED_12, 0x123),
        raw_event(Evt3EventTypes_4bits::CONTINUED_12, 0x456),
        raw_event(Evt3EventTypes_4bits::CONTINUED_4, 0x7),
    });

    const std::vector<EventERCCounter> expected_events = {
        // t, count,                         output
        {0, 0x7 << 24 | 0x456 << 12 | 0x123, 0},
    };
    EXPECT_THAT(events, ContainerEq(expected_events)) << "-- Actual events: \n"
                                                      << events << "-- Expected events: \n"
                                                      << expected_events;
}

using Metavision::DecoderProtocolViolation;

struct Evt3RobustDecoderTest : public ::testing::Test {
    std::shared_ptr<EventCdDecoder> event_cd_decoder   = std::make_shared<EventCdDecoder>();
    std::shared_ptr<EventExtDecoder> event_ext_decoder = std::make_shared<EventExtDecoder>();
    std::shared_ptr<EventErcDecoder> event_erc_decoder = std::make_shared<EventErcDecoder>();
    std::shared_ptr<EventMonitoringDecoder> event_monitoring_decoder = std::make_shared<EventMonitoringDecoder>();

    RobustEVT3Decoder robust_decoder{false, 100, 100, event_cd_decoder, event_ext_decoder, event_erc_decoder,
        event_monitoring_decoder};

    template<class T>
    using StrictMockFunction = StrictMock<MockFunction<T>>;
    StrictMockFunction<void(DecoderProtocolViolation)> mock_protocol_violation;

    Evt3RobustDecoderTest() {
        robust_decoder.add_protocol_violation_callback(mock_protocol_violation.AsStdFunction());
    }

    DecodedBuffers decode(DataBuffer &&data) {
        return decode_buffer(data, robust_decoder, *event_cd_decoder, *event_ext_decoder, *event_erc_decoder,
                             *event_monitoring_decoder);
    }

    template<class T>
    T decode(DataBuffer &&data) {
        auto events_buffers = decode_buffer(data, robust_decoder, *event_cd_decoder, *event_ext_decoder,
                                            *event_erc_decoder, *event_monitoring_decoder);
        return std::get<T>(events_buffers);
    }
};

TEST_F(Evt3RobustDecoderTest, should_not_validate_time_high_when_uninitialised) {
    auto events = decode<EventCdBuffer>({
        addr_y(1),
        addr_x(2, false),
    });

    const std::vector<EventCD> expected_events;
    EXPECT_THAT(events, ContainerEq(expected_events)) << "-- Actual events: \n"
                                                      << events << "-- Expected events: \n"
                                                      << expected_events;
}

TEST_F(Evt3RobustDecoderTest, should_decode_basic_evt3_stream) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        addr_y(1),
        addr_x(2, false),
        addr_x(3, true),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {2, 1, 0, 0},
        {3, 1, 1, 0},
    };
    EXPECT_THAT(events, ContainerEq(expected_events)) << "-- Actual events: \n"
                                                      << events << "-- Expected events: \n"
                                                      << expected_events;
}

TEST_F(Evt3RobustDecoderTest, should_not_decode_non_monotonic_timehigh_by_default) {
    EXPECT_CALL(mock_protocol_violation, Call(DecoderProtocolViolation::NonMonotonicTimeHigh)).Times(1);

    auto events = decode<EventCdBuffer>({
        time_high(0),
        addr_y(1),
        addr_x(2),
        time_high(1),
        addr_x(3),
        time_high(0), // HERE We go back in time
        addr_x(4),    // This event should be dropped
        time_high(1),
        addr_x(5),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {2, 1, 0, 0 << 12},
        {3, 1, 0, 1 << 12},
        // {4, 1, 0, 0 << 12 }, <-- dropped event as timestamp goes back in time
        {5, 1, 0, 1 << 12},
    };
    EXPECT_THAT(events, ContainerEq(expected_events)) << "-- Actual events: \n"
                                                      << events << "-- Expected events: \n"
                                                      << expected_events;
}

TEST_F(Evt3RobustDecoderTest, should_detect_timehigh_count_jump) {
    EXPECT_CALL(mock_protocol_violation, Call(DecoderProtocolViolation::NonContinuousTimeHigh)).Times(1);

    auto events                                = decode<EventCdBuffer>({
        time_high(0),
        addr_y(1),
        addr_x(2),
        time_high(2), // <-- time high count jump
        addr_x(3),
        time_high(3),
        addr_x(4),
    });
    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {2, 1, 0, 0 << 12},
        {3, 1, 0, 2 << 12},
        {4, 1, 0, 3 << 12},
    };
    EXPECT_THAT(events, ContainerEq(expected_events)) << "-- Actual events: \n"
                                                      << events << "-- Expected events: \n"
                                                      << expected_events;
}

TEST_F(Evt3RobustDecoderTest, should_handle_timehigh_overflow) {
    DataBuffer data;

    for (int i = 0; i <= 0xFFF; i++) {
        data.emplace_back(time_high(i));
    }
    data.insert(data.end(), {
                                addr_y(1),
                                addr_x(2),
                                time_high(0), // Flip around time_high counter
                                addr_x(3),    // perfeclyt valid event
                                time_high(1),
                                addr_x(4),
                            });
    auto events = decode<EventCdBuffer>(std::move(data));

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {2, 1, 0, 0xFFF << 12},
        {3, 1, 0, 0x1000 << 12},
        {4, 1, 0, 0x1001 << 12},
    };
    EXPECT_THAT(events, ContainerEq(expected_events)) << "-- Actual events: \n"
                                                      << events << "-- Expected events: \n"
                                                      << expected_events;
}

TEST_F(Evt3RobustDecoderTest, should_raise_non_countinous_with_timehigh_gap_on_overflow) {
    EXPECT_CALL(mock_protocol_violation, Call(DecoderProtocolViolation::NonContinuousTimeHigh)).Times(1);
    DataBuffer data;

    // We send Time high from 0 until 4093 (0xFFD)
    for (int i = 0; i <= 0xFFD; i++) {
        data.emplace_back(time_high(i));
    }
    data.insert(data.end(), {
                                addr_y(1),
                                addr_x(2),
                                time_high(1), // Flip around time_high counter
                                addr_x(3),    // perfeclyt valid event
                                time_high(2),
                                addr_x(4),
                            });
    auto events = decode<EventCdBuffer>(std::move(data));

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {2, 1, 0, 0xFFD << 12},
        {3, 1, 0, 0x1001 << 12},
        {4, 1, 0, 0x1002 << 12},
    };
    EXPECT_THAT(events, ContainerEq(expected_events)) << "-- Actual events: \n"
                                                      << events << "-- Expected events: \n"
                                                      << expected_events;
}

TEST_F(Evt3RobustDecoderTest, should_decode_vect_12_12_8_words) {
    auto events = decode<EventCdBuffer>({
        time_high(0), addr_y(1), xbase(1),
        vect12(0xFFFF), // 12 events
        vect12(0xFFFF), // 12 events
        vect8(0xFF),    // 8 events
    });

    EXPECT_EQ(events.size(), 12 + 12 + 8);
}

TEST_F(Evt3RobustDecoderTest, should_NOT_decode_partial_vect_12_12_8_words) {
    EXPECT_CALL(mock_protocol_violation, Call(DecoderProtocolViolation::PartialVect_12_12_8)).Times(1);

    auto events = decode<EventCdBuffer>({
        time_high(0),
        addr_y(1),
        xbase(1),
        vect12(0xFFFF),
        addr_y(2), // unexpected event ! should be vect_12
        vect8(0xFF),
    });

    //  vect_12_12_8 pattern is broken, the whole 3 words are dropped
    EXPECT_EQ(events.size(), 0);
}

TEST_F(Evt3RobustDecoderTest, should_skip_single_vect8) {
    auto events = decode<EventCdBuffer>({
        time_high(0), addr_y(1), xbase(1),
        vect8(0xFF), // Missing 2 vect12, hence ignored
        addr_x(2),   // This event is valid
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {2, 1, 0, 0},
    };
    EXPECT_THAT(events, ContainerEq(expected_events)) << "-- Actual events: \n"
                                                      << events << "-- Expected events: \n"
                                                      << expected_events;
}

TEST_F(Evt3RobustDecoderTest, should_decode_continue_12_12_4_words) {
    auto events = decode<EventErcBuffer>({
        time_high(0),
        raw_event(Evt3EventTypes_4bits::OTHERS, uint16_t(Evt3MasterEventTypes::MASTER_IN_CD_EVENT_COUNT)),
        raw_event(Evt3EventTypes_4bits::CONTINUED_12, 0x123),
        raw_event(Evt3EventTypes_4bits::CONTINUED_12, 0x456),
        raw_event(Evt3EventTypes_4bits::CONTINUED_4, 0x7),
    });

    const std::vector<EventERCCounter> expected_events = {
        // t, count,                         output
        {0, 0x7 << 24 | 0x456 << 12 | 0x123, 0},
    };
    EXPECT_THAT(events, ContainerEq(expected_events)) << "-- Actual events: \n"
                                                      << events << "-- Expected events: \n"
                                                      << expected_events;
}

TEST_F(Evt3RobustDecoderTest, should_not_decode_partial_continue_12_12_4) {
    EXPECT_CALL(mock_protocol_violation, Call(DecoderProtocolViolation::PartialContinued_12_12_4)).Times(1);

    auto events = decode<EventErcBuffer>({
        time_high(0),
        raw_event(Evt3EventTypes_4bits::OTHERS, uint16_t(Evt3MasterEventTypes::MASTER_IN_CD_EVENT_COUNT)),
        raw_event(Evt3EventTypes_4bits::CONTINUED_12, 0x123),
        addr_x(0),
        raw_event(Evt3EventTypes_4bits::CONTINUED_12, 0x456),
        raw_event(Evt3EventTypes_4bits::CONTINUED_4, 0x7),
    });

    EXPECT_EQ(events.size(), 0);
}
