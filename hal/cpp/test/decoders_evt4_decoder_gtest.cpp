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

#include "metavision/hal/decoders/base/event_base.h"
#include "metavision/hal/decoders/evt4/evt4_decoder.h"
#include "metavision/sdk/base/events/event_cd.h"

#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace Evt4Test {

using namespace Metavision;
using namespace ::testing;

using EventCdDecoder  = I_EventDecoder<EventCD>;
using EventExtDecoder = I_EventDecoder<EventExtTrigger>;
using EventErcDecoder = I_EventDecoder<EventERCCounter>;

using EventCdBuffer  = std::vector<EventCD>;
using EventExtBuffer = std::vector<EventExtTrigger>;
using EventErcBuffer = std::vector<EventERCCounter>;

using DataBuffer = std::vector<std::uint32_t>;

I_Decoder::RawData const *const begin(DataBuffer &buff) {
    return reinterpret_cast<I_Decoder::RawData *>(buff.data());
}

I_Decoder::RawData const *const end(DataBuffer &buff) {
    return reinterpret_cast<I_Decoder::RawData *>(buff.data() + buff.size());
}

std::uint32_t time_high(std::uint32_t time = 0) {
    EventBase::RawEvent ev_time{time, static_cast<std::uint32_t>(EVT4EventTypes::EVT_TIME_HIGH)};
    return *reinterpret_cast<std::uint32_t *>(&ev_time);
}

std::uint32_t event_cd(std::uint16_t x, std::uint16_t y, std::uint8_t ts, bool pol) {
    Evt4Raw::EVT4EventCD ev_cd{y, x, ts, std::uint32_t(pol ? EVT4EventTypes::CD_ON : EVT4EventTypes::CD_OFF)};
    return *reinterpret_cast<std::uint32_t *>(&ev_cd);
}

std::uint32_t event_cd_vec(std::uint16_t x, std::uint16_t y, std::uint8_t ts, bool pol) {
    Evt4Raw::EVT4EventCD ev_cd_vec{y, x, ts,
                                   std::uint32_t(pol ? EVT4EventTypes::CD_VEC_ON : EVT4EventTypes::CD_VEC_OFF)};
    return *reinterpret_cast<std::uint32_t *>(&ev_cd_vec);
}

std::uint32_t event_cd_vec_mask(std::uint32_t valid) {
    return valid;
}

std::uint32_t padding() {
    return 0xFFFFFFFF;
}

std::uint32_t raw_event(EVT4EventTypes raw_event_type, std::uint32_t data) {
    EventBase::RawEvent raw{data, static_cast<std::uint32_t>(raw_event_type)};
    return *reinterpret_cast<std::uint32_t *>(&raw);
}

std::uint32_t raw_event(std::uint32_t raw_event_type, std::uint32_t data) {
    EventBase::RawEvent raw{data, raw_event_type};
    return *reinterpret_cast<std::uint32_t *>(&raw);
}

using DecodedBuffers = std::tuple<EventCdBuffer, EventExtBuffer, EventErcBuffer>;

DecodedBuffers decode_buffer(DataBuffer &data, I_Decoder &decoder, EventCdDecoder &event_cd_decoder,
                             EventExtDecoder &event_ext_decoder, EventErcDecoder &event_erc_decoder) {
    std::vector<EventCD> event_cd_buffer;
    auto cb_cb_id = event_cd_decoder.add_event_buffer_callback(
        [&](auto beg, auto end) { std::copy(beg, end, std::back_inserter(event_cd_buffer)); });

    std::vector<EventExtTrigger> event_ext_buffer;
    auto ext_cb_id = event_ext_decoder.add_event_buffer_callback(
        [&](auto beg, auto end) { std::copy(beg, end, std::back_inserter(event_ext_buffer)); });

    std::vector<EventERCCounter> event_erc_buffer;
    auto erc_cb_id = event_erc_decoder.add_event_buffer_callback(
        [&](auto beg, auto end) { std::copy(beg, end, std::back_inserter(event_erc_buffer)); });

    decoder.decode(begin(data), end(data));

    event_cd_decoder.remove_callback(cb_cb_id);
    event_ext_decoder.remove_callback(ext_cb_id);
    event_erc_decoder.remove_callback(erc_cb_id);

    return std::make_tuple(event_cd_buffer, event_ext_buffer, event_erc_buffer);
}

TEST(Evt4Decoder, should_construct_evt4_decoder) {
    EXPECT_NO_THROW((EVT4Decoder{false}));
}

struct Evt4DecoderTest : public ::testing::Test {
    std::shared_ptr<EventCdDecoder> event_cd_decoder   = std::make_shared<EventCdDecoder>();
    std::shared_ptr<EventExtDecoder> event_ext_decoder = std::make_shared<EventExtDecoder>();
    std::shared_ptr<EventErcDecoder> event_erc_decoder = std::make_shared<EventErcDecoder>();

    EVT4Decoder decoder{false, std::nullopt, std::nullopt, event_cd_decoder, event_ext_decoder, event_erc_decoder};

    DecodedBuffers decode(DataBuffer &&data) {
        return decode_buffer(data, decoder, *event_cd_decoder, *event_ext_decoder, *event_erc_decoder);
    }

    template<class T>
    T decode(DataBuffer &&data) {
        auto events_buffers = decode_buffer(data, decoder, *event_cd_decoder, *event_ext_decoder, *event_erc_decoder);
        return std::get<T>(events_buffers);
    }
};

TEST_F(Evt4DecoderTest, should_decode_empty_evt4_stream) {
    auto events = decode<EventCdBuffer>({});
    EXPECT_EQ(events.size(), 0);
}

TEST_F(Evt4DecoderTest, should_decode_basic_evt4_stream) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        event_cd(3, 2, 0, false),
        event_cd(6, 5, 0, true),
    });

    const std::vector<EventCD> expected_events = {
        // x, y, p, t
        {3, 2, 0, 0},
        {6, 5, 1, 0},
    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(Evt4DecoderTest, should_drop_events_before_1st_timehigh) {
    auto events = decode<EventCdBuffer>({
        event_cd(3, 2, 1, false),
        event_cd(6, 5, 4, true),
        event_cd_vec(9, 8, 7, false),
        event_cd_vec_mask(0x11),
        event_cd_vec(12, 11, 10, true),
        event_cd_vec_mask(0x22),
        time_high(3),
        event_cd(5, 4, 1, false),
        event_cd(7, 6, 2, true),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, (3 << 6) + 1},
        {7, 6, 1, (3 << 6) + 2},
    };
    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(Evt4DecoderTest, should_decode_event_vect) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        event_cd_vec(5, 4, 0, false),
        event_cd_vec_mask(1 << 7 | 1 << 3 | 1),
        event_cd_vec(10, 6, 0, true),
        event_cd_vec_mask(1 << 14 | 1 << 10 | 1 << 4),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, 0}, {5 + 3, 4, 0, 0}, {5 + 7, 4, 0, 0}, {10 + 4, 6, 1, 0}, {10 + 10, 6, 1, 0}, {10 + 14, 6, 1, 0},
    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(Evt4DecoderTest, should_decode_event_timestamps) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        event_cd(5, 4, 5, false),
        event_cd(10, 6, 20, true),
        time_high(15),
        event_cd(5, 4, 5, false),
        event_cd(10, 6, 20, true),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, 5},
        {10, 6, 1, 20},
        {5, 4, 0, (15 << 6) + 5},
        {10, 6, 1, (15 << 6) + 20},
    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(Evt4DecoderTest, should_decode_event_timestamp_loop) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        event_cd(5, 4, 5, false),
        time_high((1ULL << 28) - 1),
        time_high((1ULL << 28) - 1),
        time_high(0),
        event_cd(5, 4, 5, false),
        time_high((1ULL << 28) - 1),
        time_high((1ULL << 28) - 1),
        time_high(0),
        event_cd(5, 4, 5, false),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, 5},
        {5, 4, 0, (1ULL << 34) + 5},
        {5, 4, 0, (2ULL << 34) + 5},

    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(Evt4DecoderTest, should_decode_negative_32bit_as_unsigned_timehigh) {
    // Timehigh with bit 25 set creates a timestamp with bit 32 set which is a negative value for signed ints
    auto events = decode<EventCdBuffer>({
        time_high(1ULL << 25),
        event_cd(5, 4, 5, false),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, ((1ULL << 25) << 6) + 5},
    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(Evt4DecoderTest, should_decode_cd_vec_accross_multiple_calls) {
    auto events_1 = decode<EventCdBuffer>({
        time_high(0),
        event_cd_vec(5, 4, 0, false),
        event_cd_vec_mask(1 << 7 | 1 << 3 | 1),
        event_cd_vec(10, 6, 0, true),
    });
    EXPECT_EQ(events_1.size(), 3);

    auto events_2 = decode<EventCdBuffer>({
        event_cd_vec_mask(1 << 14 | 1 << 10 | 1 << 4),
        event_cd_vec(15, 8, 0, false),
        event_cd_vec_mask(1 << 21 | 1 << 17 | 8),
    });
    EXPECT_EQ(events_2.size(), 6);
}

TEST_F(Evt4DecoderTest, should_skip_padding) {
    auto events = decode<EventCdBuffer>({
        padding(),
        time_high(0),
        padding(),
        event_cd(3, 2, 0, false),
        padding(),
        event_cd(6, 5, 0, true),
        padding(),
    });
    EXPECT_EQ(events.size(), 2);
}

TEST_F(Evt4DecoderTest, should_skip_unknown_type) {
    auto events = decode<EventCdBuffer>({
        raw_event(1, 0),
        time_high(0),
        raw_event(3, 2),
        event_cd(3, 2, 0, false),
        raw_event(5, 4),
        event_cd(6, 5, 0, true),
        padding(),
    });
    EXPECT_EQ(events.size(), 2);
}

TEST(UnsafeEVT4Decoder, should_construct_evt4_decoder) {
    EXPECT_NO_THROW((UnsafeEVT4Decoder{false}));
}

struct UnsafeEvt4DecoderTest : public ::testing::Test {
    std::shared_ptr<EventCdDecoder> event_cd_decoder   = std::make_shared<EventCdDecoder>();
    std::shared_ptr<EventExtDecoder> event_ext_decoder = std::make_shared<EventExtDecoder>();
    std::shared_ptr<EventErcDecoder> event_erc_decoder = std::make_shared<EventErcDecoder>();

    UnsafeEVT4Decoder decoder{
        false, std::nullopt, std::nullopt, event_cd_decoder, event_ext_decoder, event_erc_decoder};

    DecodedBuffers decode(DataBuffer &&data) {
        return decode_buffer(data, decoder, *event_cd_decoder, *event_ext_decoder, *event_erc_decoder);
    }

    template<class T>
    T decode(DataBuffer &&data) {
        auto events_buffers = decode_buffer(data, decoder, *event_cd_decoder, *event_ext_decoder, *event_erc_decoder);
        return std::get<T>(events_buffers);
    }
};

TEST_F(UnsafeEvt4DecoderTest, should_decode_empty_evt4_stream) {
    auto events = decode<EventCdBuffer>({});
    EXPECT_EQ(events.size(), 0);
}

TEST_F(UnsafeEvt4DecoderTest, should_decode_basic_evt4_stream) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        event_cd(3, 2, 0, false),
        event_cd(6, 5, 0, true),
    });

    const std::vector<EventCD> expected_events = {
        // x, y, p, t
        {3, 2, 0, 0},
        {6, 5, 1, 0},
    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(UnsafeEvt4DecoderTest, should_drop_events_before_1st_timehigh) {
    auto events = decode<EventCdBuffer>({
        event_cd(3, 2, 1, false),
        event_cd(6, 5, 4, true),
        event_cd_vec(9, 8, 7, false),
        event_cd_vec_mask(0x11),
        event_cd_vec(12, 11, 10, true),
        event_cd_vec_mask(0x22),
        time_high(3),
        event_cd(5, 4, 1, false),
        event_cd(7, 6, 2, true),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, (3 << 6) + 1},
        {7, 6, 1, (3 << 6) + 2},
    };
    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(UnsafeEvt4DecoderTest, should_decode_event_vect) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        event_cd_vec(5, 4, 0, false),
        event_cd_vec_mask(1 << 7 | 1 << 3 | 1),
        event_cd_vec(10, 6, 0, true),
        event_cd_vec_mask(1 << 14 | 1 << 10 | 1 << 4),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, 0}, {5 + 3, 4, 0, 0}, {5 + 7, 4, 0, 0}, {10 + 4, 6, 1, 0}, {10 + 10, 6, 1, 0}, {10 + 14, 6, 1, 0},
    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(UnsafeEvt4DecoderTest, should_decode_event_timestamps) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        event_cd(5, 4, 5, false),
        event_cd(10, 6, 20, true),
        time_high(15),
        event_cd(5, 4, 5, false),
        event_cd(10, 6, 20, true),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, 5},
        {10, 6, 1, 20},
        {5, 4, 0, (15 << 6) + 5},
        {10, 6, 1, (15 << 6) + 20},
    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(UnsafeEvt4DecoderTest, should_decode_event_timestamp_loop) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        event_cd(5, 4, 5, false),
        time_high((1ULL << 28) - 1),
        time_high(0),
        event_cd(5, 4, 5, false),
        time_high((1ULL << 28) - 1),
        time_high(0),
        event_cd(5, 4, 5, false),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, 5},
        {5, 4, 0, (1ULL << 34) + 5},
        {5, 4, 0, (2ULL << 34) + 5},

    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(UnsafeEvt4DecoderTest, should_decode_negative_32bit_as_unsigned_timehigh) {
    // Timehigh with bit 25 set creates a timestamp with bit 32 set which is a negative value for signed ints
    auto events = decode<EventCdBuffer>({
        time_high(1ULL << 25),
        event_cd(5, 4, 5, false),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, ((1ULL << 25) << 6) + 5},
    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(UnsafeEvt4DecoderTest, should_decode_cd_vec_accross_multiple_calls) {
    auto events_1 = decode<EventCdBuffer>({
        time_high(0),
        event_cd_vec(5, 4, 0, false),
        event_cd_vec_mask(1 << 7 | 1 << 3 | 1),
        event_cd_vec(10, 6, 0, true),
    });
    EXPECT_EQ(events_1.size(), 3);

    auto events_2 = decode<EventCdBuffer>({
        event_cd_vec_mask(1 << 14 | 1 << 10 | 1 << 4),
        event_cd_vec(15, 8, 0, false),
        event_cd_vec_mask(1 << 21 | 1 << 17 | 8),
    });
    EXPECT_EQ(events_2.size(), 6);
}

TEST_F(UnsafeEvt4DecoderTest, should_skip_padding) {
    auto events = decode<EventCdBuffer>({
        padding(),
        time_high(0),
        padding(),
        event_cd(3, 2, 0, false),
        padding(),
        event_cd(6, 5, 0, true),
        padding(),
    });
    EXPECT_EQ(events.size(), 2);
}

TEST_F(UnsafeEvt4DecoderTest, should_skip_unknown_type) {
    auto events = decode<EventCdBuffer>({
        raw_event(1, 0),
        time_high(0),
        raw_event(3, 2),
        event_cd(3, 2, 0, false),
        raw_event(5, 4),
        event_cd(6, 5, 0, true),
        padding(),
    });
    EXPECT_EQ(events.size(), 2);
}

TEST(RobustEvt4Decoder, should_construct_evt4_decoder) {
    EXPECT_NO_THROW((RobustEVT4Decoder{false}));
}

struct RobustEvt4DecoderTest : public ::testing::Test {
    std::shared_ptr<EventCdDecoder> event_cd_decoder   = std::make_shared<EventCdDecoder>();
    std::shared_ptr<EventExtDecoder> event_ext_decoder = std::make_shared<EventExtDecoder>();
    std::shared_ptr<EventErcDecoder> event_erc_decoder = std::make_shared<EventErcDecoder>();

    RobustEVT4Decoder decoder{false, 1280, 720, event_cd_decoder, event_ext_decoder, event_erc_decoder};

    DecodedBuffers decode(DataBuffer &&data) {
        return decode_buffer(data, decoder, *event_cd_decoder, *event_ext_decoder, *event_erc_decoder);
    }

    template<class T>
    T decode(DataBuffer &&data) {
        auto events_buffers = decode_buffer(data, decoder, *event_cd_decoder, *event_ext_decoder, *event_erc_decoder);
        return std::get<T>(events_buffers);
    }
};

TEST_F(RobustEvt4DecoderTest, should_decode_empty_evt4_stream) {
    auto events = decode<EventCdBuffer>({});
    EXPECT_EQ(events.size(), 0);
}

TEST_F(RobustEvt4DecoderTest, should_decode_basic_evt4_stream) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        event_cd(3, 2, 0, false),
        event_cd(6, 5, 0, true),
    });

    const std::vector<EventCD> expected_events = {
        // x, y, p, t
        {3, 2, 0, 0},
        {6, 5, 1, 0},
    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(RobustEvt4DecoderTest, should_drop_events_before_1st_timehigh) {
    auto events = decode<EventCdBuffer>({
        event_cd(3, 2, 1, false),
        event_cd(6, 5, 4, true),
        event_cd_vec(9, 8, 7, false),
        event_cd_vec_mask(0x11),
        event_cd_vec(12, 11, 10, true),
        event_cd_vec_mask(0x22),
        time_high(3),
        event_cd(5, 4, 1, false),
        event_cd(7, 6, 2, true),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, (3 << 6) + 1},
        {7, 6, 1, (3 << 6) + 2},
    };
    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(RobustEvt4DecoderTest, should_decode_event_vect) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        event_cd_vec(5, 4, 0, false),
        event_cd_vec_mask(1 << 7 | 1 << 3 | 1),
        event_cd_vec(10, 6, 0, true),
        event_cd_vec_mask(1 << 14 | 1 << 10 | 1 << 4),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, 0}, {5 + 3, 4, 0, 0}, {5 + 7, 4, 0, 0}, {10 + 4, 6, 1, 0}, {10 + 10, 6, 1, 0}, {10 + 14, 6, 1, 0},
    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(RobustEvt4DecoderTest, should_decode_event_timestamps) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        event_cd(5, 4, 5, false),
        event_cd(10, 6, 20, true),
        time_high(15),
        event_cd(5, 4, 5, false),
        event_cd(10, 6, 20, true),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, 5},
        {10, 6, 1, 20},
        {5, 4, 0, (15 << 6) + 5},
        {10, 6, 1, (15 << 6) + 20},
    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(RobustEvt4DecoderTest, should_decode_event_timestamp_loop) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        event_cd(5, 4, 5, false),
        time_high((1ULL << 28) - 1),
        time_high((1ULL << 28) - 1),
        time_high(0),
        event_cd(5, 4, 5, false),
        time_high((1ULL << 28) - 1),
        time_high((1ULL << 28) - 1),
        time_high(0),
        event_cd(5, 4, 5, false),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, 5},
        {5, 4, 0, (1ULL << 34) + 5},
        {5, 4, 0, (2ULL << 34) + 5},

    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(RobustEvt4DecoderTest, should_decode_negative_32bit_as_unsigned_timehigh) {
    // Timehigh with bit 25 set creates a timestamp with bit 32 set which is a negative value for signed ints
    auto events = decode<EventCdBuffer>({
        time_high(1ULL << 25),
        event_cd(5, 4, 5, false),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, ((1ULL << 25) << 6) + 5},
    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(RobustEvt4DecoderTest, should_decode_cd_vec_accross_multiple_calls) {
    auto events_1 = decode<EventCdBuffer>({
        time_high(0),
        event_cd_vec(5, 4, 0, false),
        event_cd_vec_mask(1 << 7 | 1 << 3 | 1),
        event_cd_vec(10, 6, 0, true),
    });
    EXPECT_EQ(events_1.size(), 3);

    auto events_2 = decode<EventCdBuffer>({
        event_cd_vec_mask(1 << 14 | 1 << 10 | 1 << 4),
        event_cd_vec(15, 8, 0, false),
        event_cd_vec_mask(1 << 21 | 1 << 17 | 8),
    });
    EXPECT_EQ(events_2.size(), 6);
}

TEST_F(RobustEvt4DecoderTest, should_skip_padding) {
    auto events = decode<EventCdBuffer>({
        padding(),
        time_high(0),
        padding(),
        event_cd(3, 2, 0, false),
        padding(),
        event_cd(6, 5, 0, true),
        padding(),
    });
    EXPECT_EQ(events.size(), 2);
}

TEST_F(RobustEvt4DecoderTest, should_skip_unknown_type) {
    auto events = decode<EventCdBuffer>({
        raw_event(1, 0),
        time_high(0),
        raw_event(3, 2),
        event_cd(3, 2, 0, false),
        raw_event(5, 4),
        event_cd(6, 5, 0, true),
        padding(),
    });
    EXPECT_EQ(events.size(), 2);
}

TEST_F(RobustEvt4DecoderTest, should_skip_out_of_bouds_events) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        event_cd(2, 1, 0, false),
        event_cd(4, 3, 0, true),
        event_cd(1280, 1, 0, false),
        event_cd(4, 720, 0, true),
        padding(),
    });
    EXPECT_EQ(events.size(), 2);
}

TEST_F(RobustEvt4DecoderTest, should_skip_out_of_bouds_events_vect) {
    auto events = decode<EventCdBuffer>({
        time_high(0),
        event_cd_vec(5, 4, 0, false),
        event_cd_vec_mask(1 << 7 | 1 << 3 | 1),
        event_cd_vec(10, 720, 0, true),
        event_cd_vec_mask(1 << 14 | 1 << 10 | 1 << 4),
        event_cd_vec(1280 - 31, 6, 0, true),
        event_cd_vec_mask(1 << 21 | 1 << 17 | 1 << 8),
        padding(),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, 0},
        {5 + 3, 4, 0, 0},
        {5 + 7, 4, 0, 0},
    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(RobustEvt4DecoderTest, should_skip_th_jump) {
    auto events_1 = decode<EventCdBuffer>({
        time_high(0),
        event_cd(2, 1, 0, false),
        time_high(100),
        event_cd(4, 3, 0, true),
        event_cd_vec(5, 4, 0, false),
        event_cd_vec_mask(1 << 7 | 1 << 3 | 1),
        event_cd_vec(10, 720, 0, true),
    });

    auto events_2 = decode<EventCdBuffer>({
        event_cd_vec_mask(1 << 14 | 1 << 10 | 1 << 4),
        time_high(100),
        event_cd(6, 5, 0, true),
        event_cd_vec(5, 4, 0, false),
        event_cd_vec_mask(1 << 7 | 1 << 3 | 1),
    });

    const std::vector<EventCD> expected_events_1 = {
        // x. y, p, t
        {2, 1, 0, 0},
    };

    const std::vector<EventCD> expected_events_2 = {
        // x. y, p, t
        {6, 5, 1, 6400},
        {5 + 0, 4, 0, 6400},
        {5 + 3, 4, 0, 6400},
        {5 + 7, 4, 0, 6400},
    };

    EXPECT_THAT(events_1, ContainerEq(expected_events_1));
    EXPECT_THAT(events_2, ContainerEq(expected_events_2));
}

} // namespace Evt4Test
