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

#include "metavision/hal/decoders/evt21/evt21_decoder.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_cd_vector.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace Metavision;
using namespace ::testing;

template<typename EventCDType>
using EventCdDecoder = I_EventDecoder<EventCDType>;

using EventExtDecoder = I_EventDecoder<EventExtTrigger>;
using EventErcDecoder = I_EventDecoder<EventERCCounter>;

template<typename EventCDType>
using EventCdBuffer = std::vector<EventCDType>;

using EventExtBuffer = std::vector<EventExtTrigger>;
using EventErcBuffer = std::vector<EventERCCounter>;

using DataBuffer = std::vector<uint64_t>;

I_Decoder::RawData const *const begin(DataBuffer &buff) {
    return reinterpret_cast<I_Decoder::RawData *>(buff.data());
}

I_Decoder::RawData const *const end(DataBuffer &buff) {
    return reinterpret_cast<I_Decoder::RawData *>(buff.data() + buff.size());
}

uint64_t time_high(uint32_t time = 0) {
    Evt21Raw::Event_TIME_HIGH e_time{0, time, uint8_t(Evt21EventTypes_4bits::EVT_TIME_HIGH)};
    return *reinterpret_cast<uint64_t *>(&e_time);
}

uint64_t event_2d(uint16_t x, uint16_t y, uint8_t ts, bool pol, uint32_t valid_vect = 0x1) {
    Evt21Raw::Event_2D e_2d{valid_vect, y, x, ts,
                            uint8_t(pol ? Evt21EventTypes_4bits::EVT_POS : Evt21EventTypes_4bits::EVT_NEG)};
    return *reinterpret_cast<uint64_t *>(&e_2d);
}

template<typename EventCDType>
using DecodedBuffers = std::tuple<EventCdBuffer<EventCDType>, EventExtBuffer, EventErcBuffer>;

template<typename EventCDType>
DecodedBuffers<EventCDType> decode_buffer(DataBuffer &data, I_Decoder &decoder,
                                          EventCdDecoder<EventCDType> &event_cd_decoder,
                                          EventExtDecoder &event_ext_decoder, EventErcDecoder &event_erc_decoder) {
    std::vector<EventCDType> event_cd_buffer;
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

TEST(Evt21_decoder, should_construct_evt21_decoder) {
    EXPECT_NO_THROW((EVT21Decoder{false}));
}

TEST(Evt21_vector_decoder, should_construct_evt21_vector_decoder) {
    EXPECT_NO_THROW((EVT21VectorizedDecoder{false}));
}

struct Evt21DecoderTest : public ::testing::Test {
    std::shared_ptr<EventCdDecoder<EventCD>> event_cd_decoder = std::make_shared<EventCdDecoder<EventCD>>();
    std::shared_ptr<EventExtDecoder> event_ext_decoder        = std::make_shared<EventExtDecoder>();
    std::shared_ptr<EventErcDecoder> event_erc_decoder        = std::make_shared<EventErcDecoder>();

    EVT21Decoder decoder{false, event_cd_decoder, event_ext_decoder, event_erc_decoder};

    DecodedBuffers<EventCD> decode(DataBuffer &&data) {
        return decode_buffer<EventCD>(data, decoder, *event_cd_decoder, *event_ext_decoder, *event_erc_decoder);
    }

    template<class T>
    T decode(DataBuffer &&data) {
        auto events_buffers =
            decode_buffer<EventCD>(data, decoder, *event_cd_decoder, *event_ext_decoder, *event_erc_decoder);
        return std::get<T>(events_buffers);
    }
};

TEST_F(Evt21DecoderTest, should_decode_empty_evt21_stream) {
    auto events = decode<EventCdBuffer<EventCD>>({});
    EXPECT_EQ(events.size(), 0);
}

TEST_F(Evt21DecoderTest, should_decode_basic_evt21_stream) {
    auto events = decode<EventCdBuffer<EventCD>>({
        time_high(0),
        event_2d(2, 1, 0, false),
        event_2d(3, 1, 0, true),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {2, 1, 0, 0},
        {3, 1, 1, 0},
    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(Evt21DecoderTest, should_drop_events_before_1st_timehigh) {
    auto events = decode<EventCdBuffer<EventCD>>({
        event_2d(2, 1, 0, false),
        time_high(3),
        event_2d(5, 4, 0, false),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, 3 << 6},
    };
    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(Evt21DecoderTest, should_decode_event_vect) {
    auto events = decode<EventCdBuffer<EventCD>>({
        time_high(0),
        event_2d(5, 4, 0, false, 1 << 7 | 1 << 3 | 1),
        event_2d(10, 6, 0, true, 1 << 14 | 1 << 10 | 1 << 4),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, 0}, {5 + 3, 4, 0, 0}, {5 + 7, 4, 0, 0}, {10 + 4, 6, 1, 0}, {10 + 10, 6, 1, 0}, {10 + 14, 6, 1, 0},
    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(Evt21DecoderTest, should_decode_event_timestamps) {
    auto events = decode<EventCdBuffer<EventCD>>({
        time_high(0),
        event_2d(5, 4, 5, false),
        event_2d(10, 6, 20, true),
        time_high(15),
        event_2d(5, 4, 5, false),
        event_2d(10, 6, 20, true),
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

TEST_F(Evt21DecoderTest, should_decode_event_timestamp_loop) {
    auto events = decode<EventCdBuffer<EventCD>>({
        time_high(0),
        event_2d(5, 4, 5, false),
        time_high((1ULL << 28) - 1),
        time_high(0),
        event_2d(5, 4, 5, false),
        time_high((1ULL << 28) - 1),
        time_high(0),
        event_2d(5, 4, 5, false),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, 5},
        {5, 4, 0, (1ULL << 34) + 5},
        {5, 4, 0, (2ULL << 34) + 5},

    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(Evt21DecoderTest, should_decode_negative_32bit_as_unsigned_timehigh) {
    // Timehigh with bit 25 set creates a timestamp with bit 32 set which is a negative value for signed ints
    auto events = decode<EventCdBuffer<EventCD>>({
        time_high(1ULL << 25),
        event_2d(5, 4, 5, false),
    });

    const std::vector<EventCD> expected_events = {
        // x. y, p, t
        {5, 4, 0, ((1ULL << 25) << 6) + 5},
    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

struct Evt21VectorDecoderTest : public ::testing::Test {
    std::shared_ptr<EventCdDecoder<EventCDVector>> event_cd_decoder = std::make_shared<EventCdDecoder<EventCDVector>>();
    std::shared_ptr<EventExtDecoder> event_ext_decoder              = std::make_shared<EventExtDecoder>();
    std::shared_ptr<EventErcDecoder> event_erc_decoder              = std::make_shared<EventErcDecoder>();

    EVT21VectorizedDecoder decoder{false, event_cd_decoder, event_ext_decoder, event_erc_decoder};

    DecodedBuffers<EventCDVector> decode(DataBuffer &&data) {
        return decode_buffer<EventCDVector>(data, decoder, *event_cd_decoder, *event_ext_decoder, *event_erc_decoder);
    }

    template<class T>
    T decode(DataBuffer &&data) {
        auto events_buffers =
            decode_buffer<EventCDVector>(data, decoder, *event_cd_decoder, *event_ext_decoder, *event_erc_decoder);
        return std::get<T>(events_buffers);
    }
};

TEST_F(Evt21VectorDecoderTest, should_decode_empty_evt21_stream) {
    auto events = decode<EventCdBuffer<EventCDVector>>({});
    EXPECT_EQ(events.size(), 0);
}

TEST_F(Evt21VectorDecoderTest, should_decode_basic_evt21_stream) {
    auto events = decode<EventCdBuffer<EventCDVector>>({
        time_high(0),
        event_2d(2, 1, 0, false),
        event_2d(3, 1, 0, true),
    });

    const std::vector<EventCDVector> expected_events = {// base_x, y, polarity, vector_mask, event_timestamp
                                                        {2, 1, false, 1U, 0},
                                                        {3, 1, true, 1U, 0}};

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(Evt21VectorDecoderTest, should_drop_events_before_1st_timehigh) {
    auto events = decode<EventCdBuffer<EventCDVector>>({
        event_2d(2, 1, 0, false),
        time_high(3),
        event_2d(5, 4, 0, false),
    });

    const std::vector<EventCDVector> expected_events = {// base_x, y, polarity, vector_mask, event_timestamp
                                                        {5, 4, false, 1U, 3 << 6}};
    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(Evt21VectorDecoderTest, should_decode_event_vect) {
    auto events = decode<EventCdBuffer<EventCDVector>>({
        time_high(0),
        event_2d(5, 4, 0, false, (1 << 7 | 1 << 3 | 1)),
        event_2d(10, 6, 0, true, (1 << 14 | 1 << 10 | 1 << 4)),
    });

    const std::vector<EventCDVector> expected_events = {// base_x, y, polarity, vector_mask, event_timestamp
                                                        {5, 4, false, (1 << 7 | 1 << 3 | 1), 0},
                                                        {10, 6, true, (1 << 14 | 1 << 10 | 1 << 4), 0}};

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(Evt21VectorDecoderTest, should_decode_event_timestamps) {
    auto events = decode<EventCdBuffer<EventCDVector>>({
        time_high(0),
        event_2d(5, 4, 5, false),
        event_2d(10, 6, 20, true),
        time_high(15),
        event_2d(5, 4, 5, false),
        event_2d(10, 6, 20, true),
    });

    const std::vector<EventCDVector> expected_events = {// base_x, y, polarity, vector_mask, event_timestamp
                                                        {5, 4, false, 1U, 5},
                                                        {10, 6, true, 1U, 20},
                                                        {5, 4, false, 1U, (15 << 6) + 5},
                                                        {10, 6, true, 1U, (15 << 6) + 20}};

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(Evt21VectorDecoderTest, should_decode_event_timestamp_loop) {
    auto events = decode<EventCdBuffer<EventCDVector>>({
        time_high(0),
        event_2d(5, 4, 5, false),
        time_high((1ULL << 28) - 1),
        time_high(0),
        event_2d(5, 4, 5, false),
        time_high((1ULL << 28) - 1),
        time_high(0),
        event_2d(5, 4, 5, false),
    });

    const std::vector<EventCDVector> expected_events = {// base_x, y, polarity, vector_mask, event_timestamp
                                                        {5, 4, false, 1U, 5},
                                                        {5, 4, false, 1U, (1ULL << 34) + 5},
                                                        {5, 4, false, 1U, (2ULL << 34) + 5}

    };

    EXPECT_THAT(events, ContainerEq(expected_events));
}

TEST_F(Evt21VectorDecoderTest, should_decode_negative_32bit_as_unsigned_timehigh) {
    // Timehigh with bit 25 set creates a timestamp with bit 32 set which is a negative value for signed ints
    auto events = decode<EventCdBuffer<EventCDVector>>({
        time_high(1ULL << 25),
        event_2d(5, 4, 5, false),
    });

    const std::vector<EventCDVector> expected_events = {// base_x, y, polarity, vector_mask, event_timestamp
                                                        {5, 4, false, 1U, ((1ULL << 25) << 6) + 5}};

    EXPECT_THAT(events, ContainerEq(expected_events));
}
