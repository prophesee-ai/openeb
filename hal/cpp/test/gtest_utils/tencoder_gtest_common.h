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

#ifndef METAVISION_HAL_TENCODER_GTEST_COMMON_H
#define METAVISION_HAL_TENCODER_GTEST_COMMON_H

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/utils/device_builder.h"
#include "metavision/hal/decoders/evt2/evt2_decoder.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "evt2_raw_format.h"
#include "tencoder.h"
#include "device_builder_maker.h"

namespace detail {
template<class T, std::size_t N>
std::vector<T> make_vector(const T (&array)[N]) noexcept {
    return std::vector<T>{array, array + N};
}
} // namespace detail

template<typename EvtFormat>
struct event2d_types_def {
    using event2d_TD_class = Metavision::EventCD;
};

// Helper functions to test for event equality
inline void expect_event_equality(const Metavision::Event2d &left, const Metavision::Event2d &right) {
    EXPECT_EQ(left.x, right.x);
    EXPECT_EQ(left.y, right.y);
    EXPECT_EQ(left.p, right.p);
    EXPECT_EQ(left.t, right.t);
}

inline void expect_event_equality(const Metavision::EventExtTrigger &left, const Metavision::EventExtTrigger &right) {
    EXPECT_EQ(left.p, right.p);
    EXPECT_EQ(left.t, right.t);
    EXPECT_EQ(left.id, right.id);
}

template<typename EventType>
void compare_vectors(const std::vector<EventType> &events_expected, const std::vector<EventType> &events) {
    // Now, check that what we decoded back is the same as what we encoded
    ASSERT_EQ(events_expected.size(), events.size());
    for (auto it = events.begin(), it_expected = events_expected.begin(), it_end = events.end(); it != it_end;
         ++it, ++it_expected) {
        expect_event_equality(*it_expected, *it);
    }
}

template<typename RawBaseType>
size_t count_how_many_time_high(const RawBaseType &ev_to_find, const std::vector<uint8_t> &encoded) {
    size_t tot_size_encoded = encoded.size();
    size_t count            = 0;
    if (tot_size_encoded % sizeof(RawBaseType) != 0) {
        std::cerr << "ERROR : size mismatch. " << tot_size_encoded << " vs " << sizeof(RawBaseType) << std::endl;
    }

    const RawBaseType *ptr_begin(reinterpret_cast<const RawBaseType *>(encoded.data()));
    const RawBaseType *ptr_end(reinterpret_cast<const RawBaseType *>(encoded.data() + tot_size_encoded));
    for (auto ptr = ptr_begin; ptr != ptr_end; ++ptr) {
        if (ptr->type == ev_to_find.type && ptr->trail == ev_to_find.trail) {
            ++count;
        }
    }
    return count;
}

using namespace Metavision;

template<class EvtFormat>
inline void build_decoder(DeviceBuilder &device_builder) {}

template<class EvtFormat, class Event>
inline std::vector<Event> build_vector_of_events() {
    return std::vector<Event>();
}

template<>
inline void build_decoder<Evt2RawFormat>(DeviceBuilder &device_builder) {
    static constexpr bool TimeShiftingEnabled = false;

    auto cd_event_decoder          = device_builder.add_facility(std::make_unique<I_EventDecoder<EventCD>>());
    auto ext_trigger_event_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventExtTrigger>>());
    device_builder.add_facility(
        std::make_unique<EVT2Decoder>(TimeShiftingEnabled, cd_event_decoder, ext_trigger_event_decoder));
}

template<typename EventType>
void register_decode_event_cb(Device &device, std::vector<EventType> &decoded_events) {
    auto event_decoder = device.get_facility<I_EventDecoder<EventType>>();
    event_decoder->add_event_buffer_callback([&decoded_events](const EventType *begin, const EventType *end) {
        decoded_events.insert(decoded_events.end(), begin, end);
    });
}

template<typename EventType, typename... EventTypes>
void register_decode_event_cb(Device &device, std::vector<EventType> &decoded_events,
                              std::vector<EventTypes> &...decoded_events_other) {
    auto event_decoder = device.get_facility<I_EventDecoder<EventType>>();
    event_decoder->add_event_buffer_callback([&decoded_events](const EventType *begin, const EventType *end) {
        decoded_events.insert(decoded_events.end(), begin, end);
    });

    register_decode_event_cb(device, decoded_events_other...);
}

template<typename EvtFormat, typename EventType, typename... EventTypes>
void setup_decoders_and_decode(std::vector<uint8_t> &encoded_events, std::vector<EventType> &decoded_events,
                               std::vector<EventTypes> &...decoded_events_other) {
    DeviceBuilder device_builder = make_device_builder();
    build_decoder<EvtFormat>(device_builder);
    auto device = device_builder();

    register_decode_event_cb(*device, decoded_events, decoded_events_other...);

    device->get_facility<I_EventsStreamDecoder>()->decode(encoded_events.data(),
                                                          encoded_events.data() + encoded_events.size());
}

template<typename EvtFormat, typename TimerHighRedundancyPolicy, typename EventType>
void encode_events_and_decode_them_back(const std::vector<EventType> &events_to_encode,
                                        std::vector<EventType> &decoded_events, int slices = 1) {
    std::vector<uint8_t> encoded_events;

    // Encode the events
    using EncoderType = TEncoder<EvtFormat, TimerHighRedundancyPolicy>;
    EncoderType encoder;
    encoder.set_encode_event_callback(
        [&encoded_events](const uint8_t *b, const uint8_t *e) { encoded_events.insert(encoded_events.end(), b, e); });

    size_t n_to_decode = events_to_encode.size();
    auto it            = events_to_encode.data();
    auto it_end        = it + n_to_decode;
    size_t step        = n_to_decode / slices;
    if (slices == 1 || step == 0) {
        encoder.encode(it, it_end);
    } else {
        while (it + step <= it_end) {
            encoder.encode(it, it + step);
            it += step;
        }
        encoder.encode(it, it_end);
    }
    encoder.flush();

    // Now, decode back the events
    setup_decoders_and_decode<EvtFormat>(encoded_events, decoded_events);
}

template<typename EvtFormat, typename TimerHighRedundancyPolicy, typename EventType>
void encode_all_events_and_decode_them_back(const std::vector<EventType> &events_to_encode,
                                            std::vector<EventType> &decoded_events) {
    std::vector<uint8_t> encoded_events;

    // Encode the events
    using EncoderType = TEncoder<EvtFormat, TimerHighRedundancyPolicy>;
    EncoderType encoder;
    encoder.set_encode_event_callback(
        [&encoded_events](const uint8_t *b, const uint8_t *e) { encoded_events.insert(encoded_events.end(), b, e); });

    encoder.encode(events_to_encode.cbegin(), events_to_encode.cend());
    encoder.flush();

    // Now, decode back the events
    setup_decoders_and_decode<EvtFormat>(encoded_events, decoded_events);
}

template<>
inline std::vector<Metavision::EventCD> build_vector_of_events<Evt2RawFormat, Metavision::EventCD>() {
    // clang-format off
    std::vector<Metavision::EventCD> events = {
        Metavision::Event2d(331, 261, 0, 1589), Metavision::Event2d(7, 127, 0, 3040), Metavision::Event2d(468, 114, 0, 4416), Metavision::Event2d(560, 3, 0, 5794),
        Metavision::Event2d(222, 253, 1, 7020), Metavision::Event2d(288, 8, 1, 8868), Metavision::Event2d(314, 296, 1, 10480), Metavision::Event2d(16, 383, 0, 11977),
        Metavision::Event2d(502, 87, 0, 13431), Metavision::Event2d(428, 133, 0, 14877), Metavision::Event2d(589, 8, 0, 16168), Metavision::Event2d(117, 261, 0, 17436),
        Metavision::Event2d(298, 301, 0, 18956), Metavision::Event2d(614, 340, 0, 20543), Metavision::Event2d(576, 2, 0, 22044), Metavision::Event2d(355, 345, 0, 23532),
        Metavision::Event2d(622, 341, 0, 24955), Metavision::Event2d(543, 340, 0, 26220), Metavision::Event2d(569, 471, 0, 27678), Metavision::Event2d(549, 477, 0, 29244),
        Metavision::Event2d(533, 341, 0, 30897), Metavision::Event2d(638, 336, 0, 32535), Metavision::Event2d(591, 7, 0, 34007), Metavision::Event2d(624, 173, 0, 35481),
        Metavision::Event2d(335, 433, 1, 36900), Metavision::Event2d(353, 341, 0, 38405), Metavision::Event2d(454, 243, 0, 40063), Metavision::Event2d(579, 237, 0, 41756),
        Metavision::Event2d(607, 381, 1, 43436), Metavision::Event2d(39, 300, 1, 44967), Metavision::Event2d(611, 187, 0, 46435), Metavision::Event2d(554, 282, 0, 47857),
        Metavision::Event2d(542, 355, 1, 49551), Metavision::Event2d(591, 215, 0, 51227), Metavision::Event2d(610, 52, 0, 52812), Metavision::Event2d(152, 433, 0, 54286),
        Metavision::Event2d(193, 340, 0, 55751), Metavision::Event2d(461, 372, 1, 57181), Metavision::Event2d(53, 348, 0, 58627), Metavision::Event2d(199, 340, 0, 60222),
        Metavision::Event2d(421, 448, 0, 61841), Metavision::Event2d(629, 56, 1, 63300), Metavision::Event2d(130, 313, 0, 64713), Metavision::Event2d(579, 33, 0, 66008),
        Metavision::Event2d(574, 333, 0, 67308), Metavision::Event2d(228, 82, 0, 68815), Metavision::Event2d(521, 358, 1, 70322), Metavision::Event2d(577, 333, 0, 71834),
        Metavision::Event2d(550, 357, 0, 73187), Metavision::Event2d(553, 284, 0, 74505), Metavision::Event2d(289, 388, 0, 75754), Metavision::Event2d(628, 334, 0, 76950),
        Metavision::Event2d(288, 8, 1, 78226), Metavision::Event2d(546, 334, 0, 79694), Metavision::Event2d(39, 300, 0, 81040), Metavision::Event2d(293, 299, 0, 82427),
        Metavision::Event2d(548, 351, 1, 83709), Metavision::Event2d(542, 354, 0, 84873), Metavision::Event2d(523, 332, 0, 86026), Metavision::Event2d(382, 412, 0, 87271),
        Metavision::Event2d(466, 359, 0, 88489), Metavision::Event2d(629, 333, 0, 89722), Metavision::Event2d(8, 200, 1, 91044), Metavision::Event2d(540, 257, 0, 92330),
        Metavision::Event2d(374, 404, 0, 93548), Metavision::Event2d(218, 291, 0, 94759), Metavision::Event2d(522, 262, 0, 95818), Metavision::Event2d(458, 408, 1, 96923),
        Metavision::Event2d(587, 224, 0, 98070), Metavision::Event2d(580, 0, 1, 99294), Metavision::Event2d(4, 334, 0, 100526), Metavision::Event2d(562, 242, 0, 101719),
        Metavision::Event2d(302, 95, 1, 102869), Metavision::Event2d(554, 276, 0, 103972), Metavision::Event2d(458, 408, 1, 105079), Metavision::Event2d(527, 332, 0, 106183),
        Metavision::Event2d(101, 243, 0, 107308), Metavision::Event2d(550, 350, 0, 108484), Metavision::Event2d(604, 331, 0, 109715), Metavision::Event2d(562, 325, 0, 110930),
        Metavision::Event2d(536, 332, 0, 112167), Metavision::Event2d(584, 30, 0, 113378), Metavision::Event2d(635, 328, 0, 114433), Metavision::Event2d(538, 257, 0, 115458),
        Metavision::Event2d(617, 50, 0, 116540), Metavision::Event2d(607, 187, 0, 117600), Metavision::Event2d(565, 323, 0, 118708), Metavision::Event2d(620, 30, 0, 119883),
        Metavision::Event2d(587, 222, 0, 121097), Metavision::Event2d(545, 323, 0, 122261), Metavision::Event2d(108, 268, 0, 123368), Metavision::Event2d(479, 7, 0, 124426),
        Metavision::Event2d(637, 327, 0, 125417), Metavision::Event2d(557, 242, 0, 126442), Metavision::Event2d(526, 340, 1, 127455), Metavision::Event2d(579, 325, 0, 128604),
        Metavision::Event2d(291, 296, 0, 129728), Metavision::Event2d(531, 381, 1, 130830), Metavision::Event2d(122, 336, 1, 131928), Metavision::Event2d(355, 334, 0, 132986),
        Metavision::Event2d(537, 259, 0, 134026), Metavision::Event2d(381, 84, 1, 135081), Metavision::Event2d(596, 331, 0, 136042), Metavision::Event2d(611, 181, 0, 137014),
        Metavision::Event2d(569, 0, 0, 137965), Metavision::Event2d(601, 331, 0, 139033), Metavision::Event2d(547, 280, 0, 140146), Metavision::Event2d(365, 71, 0, 141204),
        Metavision::Event2d(131, 319, 0, 142242), Metavision::Event2d(611, 328, 0, 143277), Metavision::Event2d(622, 383, 0, 144244), Metavision::Event2d(605, 40, 0, 145175),
        Metavision::Event2d(457, 182, 0, 146084), Metavision::Event2d(599, 273, 1, 146945), Metavision::Event2d(547, 318, 1, 147927), Metavision::Event2d(597, 330, 0, 148953),
        Metavision::Event2d(343, 340, 0, 149999), Metavision::Event2d(11, 288, 0, 150988), Metavision::Event2d(133, 312, 0, 152014), Metavision::Event2d(616, 178, 0, 153003),
        Metavision::Event2d(606, 326, 0, 154026), Metavision::Event2d(428, 133, 0, 154892), Metavision::Event2d(575, 319, 0, 155769), Metavision::Event2d(633, 323, 0, 156644),
        Metavision::Event2d(535, 342, 0, 157538), Metavision::Event2d(554, 316, 1, 158482), Metavision::Event2d(590, 259, 0, 159388), Metavision::Event2d(533, 261, 0, 160300),
        Metavision::Event2d(20, 293, 0, 161228), Metavision::Event2d(561, 241, 0, 162056), Metavision::Event2d(542, 321, 0, 162961), Metavision::Event2d(325, 2, 0, 163694),
        Metavision::Event2d(294, 317, 0, 164489), Metavision::Event2d(210, 72, 0, 165219), Metavision::Event2d(629, 322, 0, 165935), Metavision::Event2d(583, 25, 0, 166607),
        Metavision::Event2d(461, 372, 1, 167379), Metavision::Event2d(555, 256, 0, 168145), Metavision::Event2d(129, 343, 1, 168919), Metavision::Event2d(527, 266, 0, 169720),
        Metavision::Event2d(127, 344, 1, 170454), Metavision::Event2d(600, 34, 0, 171222), Metavision::Event2d(563, 237, 0, 171968), Metavision::Event2d(591, 26, 0, 172801),
        Metavision::Event2d(139, 311, 0, 173389), Metavision::Event2d(543, 317, 0, 174103), Metavision::Event2d(117, 336, 0, 174798), Metavision::Event2d(116, 347, 1, 175450),
        Metavision::Event2d(122, 318, 0, 176105), Metavision::Event2d(634, 321, 0, 176808), Metavision::Event2d(155, 477, 0, 177401), Metavision::Event2d(361, 43, 0, 178037),
        Metavision::Event2d(549, 268, 0, 178687), Metavision::Event2d(293, 329, 1, 179386), Metavision::Event2d(152, 433, 1, 180110), Metavision::Event2d(548, 242, 0, 180844),
        Metavision::Event2d(0, 479, 0, 181480), Metavision::Event2d(543, 284, 0, 182168), Metavision::Event2d(118, 339, 0, 182831), Metavision::Event2d(600, 199, 0, 183419),
        Metavision::Event2d(214, 375, 0, 184068), Metavision::Event2d(572, 230, 0, 184677), Metavision::Event2d(627, 226, 0, 185275), Metavision::Event2d(615, 176, 0, 185892),
        Metavision::Event2d(573, 208, 0, 186506), Metavision::Event2d(601, 266, 0, 187081), Metavision::Event2d(602, 321, 0, 187684), Metavision::Event2d(617, 221, 0, 188331),
        Metavision::Event2d(581, 318, 0, 188948), Metavision::Event2d(600, 270, 0, 189615), Metavision::Event2d(322, 263, 0, 190303), Metavision::Event2d(116, 342, 0, 190967),
        Metavision::Event2d(593, 207, 0, 191607), Metavision::Event2d(115, 344, 0, 192257), Metavision::Event2d(293, 334, 1, 192885), Metavision::Event2d(574, 312, 0, 193520),
        Metavision::Event2d(123, 347, 0, 194116), Metavision::Event2d(291, 286, 0, 194759), Metavision::Event2d(567, 344, 0, 195336), Metavision::Event2d(68, 377, 0, 195932),
        Metavision::Event2d(578, 222, 0, 196581), Metavision::Event2d(105, 270, 0, 197186), Metavision::Event2d(101, 243, 1, 197853), Metavision::Event2d(608, 37, 0, 198479),
        Metavision::Event2d(490, 467, 1, 199144), Metavision::Event2d(137, 319, 1, 199799), Metavision::Event2d(623, 216, 1, 200470), Metavision::Event2d(556, 236, 0, 201153),
        Metavision::Event2d(590, 263, 0, 201806), Metavision::Event2d(543, 314, 0, 202462), Metavision::Event2d(551, 237, 0, 203090), Metavision::Event2d(322, 257, 0, 203750),
        Metavision::Event2d(297, 328, 0, 204358), Metavision::Event2d(25, 357, 0, 204956), Metavision::Event2d(590, 207, 0, 205570), Metavision::Event2d(122, 329, 0, 206200),
        Metavision::Event2d(622, 316, 0, 206783), Metavision::Event2d(542, 310, 0, 207432), Metavision::Event2d(572, 224, 0, 208066), Metavision::Event2d(317, 340, 1, 208744),
        Metavision::Event2d(295, 325, 0, 209382), Metavision::Event2d(254, 351, 0, 210050), Metavision::Event2d(78, 350, 0, 210707), Metavision::Event2d(92, 377, 1, 211384),
        Metavision::Event2d(559, 270, 1, 212102), Metavision::Event2d(79, 350, 0, 212812), Metavision::Event2d(82, 346, 0, 213477), Metavision::Event2d(267, 441, 0, 214158),
        Metavision::Event2d(127, 303, 0, 214773), Metavision::Event2d(32, 67, 0, 215403), Metavision::Event2d(436, 27, 1, 216036), Metavision::Event2d(609, 315, 0, 216620),
        Metavision::Event2d(287, 293, 1, 217220), Metavision::Event2d(313, 349, 1, 217920), Metavision::Event2d(574, 288, 1, 218535), Metavision::Event2d(292, 312, 0, 219208),
        Metavision::Event2d(200, 308, 0, 219879), Metavision::Event2d(97, 375, 1, 220539), Metavision::Event2d(312, 348, 1, 221172), Metavision::Event2d(13, 268, 1, 221860)
    };
    // clang-format on
    return events;
}

template<>
inline std::vector<Metavision::EventExtTrigger> build_vector_of_events<Evt2RawFormat, Metavision::EventExtTrigger>() {
    // clang-format off
    std::vector<Metavision::EventExtTrigger> events = {
        Metavision::EventExtTrigger(0, 1589, 28), Metavision::EventExtTrigger(0, 3040, 10), Metavision::EventExtTrigger(0, 4416, 6), Metavision::EventExtTrigger(0, 5794, 13), Metavision::EventExtTrigger(1, 7020, 23),
        Metavision::EventExtTrigger(1, 8868, 4), Metavision::EventExtTrigger(1, 10480, 22), Metavision::EventExtTrigger(0, 11977, 6), Metavision::EventExtTrigger(0, 13431, 3), Metavision::EventExtTrigger(0, 14877, 1),
        Metavision::EventExtTrigger(0, 16168, 8), Metavision::EventExtTrigger(0, 17436, 7), Metavision::EventExtTrigger(0, 18956, 5), Metavision::EventExtTrigger(0, 20543, 25), Metavision::EventExtTrigger(0, 22044, 29),
        Metavision::EventExtTrigger(0, 23532, 4), Metavision::EventExtTrigger(0, 24955, 9), Metavision::EventExtTrigger(0, 26220, 21), Metavision::EventExtTrigger(0, 27678, 13), Metavision::EventExtTrigger(0, 29244, 22),
        Metavision::EventExtTrigger(0, 30897, 0), Metavision::EventExtTrigger(0, 32535, 8), Metavision::EventExtTrigger(0, 34007, 27), Metavision::EventExtTrigger(0, 35481, 0), Metavision::EventExtTrigger(1, 36900, 26),
        Metavision::EventExtTrigger(0, 38405, 17), Metavision::EventExtTrigger(0, 40063, 11), Metavision::EventExtTrigger(0, 41756, 10), Metavision::EventExtTrigger(1, 43436, 25), Metavision::EventExtTrigger(1, 44967, 10),
        Metavision::EventExtTrigger(0, 46435, 0), Metavision::EventExtTrigger(0, 47857, 18), Metavision::EventExtTrigger(1, 49551, 21), Metavision::EventExtTrigger(0, 51227, 3), Metavision::EventExtTrigger(0, 52812, 29),
        Metavision::EventExtTrigger(0, 54286, 9), Metavision::EventExtTrigger(0, 55751, 7), Metavision::EventExtTrigger(1, 57181, 19), Metavision::EventExtTrigger(0, 58627, 13), Metavision::EventExtTrigger(0, 60222, 10),
        Metavision::EventExtTrigger(0, 61841, 18), Metavision::EventExtTrigger(1, 63300, 21), Metavision::EventExtTrigger(0, 64713, 15), Metavision::EventExtTrigger(0, 66008, 23), Metavision::EventExtTrigger(0, 67308, 11),
        Metavision::EventExtTrigger(0, 68815, 10), Metavision::EventExtTrigger(1, 70322, 26), Metavision::EventExtTrigger(0, 71834, 20), Metavision::EventExtTrigger(0, 73187, 29), Metavision::EventExtTrigger(0, 74505, 6),
        Metavision::EventExtTrigger(0, 82427, 6), Metavision::EventExtTrigger(1, 83709, 17), Metavision::EventExtTrigger(0, 84873, 5), Metavision::EventExtTrigger(0, 86026, 16), Metavision::EventExtTrigger(0, 87271, 7),
        Metavision::EventExtTrigger(0, 106183, 21), Metavision::EventExtTrigger(0, 107308, 23), Metavision::EventExtTrigger(0, 108484, 2), Metavision::EventExtTrigger(0, 109715, 6), Metavision::EventExtTrigger(0, 110930, 20),
        Metavision::EventExtTrigger(0, 134026, 30), Metavision::EventExtTrigger(1, 135081, 25), Metavision::EventExtTrigger(0, 136042, 8), Metavision::EventExtTrigger(0, 137014, 7), Metavision::EventExtTrigger(0, 137965, 5),
        Metavision::EventExtTrigger(0, 139033, 18), Metavision::EventExtTrigger(0, 140146, 26), Metavision::EventExtTrigger(0, 141204, 29), Metavision::EventExtTrigger(0, 142242, 18), Metavision::EventExtTrigger(0, 143277, 30),
        Metavision::EventExtTrigger(0, 144244, 16), Metavision::EventExtTrigger(0, 145175, 23), Metavision::EventExtTrigger(0, 146084, 10), Metavision::EventExtTrigger(1, 146945, 29), Metavision::EventExtTrigger(1, 147927, 8),
        Metavision::EventExtTrigger(0, 148953, 22), Metavision::EventExtTrigger(0, 149999, 6), Metavision::EventExtTrigger(0, 150988, 0), Metavision::EventExtTrigger(0, 152014, 19), Metavision::EventExtTrigger(0, 153003, 20),
        Metavision::EventExtTrigger(1, 158482, 14), Metavision::EventExtTrigger(0, 159388, 27), Metavision::EventExtTrigger(0, 160300, 2), Metavision::EventExtTrigger(0, 161228, 21), Metavision::EventExtTrigger(0, 162056, 19),
        Metavision::EventExtTrigger(0, 162961, 21), Metavision::EventExtTrigger(0, 163694, 16), Metavision::EventExtTrigger(0, 164489, 11), Metavision::EventExtTrigger(0, 165219, 27), Metavision::EventExtTrigger(0, 165935, 21),
        Metavision::EventExtTrigger(0, 166607, 17), Metavision::EventExtTrigger(1, 167379, 10), Metavision::EventExtTrigger(0, 168145, 14), Metavision::EventExtTrigger(1, 168919, 11), Metavision::EventExtTrigger(0, 169720, 29)
    };
    // clang-format on
    return events;
}

#endif // METAVISION_HAL_TENCODER_GTEST_COMMON_H
