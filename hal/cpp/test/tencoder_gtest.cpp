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

#include "metavision/sdk/base/events/event2d.h"

#include "tencoder.h"
#include "encoding_policies.h"
#include "tencoder_gtest_common.h"
#include "tencoder_gtest_instantiation.h"

using namespace Metavision;

template<typename Format, typename TimerHighRedundancyPolicyType, timestamp T_STEP>
struct GtestsParameters {
    using EvtFormat = Format;

    using TimerHighRedundancyPolicy = TimerHighRedundancyPolicyType;

    static constexpr timestamp TIME_STEP = T_STEP;
};

template<class Parameters>
class TEncoder_Gtest : public ::testing::Test {
    using EvtFormat = typename Parameters::EvtFormat;

    using TimerHighRedundancyPolicy = typename Parameters::TimerHighRedundancyPolicy;

    using EncoderType = TEncoder<EvtFormat, TimerHighRedundancyPolicy>;

public:
    TEncoder_Gtest() : encoder_(Parameters::TIME_STEP) {}

    virtual ~TEncoder_Gtest() {}

    template<typename EventType>
    void encode_and_decode(const std::vector<EventType> &events_to_encode, std::vector<EventType> &decoded_events) {
        encode_all_events_and_decode_them_back<EvtFormat, TimerHighRedundancyPolicy>(events_to_encode, decoded_events);
    }

    template<typename EventType, typename EventType2>
    void encode_and_decode(const std::vector<EventType> &events_to_encode, std::vector<EventType> &decoded_events,
                           const std::vector<EventType2> &events_to_encode2, std::vector<EventType2> &decoded_events2) {
        // Encode the events
        encoder_.encode(events_to_encode.cbegin(), events_to_encode.cend(), events_to_encode2.cbegin(),
                        events_to_encode2.cend());
        encoder_.flush();

        // Now, decode back the events
        setup_decoders_and_decode<EvtFormat>(encoded_buffer_, decoded_events, decoded_events2);
    }

    template<typename EventType, typename EventType2, typename EventType3>
    void encode_and_decode(const std::vector<EventType> &events_to_encode, std::vector<EventType> &decoded_events,
                           const std::vector<EventType2> &events_to_encode2, std::vector<EventType2> &decoded_events2,
                           const std::vector<EventType3> &events_to_encode3, std::vector<EventType3> &decoded_events3) {
        // Encode the events
        encoder_.encode(events_to_encode.cbegin(), events_to_encode.cend(), events_to_encode2.cbegin(),
                        events_to_encode2.cend(), events_to_encode3.cbegin(), events_to_encode3.cend());
        encoder_.flush();

        // Now, decode back the events
        setup_decoders_and_decode<EvtFormat>(encoded_buffer_, decoded_events, decoded_events2, decoded_events3);
    }

    template<typename EventType, typename EventType2, typename EventType3, typename EventType4>
    void encode_and_decode(const std::vector<EventType> &events_to_encode, std::vector<EventType> &decoded_events,
                           const std::vector<EventType2> &events_to_encode2, std::vector<EventType2> &decoded_events2,
                           const std::vector<EventType3> &events_to_encode3, std::vector<EventType3> &decoded_events3,
                           const std::vector<EventType4> &events_to_encode4, std::vector<EventType4> &decoded_events4) {
        // Encode the events
        encoder_.encode(events_to_encode.cbegin(), events_to_encode.cend(), events_to_encode2.cbegin(),
                        events_to_encode2.cend(), events_to_encode3.cbegin(), events_to_encode3.cend(),
                        events_to_encode4.cbegin(), events_to_encode4.cend());
        encoder_.flush();

        // Now, decode back the events
        setup_decoders_and_decode<EvtFormat>(encoded_buffer_, decoded_events, decoded_events2, decoded_events3,
                                             decoded_events4);
    }

protected:
    virtual void SetUp() override {
        encoder_.set_encode_event_callback(
            [this](const uint8_t *b, const uint8_t *e) { encoded_buffer_.insert(encoded_buffer_.end(), b, e); });
    }

    virtual void TearDown() override {}

    EncoderType encoder_;
    std::vector<uint8_t> encoded_buffer_;
};

TYPED_TEST_CASE(TEncoder_Gtest, TestingTypes);

TYPED_TEST(TEncoder_Gtest, test_timer_high_redundancy_timer_high_value) {
    using EvtFormat                 = typename TypeParam::EvtFormat;
    using TimerHighRedundancyPolicy = typename TypeParam::TimerHighRedundancyPolicy;
    using CameraEvent2dTD           = typename event2d_types_def<EvtFormat>::event2d_TD_class;

    static constexpr timestamp TH  = (1 << event_raw_format_traits<EvtFormat>::NLowerBitsTH) - 1;
    static constexpr timestamp TH2 = 2 * (1 << event_raw_format_traits<EvtFormat>::NLowerBitsTH);

    std::vector<CameraEvent2dTD> tds = {CameraEvent2dTD(288, 64, 1, TH), CameraEvent2dTD(288, 64, 1, TH2)};

    // Encode
    this->encoder_.encode(tds.begin(), tds.end());
    this->encoder_.flush();

    // Count number of Timer High events

    // Get the expected value :

    // Timer-high raw event
    TimerHighEncoder<EvtFormat, TimerHighRedundancyPolicy> th_encoder;
    ASSERT_EQ(sizeof(typename event_raw_format_traits<EvtFormat>::BaseEventType),
              th_encoder.template get_size_encoded<>());
    typename event_raw_format_traits<EvtFormat>::BaseEventType ev_th;
    th_encoder.initialize(TH);
    th_encoder.template encode_next_event<>(reinterpret_cast<uint8_t *>(&ev_th));

    // First is always 1 (because when we start encoding we just encode the latest time high):
    EXPECT_EQ(1, count_how_many_time_high(ev_th, this->encoded_buffer_));

    // The second must be TypeParam::TimerHighRedundancyPolicy::REDUNDANCY_FACTOR
    th_encoder.template encode_next_event<>(reinterpret_cast<uint8_t *>(&ev_th));
    EXPECT_EQ(TypeParam::TimerHighRedundancyPolicy::REDUNDANCY_FACTOR,
              count_how_many_time_high(ev_th, this->encoded_buffer_));
}

TYPED_TEST(TEncoder_Gtest, test_time_overflow) {
    using EvtFormat       = typename TypeParam::EvtFormat;
    using CameraEvent2dTD = typename event2d_types_def<EvtFormat>::event2d_TD_class;

    static constexpr timestamp MAX_TH = timestamp((1 << 28) - 1) << event_raw_format_traits<EvtFormat>::NLowerBitsTH;

    std::vector<CameraEvent2dTD> tds = {
        CameraEvent2dTD(288, 64, 1, MAX_TH - 5000),  CameraEvent2dTD(228, 166, 0, MAX_TH - 4000),
        CameraEvent2dTD(162, 166, 1, MAX_TH - 3000), CameraEvent2dTD(186, 166, 1, MAX_TH - 2000),
        CameraEvent2dTD(288, 64, 1, MAX_TH - 1000),  CameraEvent2dTD(228, 166, 0, MAX_TH - 900),
        CameraEvent2dTD(162, 166, 1, MAX_TH - 800),  CameraEvent2dTD(186, 166, 1, MAX_TH - 700),
        CameraEvent2dTD(17, 236, 0, MAX_TH - 600),   CameraEvent2dTD(85, 159, 1, MAX_TH - 500),
        CameraEvent2dTD(274, 154, 0, MAX_TH - 400),  CameraEvent2dTD(233, 154, 1, MAX_TH - 300),
        CameraEvent2dTD(0, 154, 1, MAX_TH - 200),    CameraEvent2dTD(184, 175, 0, MAX_TH - 100),
        CameraEvent2dTD(64, 175, 0, MAX_TH - 10),    CameraEvent2dTD(155, 161, 1, MAX_TH - 5),
        CameraEvent2dTD(16, 161, 1, MAX_TH),         CameraEvent2dTD(169, 223, 0, MAX_TH + 5),
        CameraEvent2dTD(162, 223, 1, MAX_TH + 10),   CameraEvent2dTD(206, 173, 0, MAX_TH + 100),
        CameraEvent2dTD(175, 173, 1, MAX_TH + 200),  CameraEvent2dTD(177, 173, 0, MAX_TH + 300),
        CameraEvent2dTD(153, 173, 1, MAX_TH + 400),  CameraEvent2dTD(71, 171, 1, MAX_TH + 500),
        CameraEvent2dTD(233, 163, 1, MAX_TH + 600),  CameraEvent2dTD(137, 158, 0, MAX_TH + 700),
        CameraEvent2dTD(36, 135, 0, MAX_TH + 800),   CameraEvent2dTD(197, 112, 0, MAX_TH + 900),
        CameraEvent2dTD(26, 188, 1, MAX_TH + 1000),  CameraEvent2dTD(25, 169, 1, MAX_TH + 2000),
        CameraEvent2dTD(113, 220, 1, MAX_TH + 3000), CameraEvent2dTD(147, 193, 1, MAX_TH + 4000),
        CameraEvent2dTD(224, 138, 0, MAX_TH + 5000)};

    // Encode and decode back
    std::vector<CameraEvent2dTD> tds_decoded;
    this->encode_and_decode(tds, tds_decoded);

    // Now check the contents of vectors of decoded events

    // TDs :
    compare_vectors(tds, tds_decoded);
}

TYPED_TEST(TEncoder_Gtest, td_and_ext_trigger) {
    using EvtFormat       = typename TypeParam::EvtFormat;
    using CameraEvent2dTD = typename event2d_types_def<EvtFormat>::event2d_TD_class;

    std::vector<CameraEvent2dTD> tds    = build_vector_of_events<EvtFormat, CameraEvent2dTD>();
    std::vector<EventExtTrigger> triggs = build_vector_of_events<EvtFormat, EventExtTrigger>();

    // Encode and decode back
    std::vector<CameraEvent2dTD> tds_decoded;
    std::vector<EventExtTrigger> triggs_decoded;
    this->encode_and_decode(tds, tds_decoded, triggs, triggs_decoded);

    // Now check the contents of vectors of decoded events

    // TDs :
    compare_vectors(tds, tds_decoded);

    // Triggers :
    compare_vectors(triggs, triggs_decoded);
}

TYPED_TEST(TEncoder_Gtest, td_empty_encoding) {
    using EvtFormat       = typename TypeParam::EvtFormat;
    using CameraEvent2dTD = typename event2d_types_def<EvtFormat>::event2d_TD_class;

    std::vector<CameraEvent2dTD> tds;

    // Encode and decode back
    std::vector<CameraEvent2dTD> tds_decoded;
    this->encode_and_decode(tds, tds_decoded);

    // Now check the contents of vectors of decoded events

    // TDs :
    compare_vectors(tds, tds_decoded);
}
