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

#include <iostream>
#include <fstream>
#include <memory>
#include <numeric>
#include <boost/filesystem.hpp>

#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/utils/gtest/gtest_custom.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/utils/get_time.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/facilities/i_decoder.h"
#include "devices/utils/device_system_id.h"
#include "boards/rawfile/psee_raw_file_header.h"
#include "tencoder_gtest_common.h"

using namespace Metavision;
class PseeDecoder_Gtest : public GTestWithTmpDir {
protected:
    virtual void SetUp() {
        tmp_file_ = tmpdir_handler_->get_full_path("PseeDecoder_Gtest.raw");
    }

    virtual void TearDown() {
        close_file();
    }

    void write_header(const PseeRawFileHeader &header) {
        (*log_raw_data_) << header;
    }

    PseeRawFileHeader get_default_header() {
        auto header = std::stringstream();
        header << "\% Date 2014-02-28 13:37:42" << std::endl
               << "\% system_ID " << dummy_system_id << std::endl
               << "\% integrator_name " << get_psee_plugin_integrator_name() << std::endl
               << "\% plugin_name hal_plugin_gen31_fx3" << std::endl
               << "\% firmware_version 0.0.0" << std::endl
               << "\% evt 2.0" << std::endl
               << "\% subsystem_ID " << dummy_sub_system_id_ << std::endl
               << "\% " << dummy_custom_key_ << " " << dummy_custom_value_ << std::endl
               << "\% serial_number " << dummy_serial_ << std::endl;
        return PseeRawFileHeader(header);
    }

    void open_file() {
        log_raw_data_.reset(new std::ofstream(tmp_file_, std::ios::binary));
    }

    void close_file() {
        if (log_raw_data_) {
            log_raw_data_->close();
        }
    }

    std::pair<std::vector<EventCD>, std::vector<Metavision::EventExtTrigger>> write_evt2_raw_data_with_trigger() {
        std::pair<std::vector<EventCD>, std::vector<Metavision::EventExtTrigger>> data;
        open_file();

        write_header(get_default_header());

        data.first  = build_vector_of_events<Evt2RawFormat, EventCD>();
        data.second = build_vector_of_events<Evt2RawFormat, EventExtTrigger>();
        TEncoder<Evt2RawFormat, TimerHighRedundancyEvt2Default> encoder;
        encoder.set_encode_event_callback([&](const uint8_t *data, const uint8_t *data_end) {
            log_raw_data_->write(reinterpret_cast<const char *>(data), std::distance(data, data_end));
            bytes_written_ += std::distance(data, data_end);
        });

        encoder.encode(data.first.cbegin(), data.first.cend(), data.second.cbegin(), data.second.cend());
        encoder.flush();
        close_file();

        return data;
    }

    std::string tmp_file_;
    std::unique_ptr<std::ofstream> log_raw_data_;
    size_t bytes_written_{0};

    static const std::string dummy_serial_;
    static const std::string dummy_plugin_name_;
    static const std::string integrator_name_;
    static const std::string dummy_events_type_;
    static const std::string dummy_custom_key_;
    static const std::string dummy_custom_value_;

    static constexpr SystemId dummy_system_id   = SystemId::SYSTEM_CCAM3_GEN31;
    static constexpr long dummy_system_version_ = 0;
    static constexpr long dummy_sub_system_id_  = 0;
};

constexpr long PseeDecoder_Gtest::dummy_system_version_;
constexpr SystemId PseeDecoder_Gtest::dummy_system_id;
constexpr long PseeDecoder_Gtest::dummy_sub_system_id_;
const std::string PseeDecoder_Gtest::dummy_serial_       = "dummy_serial";
const std::string PseeDecoder_Gtest::dummy_plugin_name_  = "plugin_name";
const std::string PseeDecoder_Gtest::integrator_name_    = get_psee_plugin_integrator_name();
const std::string PseeDecoder_Gtest::dummy_events_type_  = "events_type";
const std::string PseeDecoder_Gtest::dummy_custom_key_   = "custom";
const std::string PseeDecoder_Gtest::dummy_custom_value_ = "field";

using SizeTypeFirst  = std::vector<EventCD>::size_type;
using SizeTypeSecond = std::vector<Metavision::EventExtTrigger>::size_type;

TEST_F(PseeDecoder_Gtest, decode_evt2_data_nominal) {
    // GIVEN a RAW file in EVT2 format with a known content
    const auto expected_events = write_evt2_raw_data_with_trigger();
    std::vector<EventCD> received_cd_events;
    std::vector<EventExtTrigger> received_triggers_events;

    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(tmp_file_, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder         = device->get_facility<I_Decoder>();
    auto cd_decoder      = device->get_facility<I_EventDecoder<EventCD>>();
    auto trigger_decoder = device->get_facility<I_EventDecoder<EventExtTrigger>>();
    auto es              = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, trigger_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_events.insert(received_cd_events.end(), ev_begin, ev_end); });
    trigger_decoder->add_event_buffer_callback([&](auto trigger, auto trigger_end) {
        received_triggers_events.insert(received_triggers_events.end(), trigger, trigger_end);
    });

    es->start();

    // WHEN we stream and decode the events in the file
    long int bytes_polled_count;
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data(bytes_polled_count);
        decoder->decode(raw_buffer, raw_buffer + bytes_polled_count);
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(expected_events.first.size(), received_cd_events.size());
    for (SizeTypeFirst i = 0, i_end = expected_events.first.size(); i < i_end; ++i) {
        ASSERT_EQ(expected_events.first[i].x, received_cd_events[i].x);
        ASSERT_EQ(expected_events.first[i].y, received_cd_events[i].y);
        ASSERT_EQ(expected_events.first[i].p, received_cd_events[i].p);
        ASSERT_EQ(expected_events.first[i].t, received_cd_events[i].t);
    }

    ASSERT_EQ(expected_events.second.size(), received_triggers_events.size());
    for (SizeTypeSecond i = 0, i_end = expected_events.second.size(); i < i_end; ++i) {
        ASSERT_EQ(expected_events.second[i].p, received_triggers_events[i].p);
        ASSERT_EQ(expected_events.second[i].t, received_triggers_events[i].t);
        ASSERT_EQ(expected_events.second[i].id, received_triggers_events[i].id);
    }
}

TEST_F(PseeDecoder_Gtest, decode_evt2_data_random_split_in_buffer) {
    // GIVEN a RAW file in EVT2 format with a known content
    const auto expected_events = write_evt2_raw_data_with_trigger();
    std::vector<EventCD> received_cd_events;
    std::vector<EventExtTrigger> received_triggers_events;

    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(tmp_file_, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder         = device->get_facility<I_Decoder>();
    auto cd_decoder      = device->get_facility<I_EventDecoder<EventCD>>();
    auto trigger_decoder = device->get_facility<I_EventDecoder<EventExtTrigger>>();
    auto es              = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, trigger_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_events.insert(received_cd_events.end(), ev_begin, ev_end); });
    trigger_decoder->add_event_buffer_callback([&](auto trigger, auto trigger_end) {
        received_triggers_events.insert(received_triggers_events.end(), trigger, trigger_end);
    });

    es->start();

    // WHEN we stream and decode events in buffer of random size
    long int bytes_polled_count;
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data(bytes_polled_count);
        auto raw_buffer_end           = raw_buffer + bytes_polled_count;
        auto raw_data_to_decode_count = 2 * (bytes_polled_count / 10) +
                                        1; // Ensures odd number of bytes so that we have split in middle of raw event
        auto raw_buffer_decode_to = raw_buffer + raw_data_to_decode_count;

        for (; raw_buffer < raw_buffer_end;) {
            decoder->decode(raw_buffer, raw_buffer_decode_to);
            raw_buffer           = raw_buffer_decode_to;
            raw_buffer_decode_to = std::min(raw_buffer + raw_data_to_decode_count, raw_buffer_end);
        }
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(expected_events.first.size(), received_cd_events.size());
    for (SizeTypeFirst i = 0, i_end = expected_events.first.size(); i < i_end; ++i) {
        ASSERT_EQ(expected_events.first[i].x, received_cd_events[i].x);
        ASSERT_EQ(expected_events.first[i].y, received_cd_events[i].y);
        ASSERT_EQ(expected_events.first[i].p, received_cd_events[i].p);
        ASSERT_EQ(expected_events.first[i].t, received_cd_events[i].t);
    }

    ASSERT_EQ(expected_events.second.size(), received_triggers_events.size());
    for (SizeTypeSecond i = 0, i_end = expected_events.second.size(); i < i_end; ++i) {
        ASSERT_EQ(expected_events.second[i].p, received_triggers_events[i].p);
        ASSERT_EQ(expected_events.second[i].t, received_triggers_events[i].t);
        ASSERT_EQ(expected_events.second[i].id, received_triggers_events[i].id);
    }
}

TEST_F(PseeDecoder_Gtest, decode_evt2_data_byte_by_byte) {
    // GIVEN a RAW file in EVT2 format with a known content
    const auto expected_events = write_evt2_raw_data_with_trigger();
    std::vector<EventCD> received_cd_events;
    std::vector<EventExtTrigger> received_triggers_events;

    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(tmp_file_, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder         = device->get_facility<I_Decoder>();
    auto cd_decoder      = device->get_facility<I_EventDecoder<EventCD>>();
    auto trigger_decoder = device->get_facility<I_EventDecoder<EventExtTrigger>>();
    auto es              = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, trigger_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_events.insert(received_cd_events.end(), ev_begin, ev_end); });
    trigger_decoder->add_event_buffer_callback([&](auto trigger, auto trigger_end) {
        received_triggers_events.insert(received_triggers_events.end(), trigger, trigger_end);
    });

    es->start();

    // WHEN we stream and decode data byte by byte data
    long int bytes_polled_count;
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data(bytes_polled_count);
        auto raw_buffer_end           = raw_buffer + bytes_polled_count;
        auto raw_data_to_decode_count = 1;
        auto raw_buffer_decode_to     = raw_buffer + raw_data_to_decode_count;

        for (; raw_buffer < raw_buffer_end;) {
            decoder->decode(raw_buffer, raw_buffer_decode_to);
            raw_buffer           = raw_buffer_decode_to;
            raw_buffer_decode_to = std::min(raw_buffer + raw_data_to_decode_count, raw_buffer_end);
        }
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(expected_events.first.size(), received_cd_events.size());
    for (SizeTypeFirst i = 0, i_end = expected_events.first.size(); i < i_end; ++i) {
        ASSERT_EQ(expected_events.first[i].x, received_cd_events[i].x);
        ASSERT_EQ(expected_events.first[i].y, received_cd_events[i].y);
        ASSERT_EQ(expected_events.first[i].p, received_cd_events[i].p);
        ASSERT_EQ(expected_events.first[i].t, received_cd_events[i].t);
    }

    ASSERT_EQ(expected_events.second.size(), received_triggers_events.size());
    for (SizeTypeSecond i = 0, i_end = expected_events.second.size(); i < i_end; ++i) {
        ASSERT_EQ(expected_events.second[i].p, received_triggers_events[i].p);
        ASSERT_EQ(expected_events.second[i].t, received_triggers_events[i].t);
        ASSERT_EQ(expected_events.second[i].id, received_triggers_events[i].id);
    }
}

TEST_F_WITH_DATASET(PseeDecoder_Gtest, decode_evt3_data_nominal) {
    // GIVEN a RAW file in EVT3 format with a known content
    std::string dataset_file_path =
        (boost::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt3_hand.raw").string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_Decoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode the events in the file
    long int bytes_polled_count;
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data(bytes_polled_count);
        decoder->decode(raw_buffer, raw_buffer + bytes_polled_count);
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(18094969, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeDecoder_Gtest, decode_evt3_data_random_split) {
    // GIVEN a RAW file in EVT3 format with a known content
    std::string dataset_file_path =
        (boost::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt3_hand.raw").string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_Decoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode events in buffer of random size
    long int bytes_polled_count;
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data(bytes_polled_count);
        auto raw_buffer_end           = raw_buffer + bytes_polled_count;
        auto raw_data_to_decode_count = 2 * (bytes_polled_count / 10) +
                                        1; // Ensures odd number of bytes so that we have split in middle of raw event
        auto raw_buffer_decode_to = raw_buffer + raw_data_to_decode_count;

        for (; raw_buffer < raw_buffer_end;) {
            decoder->decode(raw_buffer, raw_buffer_decode_to);
            raw_buffer           = raw_buffer_decode_to;
            raw_buffer_decode_to = std::min(raw_buffer + raw_data_to_decode_count, raw_buffer_end);
        }
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(18094969, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeDecoder_Gtest, decode_evt3_data_byte_by_byte) {
    // GIVEN a RAW file in EVT3 format with a known content
    std::string dataset_file_path =
        (boost::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt3_hand.raw").string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_Decoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode data byte by byte data
    long int bytes_polled_count;
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data(bytes_polled_count);
        auto raw_buffer_end           = raw_buffer + bytes_polled_count;
        auto raw_data_to_decode_count = 1;
        auto raw_buffer_decode_to     = raw_buffer + raw_data_to_decode_count;

        for (; raw_buffer < raw_buffer_end;) {
            decoder->decode(raw_buffer, raw_buffer_decode_to);
            raw_buffer           = raw_buffer_decode_to;
            raw_buffer_decode_to = std::min(raw_buffer + raw_data_to_decode_count, raw_buffer_end);
        }
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(18094969, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeDecoder_Gtest, decode_evt3_nevents_monotonous_timestamps) {
    // GIVEN a RAW file in EVT3 format with a known content
    std::string dataset_file_path =
        (boost::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt3_hand.raw").string();

    // AND a device configured to read by batches of n events
    RawFileConfig cfg;
    cfg.n_events_to_read_ = 10000;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder = device->get_facility<I_Decoder>();
    auto es      = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, es);

    es->start();

    // WHEN we stream and decode data byte by byte
    Metavision::timestamp previous_ts_last = 0;
    long int bytes_polled_count;
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data(bytes_polled_count);
        auto raw_buffer_end           = raw_buffer + bytes_polled_count;
        auto raw_data_to_decode_count = 1;
        auto raw_buffer_decode_to     = raw_buffer + raw_data_to_decode_count;

        for (; raw_buffer < raw_buffer_end;) {
            decoder->decode(raw_buffer, raw_buffer_decode_to);

            // THEN the timestamps increase monotonously
            Metavision::timestamp current_ts_last = decoder->get_last_timestamp();
            if (current_ts_last > 0) {
                ASSERT_GE(current_ts_last, previous_ts_last);
                previous_ts_last = current_ts_last;
            }

            raw_buffer           = raw_buffer_decode_to;
            raw_buffer_decode_to = std::min(raw_buffer + raw_data_to_decode_count, raw_buffer_end);
        }
    }
}
