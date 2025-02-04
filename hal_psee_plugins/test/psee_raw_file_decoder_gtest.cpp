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

#include <filesystem>
#include <iostream>
#include <fstream>
#include <memory>
#include <numeric>

#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/utils/gtest/gtest_custom.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_monitoring.h"
#include "metavision/sdk/base/events/event_pointcloud.h"
#include "metavision/sdk/base/events/raw_event_frame_diff.h"
#include "metavision/sdk/base/events/raw_event_frame_histo.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/i_event_frame_decoder.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "devices/utils/device_system_id.h"
#include "metavision/psee_hw_layer/boards/rawfile/psee_raw_file_header.h"
#include "plugin/psee_plugin.h"
#include "tencoder_gtest_common.h"

using namespace Metavision;
class PseeRawFileDecoder_Gtest : public GTestWithTmpDir {
protected:
    virtual void SetUp() {
        tmp_file_ = tmpdir_handler_->get_full_path("PseeRawFileDecoder_Gtest.raw");
    }

    virtual void TearDown() {
        close_file();
    }

    void write_header(const PseeRawFileHeader &header) {
        (*log_raw_data_) << header;
    }

    PseeRawFileHeader get_default_header() {
        auto header = std::stringstream();
        header << "% Date 2014-02-28 13:37:42" << std::endl
               << "% system_ID " << dummy_system_id << std::endl
               << "% integrator_name " << get_psee_plugin_integrator_name() << std::endl
               << "% plugin_name hal_plugin_gen31_fx3" << std::endl
               << "% firmware_version 0.0.0" << std::endl
               << "% evt 2.0" << std::endl
               << "% subsystem_ID " << dummy_sub_system_id_ << std::endl
               << "% " << dummy_custom_key_ << " " << dummy_custom_value_ << std::endl
               << "% serial_number " << dummy_serial_ << std::endl;
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

constexpr long PseeRawFileDecoder_Gtest::dummy_system_version_;
constexpr SystemId PseeRawFileDecoder_Gtest::dummy_system_id;
constexpr long PseeRawFileDecoder_Gtest::dummy_sub_system_id_;
const std::string PseeRawFileDecoder_Gtest::dummy_serial_       = "dummy_serial";
const std::string PseeRawFileDecoder_Gtest::dummy_plugin_name_  = "plugin_name";
const std::string PseeRawFileDecoder_Gtest::integrator_name_    = get_psee_plugin_integrator_name();
const std::string PseeRawFileDecoder_Gtest::dummy_events_type_  = "events_type";
const std::string PseeRawFileDecoder_Gtest::dummy_custom_key_   = "custom";
const std::string PseeRawFileDecoder_Gtest::dummy_custom_value_ = "field";

using SizeTypeFirst  = std::vector<EventCD>::size_type;
using SizeTypeSecond = std::vector<Metavision::EventExtTrigger>::size_type;

TEST_F(PseeRawFileDecoder_Gtest, decode_evt2_data_nominal) {
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
    auto decoder         = device->get_facility<I_EventsStreamDecoder>();
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
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        decoder->decode(raw_buffer);
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

TEST_F(PseeRawFileDecoder_Gtest, decode_evt2_data_random_split_in_buffer) {
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
    auto decoder         = device->get_facility<I_EventsStreamDecoder>();
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
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 2 * (raw_buffer.size() / 10) +
                                        1; // Ensures odd number of bytes so that we have split in middle of raw event
        auto raw_buffer_decode_to = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);
            cur_raw_ptr = raw_buffer_decode_to;
            raw_buffer_decode_to =
                std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.data() + raw_buffer.size());
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

TEST_F(PseeRawFileDecoder_Gtest, decode_evt2_data_byte_by_byte) {
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
    auto decoder         = device->get_facility<I_EventsStreamDecoder>();
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
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 1;
        auto raw_buffer_decode_to     = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);
            cur_raw_ptr = raw_buffer_decode_to;
            raw_buffer_decode_to =
                std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.data() + raw_buffer.size());
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

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt21_data_nominal) {
    // GIVEN a RAW file in EVT21 format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "claque_doigt_evt21.raw")
            .string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode the events in the file
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        decoder->decode(raw_buffer);
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(16294351, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt21_data_random_split) {
    // GIVEN a RAW file in EVT21 format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "claque_doigt_evt21.raw")
            .string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode events in buffer of random size
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 2 * (raw_buffer.size() / 10) +
                                        1; // Ensures odd number of bytes so that we have split in middle of raw event
        auto raw_buffer_decode_to = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);
            cur_raw_ptr = raw_buffer_decode_to;
            raw_buffer_decode_to =
                std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.data() + raw_buffer.size());
        }
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(16294351, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt21_data_byte_by_byte) {
    // GIVEN a RAW file in EVT21 format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "claque_doigt_evt21.raw")
            .string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode data byte by byte data
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 1;
        auto raw_buffer_decode_to     = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);
            cur_raw_ptr = raw_buffer_decode_to;
            raw_buffer_decode_to =
                std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.data() + raw_buffer.size());
        }
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(16294351, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt21_nevents_monotonous_timestamps) {
    // GIVEN a RAW file in EVT21 format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "claque_doigt_evt21.raw")
            .string();

    // AND a device configured to read by batches of n events
    RawFileConfig cfg;
    cfg.n_events_to_read_ = 10000;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder = device->get_facility<I_EventsStreamDecoder>();
    auto es      = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, es);

    es->start();

    // WHEN we stream and decode data byte by byte
    Metavision::timestamp previous_ts_last = 0;
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 1;
        auto raw_buffer_decode_to     = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);

            // THEN the timestamps increase monotonously
            Metavision::timestamp current_ts_last = decoder->get_last_timestamp();
            if (current_ts_last > 0) {
                ASSERT_GE(current_ts_last, previous_ts_last);
                previous_ts_last = current_ts_last;
            }

            cur_raw_ptr = raw_buffer_decode_to;
            raw_buffer_decode_to =
                std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.data() + raw_buffer.size());
        }
    }
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt21_legacy_data_nominal) {
    // GIVEN a RAW file in EVT21 format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "standup_evt21-legacy.raw")
            .string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode the events in the file
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        decoder->decode(raw_buffer);
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(9419216, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt21_legacy_data_random_split) {
    // GIVEN a RAW file in EVT21 format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "standup_evt21-legacy.raw")
            .string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode events in buffer of random size
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 2 * (raw_buffer.size() / 10) +
                                        1; // Ensures odd number of bytes so that we have split in middle of raw event
        auto raw_buffer_decode_to = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);
            cur_raw_ptr = raw_buffer_decode_to;
            raw_buffer_decode_to =
                std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.data() + raw_buffer.size());
        }
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(9419216, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt21_legacy_data_byte_by_byte) {
    // GIVEN a RAW file in EVT21 format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "standup_evt21-legacy.raw")
            .string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode data byte by byte data
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 1;
        auto raw_buffer_decode_to     = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);
            cur_raw_ptr = raw_buffer_decode_to;
            raw_buffer_decode_to =
                std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.data() + raw_buffer.size());
        }
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(9419216, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt21_legacy_nevents_monotonous_timestamps) {
    // GIVEN a RAW file in EVT21 format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "standup_evt21-legacy.raw")
            .string();

    // AND a device configured to read by batches of n events
    RawFileConfig cfg;
    cfg.n_events_to_read_ = 10000;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder = device->get_facility<I_EventsStreamDecoder>();
    auto es      = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, es);

    es->start();

    // WHEN we stream and decode data byte by byte
    Metavision::timestamp previous_ts_last = 0;
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 1;
        auto raw_buffer_decode_to     = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);

            // THEN the timestamps increase monotonously
            Metavision::timestamp current_ts_last = decoder->get_last_timestamp();
            if (current_ts_last > 0) {
                ASSERT_GE(current_ts_last, previous_ts_last);
                previous_ts_last = current_ts_last;
            }

            cur_raw_ptr = raw_buffer_decode_to;
            raw_buffer_decode_to =
                std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.data() + raw_buffer.size());
        }
    }
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt3_data_nominal) {
    // GIVEN a RAW file in EVT3 format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt3_hand.raw").string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode the events in the file
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        decoder->decode(raw_buffer);
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(18094969, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt3_data_random_split) {
    // GIVEN a RAW file in EVT3 format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt3_hand.raw").string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode events in buffer of random size
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 2 * (raw_buffer.size() / 10) +
                                        1; // Ensures odd number of bytes so that we have split in middle of raw event
        auto raw_buffer_decode_to = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);
            cur_raw_ptr = raw_buffer_decode_to;
            raw_buffer_decode_to =
                std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.data() + raw_buffer.size());
        }
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(18094969, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt3_data_byte_by_byte) {
    // GIVEN a RAW file in EVT3 format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt3_hand.raw").string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode data byte by byte data
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 1;
        auto raw_buffer_decode_to     = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);
            cur_raw_ptr = raw_buffer_decode_to;
            raw_buffer_decode_to =
                std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.data() + raw_buffer.size());
        }
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(18094969, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt3_nevents_monotonous_timestamps) {
    // GIVEN a RAW file in EVT3 format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt3_hand.raw").string();

    // AND a device configured to read by batches of n events
    RawFileConfig cfg;
    cfg.n_events_to_read_ = 10000;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder = device->get_facility<I_EventsStreamDecoder>();
    auto es      = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, es);

    es->start();

    // WHEN we stream and decode data byte by byte
    Metavision::timestamp previous_ts_last = 0;
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 1;
        auto raw_buffer_decode_to     = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);

            // THEN the timestamps increase monotonously
            Metavision::timestamp current_ts_last = decoder->get_last_timestamp();
            if (current_ts_last > 0) {
                ASSERT_GE(current_ts_last, previous_ts_last);
                previous_ts_last = current_ts_last;
            }

            cur_raw_ptr = raw_buffer_decode_to;
            raw_buffer_decode_to =
                std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.data() + raw_buffer.size());
        }
    }
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt3_erc_count_evts) {
    // GIVEN a RAW file in EVT3 format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt3_hand.raw").string();

    size_t cd_erc_in_total_count  = 0;
    size_t cd_erc_out_total_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD count events
    auto decoder           = device->get_facility<I_EventsStreamDecoder>();
    auto erc_count_decoder = device->get_facility<I_EventDecoder<EventERCCounter>>();
    auto es                = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, erc_count_decoder);
    ASSERT_NE(nullptr, es);

    erc_count_decoder->add_event_buffer_callback([&](auto ev_begin, auto ev_end) {
        ASSERT_EQ(1, std::distance(ev_begin, ev_end));
        if (ev_begin->is_output)
            cd_erc_out_total_count += ev_begin->event_count;
        else
            cd_erc_in_total_count += ev_begin->event_count;
    });

    es->start();

    // WHEN we stream and decode the events in the file
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        decoder->decode(raw_buffer);
    }

    // THEN We decode the same data CD incoming count events that are encoded in the RAW file
    ASSERT_EQ(18158913, cd_erc_in_total_count);
    ASSERT_EQ(18095375, cd_erc_out_total_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt4_data_nominal) {
    // GIVEN a RAW file in EVT4 format with a known content
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "claque_doigt_evt4.raw";

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode the events in the file
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        decoder->decode(raw_buffer);
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(16294351, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt4_data_random_split) {
    // GIVEN a RAW file in EVT4 format with a known content
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "claque_doigt_evt4.raw";

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode events in buffer of random size
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 2 * (raw_buffer.size() / 10) +
                                        1; // Ensures odd number of bytes so that we have split in middle of raw event
        auto raw_buffer_decode_to = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);
            cur_raw_ptr          = raw_buffer_decode_to;
            raw_buffer_decode_to = std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.end());
        }
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(16294351, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt4_data_byte_by_byte) {
    // GIVEN a RAW file in EVT4 format with a known content
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "claque_doigt_evt4.raw";

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode data byte by byte data
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 1;
        auto raw_buffer_decode_to     = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);
            cur_raw_ptr          = raw_buffer_decode_to;
            raw_buffer_decode_to = std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.end());
        }
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(16294351, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt4_nevents_monotonous_timestamps) {
    // GIVEN a RAW file in EVT4 format with a known content
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "claque_doigt_evt4.raw";

    // AND a device configured to read by batches of n events
    RawFileConfig cfg;
    cfg.n_events_to_read_ = 10000;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder = device->get_facility<I_EventsStreamDecoder>();
    auto es      = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, es);

    es->start();

    // WHEN we stream and decode data byte by byte
    Metavision::timestamp previous_ts_last = 0;
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 1;
        auto raw_buffer_decode_to     = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);

            // THEN the timestamps increase monotonously
            Metavision::timestamp current_ts_last = decoder->get_last_timestamp();
            if (current_ts_last > 0) {
                ASSERT_GE(current_ts_last, previous_ts_last);
                previous_ts_last = current_ts_last;
            }

            cur_raw_ptr          = raw_buffer_decode_to;
            raw_buffer_decode_to = std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.end());
        }
    }
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt4_erc_count_evts) {
    // GIVEN a RAW file in EVT4 format with a known content
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt4_hand.raw";

    size_t cd_erc_in_total_count  = 0;
    size_t cd_erc_out_total_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD count events
    auto decoder           = device->get_facility<I_EventsStreamDecoder>();
    auto erc_count_decoder = device->get_facility<I_EventDecoder<EventERCCounter>>();
    auto es                = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, erc_count_decoder);
    ASSERT_NE(nullptr, es);

    erc_count_decoder->add_event_buffer_callback([&](auto ev_begin, auto ev_end) {
        ASSERT_EQ(1, std::distance(ev_begin, ev_end));
        if (ev_begin->is_output)
            cd_erc_out_total_count += ev_begin->event_count;
        else
            cd_erc_in_total_count += ev_begin->event_count;
    });

    es->start();

    // WHEN we stream and decode the events in the file
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        decoder->decode(raw_buffer);
    }

    // THEN We decode the same data CD incoming count events that are encoded in the RAW file
    ASSERT_EQ(18158913, cd_erc_in_total_count);
    ASSERT_EQ(18095375, cd_erc_out_total_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_histo3d_nominal) {
    // GIVEN a RAW file in histo3d format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "histo3d.raw").string();

    size_t received_event_frame_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of histograms
    auto histo_decoder = device->get_facility<I_EventFrameDecoder<RawEventFrameHisto>>();
    auto es            = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, histo_decoder);
    ASSERT_NE(nullptr, es);

    histo_decoder->add_event_frame_callback([&](auto histo) { received_event_frame_count++; });

    es->start();

    // WHEN we stream and decode the events in the file
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        histo_decoder->decode(raw_buffer.begin(), raw_buffer.end());
    }

    // THEN We decode the same data Histogram that are encoded in the RAW file
    ASSERT_EQ(302, received_event_frame_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_histo3d_padding_nominal) {
    // GIVEN a RAW file in histo3d + padding format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "histo3d_padding.raw").string();

    size_t received_event_frame_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of histograms
    auto histo_decoder = device->get_facility<I_EventFrameDecoder<RawEventFrameHisto>>();
    auto es            = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, histo_decoder);
    ASSERT_NE(nullptr, es);

    histo_decoder->add_event_frame_callback([&](auto histo) { received_event_frame_count++; });

    es->start();

    // WHEN we stream and decode the events in the file
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        histo_decoder->decode(raw_buffer.begin(), raw_buffer.end());
    }

    // THEN We decode the same data Histogram that are encoded in the RAW file
    ASSERT_EQ(302, received_event_frame_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_diff3d_nominal) {
    // GIVEN a RAW file in histo3d + padding format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "diff3d.raw").string();

    size_t received_event_frame_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of diffs
    auto diff_decoder = device->get_facility<I_EventFrameDecoder<RawEventFrameDiff>>();
    auto es           = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, diff_decoder);
    ASSERT_NE(nullptr, es);

    diff_decoder->add_event_frame_callback([&](auto diff) { received_event_frame_count++; });

    es->start();

    // WHEN we stream and decode the events in the file
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        diff_decoder->decode(raw_buffer.begin(), raw_buffer.end());
    }

    // THEN We decode the same data Diff that are encoded in the RAW file
    ASSERT_EQ(301, received_event_frame_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_aer8_data_nominal) {
    // GIVEN a RAW file in AER format with 8bits interface with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "aer_8bits.raw").string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode the events in the file
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        decoder->decode(raw_buffer);
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(31300, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_aer8_data_random_split) {
    // GIVEN a RAW file in AER format with 8bits interface with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "aer_8bits.raw").string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode events in buffer of random size
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 2 * (raw_buffer.size() / 10) +
                                        1; // Ensures odd number of bytes so that we have split in middle of raw event
        auto raw_buffer_decode_to = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);
            cur_raw_ptr = raw_buffer_decode_to;
            raw_buffer_decode_to =
                std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.data() + raw_buffer.size());
        }
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(31300, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_aer8_data_byte_by_byte) {
    // GIVEN a RAW file in AER format with 8bits interface with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "aer_8bits.raw").string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode data byte by byte data
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 1;
        auto raw_buffer_decode_to     = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);
            cur_raw_ptr = raw_buffer_decode_to;
            raw_buffer_decode_to =
                std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.data() + raw_buffer.size());
        }
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(31300, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_aer4_data_nominal) {
    // GIVEN a RAW file in AER format with 4bits interface with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "aer_4bits.raw").string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode the events in the file
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        decoder->decode(raw_buffer);
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(31300, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_aer4_data_random_split) {
    // GIVEN a RAW file in AER format with 4bits interface with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "aer_4bits.raw").string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode events in buffer of random size
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 2 * (raw_buffer.size() / 10) +
                                        1; // Ensures odd number of bytes so that we have split in middle of raw event
        auto raw_buffer_decode_to = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);
            cur_raw_ptr = raw_buffer_decode_to;
            raw_buffer_decode_to =
                std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.data() + raw_buffer.size());
        }
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(31300, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_aer4_data_byte_by_byte) {
    // GIVEN a RAW file in AER format with 4bits interface with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "aer_4bits.raw").string();

    size_t received_cd_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE decoder of CD & triggers events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    es->start();

    // WHEN we stream and decode data byte by byte data
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer               = es->get_latest_raw_data();
        auto cur_raw_ptr              = raw_buffer.data();
        auto raw_data_to_decode_count = 1;
        auto raw_buffer_decode_to     = cur_raw_ptr + raw_data_to_decode_count;

        for (; static_cast<std::size_t>(std::distance(raw_buffer.data(), cur_raw_ptr)) < raw_buffer.size();) {
            decoder->decode(cur_raw_ptr, raw_buffer_decode_to);
            cur_raw_ptr = raw_buffer_decode_to;
            raw_buffer_decode_to =
                std::min(cur_raw_ptr + raw_data_to_decode_count, raw_buffer.data() + raw_buffer.size());
        }
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(31300, received_cd_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_mtr_nominal) {
    // GIVEN a RAW file in MTR format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "0101_cm_mtr12_output.raw")
            .string();

    size_t received_event_frame_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE MTR decoder
    auto mtr_decoder = device->get_facility<I_EventFrameDecoder<PointCloud>>();
    auto es          = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, mtr_decoder);
    ASSERT_NE(nullptr, es);

    mtr_decoder->add_event_frame_callback([&](auto pointcloud) { received_event_frame_count++; });

    es->start();

    // WHEN we stream and decode the events in the file
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        mtr_decoder->decode(raw_buffer.begin(), raw_buffer.end());
    }

    // THEN We decode the same MTR data that are encoded in the RAW file
    ASSERT_EQ(378, received_event_frame_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_mtru_nominal) {
    // GIVEN a RAW file in MTRU format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "0101_cm_mtru_output.raw")
            .string();

    size_t received_event_frame_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    if (!device) {
        std::cerr << "Failed to open raw file." << std::endl;
        FAIL();
    }

    // AND a PSEE MTR decoder
    auto mtr_decoder = device->get_facility<I_EventFrameDecoder<PointCloud>>();
    auto es          = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, mtr_decoder);
    ASSERT_NE(nullptr, es);

    mtr_decoder->add_event_frame_callback([&](auto pointcloud) { received_event_frame_count++; });

    es->start();

    // WHEN we stream and decode the events in the file
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        mtr_decoder->decode(raw_buffer.begin(), raw_buffer.end());
    }

    // THEN We decode the same MTR data that are encoded in the RAW file
    ASSERT_EQ(378, received_event_frame_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt2_monitoring_events) {
    // GIVEN a RAW file in EVT2 format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" /
         "lifo_evt2.raw").string();

    size_t received_cd_event_count = 0;
    size_t received_monitoring_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    ASSERT_TRUE(device) << "Failed to open raw file";

    // AND a PSEE decoder of CD & monitoring events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto monitoring_decoder = device->get_facility<I_EventDecoder<EventMonitoring>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    monitoring_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) {
            for (auto it = ev_begin; it < ev_end; ++it) {
                // 0x42 = MASTER_ANA_LIFO_ON
                ASSERT_EQ(0x42, it->type_id);
            }
            received_monitoring_event_count += std::distance(ev_begin, ev_end); });


    es->start();

    // WHEN we stream and decode the events in the file
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        decoder->decode(raw_buffer);
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(8314158, received_cd_event_count);
    ASSERT_EQ(493, received_monitoring_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt21_monitoring_events) {
    // GIVEN a RAW file in EVT21 format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" /
         "lifo_evt21.raw").string();

    size_t received_cd_event_count = 0;
    size_t received_monitoring_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    ASSERT_TRUE(device) << "Failed to open raw file";

    // AND a PSEE decoder of CD & monitoring events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto monitoring_decoder = device->get_facility<I_EventDecoder<EventMonitoring>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    monitoring_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) {
            for (auto it = ev_begin; it < ev_end; ++it) {
                // 0x42 = MASTER_ANA_LIFO_ON
                ASSERT_EQ(0x42, it->type_id);
            }
            received_monitoring_event_count += std::distance(ev_begin, ev_end); });


    es->start();

    // WHEN we stream and decode the events in the file
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        decoder->decode(raw_buffer);
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(8314158, received_cd_event_count);
    ASSERT_EQ(493, received_monitoring_event_count);
}

TEST_F_WITH_DATASET(PseeRawFileDecoder_Gtest, decode_evt3_monitoring_events) {
    // GIVEN a RAW file in EVT3 format with a known content
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" /
         "lifo_evt3.raw").string();

    size_t received_cd_event_count = 0;
    size_t received_monitoring_event_count = 0;
    RawFileConfig cfg;
    cfg.do_time_shifting_ = false;
    std::unique_ptr<Device> device(DeviceDiscovery::open_raw_file(dataset_file_path, cfg));

    ASSERT_TRUE(device) << "Failed to open raw file";

    // AND a PSEE decoder of CD & monitoring events
    auto decoder    = device->get_facility<I_EventsStreamDecoder>();
    auto cd_decoder = device->get_facility<I_EventDecoder<EventCD>>();
    auto monitoring_decoder = device->get_facility<I_EventDecoder<EventMonitoring>>();
    auto es         = device->get_facility<I_EventsStream>();

    ASSERT_NE(nullptr, decoder);
    ASSERT_NE(nullptr, cd_decoder);
    ASSERT_NE(nullptr, es);

    cd_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) { received_cd_event_count += std::distance(ev_begin, ev_end); });

    monitoring_decoder->add_event_buffer_callback(
        [&](auto ev_begin, auto ev_end) {
            for (auto it = ev_begin; it < ev_end; ++it) {
                // 0x42 = MASTER_ANA_LIFO_ON
                ASSERT_EQ(0x42, it->type_id);
            }
            received_monitoring_event_count += std::distance(ev_begin, ev_end); });


    es->start();

    // WHEN we stream and decode the events in the file
    while (es->wait_next_buffer() >= 0) {
        auto raw_buffer = es->get_latest_raw_data();
        decoder->decode(raw_buffer);
    }

    // THEN We decode the same data CD & triggers that are encoded in the RAW file
    ASSERT_EQ(8314158, received_cd_event_count);
    ASSERT_EQ(493, received_monitoring_event_count);
}
