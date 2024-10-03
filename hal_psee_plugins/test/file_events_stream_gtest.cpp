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
#include <sstream>
#include <memory>
#include <numeric>

#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/decoders/evt2/evt2_decoder.h"
#include "devices/utils/device_system_id.h"
#include "metavision/psee_hw_layer/boards/rawfile/psee_raw_file_header.h"

using namespace Metavision;

using RawEventType = EVT2Decoder::Event_Word_Type;

class FileEventsStream_Gtest : public GTestWithTmpDir {
protected:
    virtual void SetUp() override {
        reset_device();

        // Create and open the rawfile
        static int raw_counter = 1;
        rawfile_to_log_path_   = tmpdir_handler_->get_full_path("rawfile_" + std::to_string(++raw_counter) + ".raw");
        rawfile_to_log_from_rawfile_path_ = rawfile_to_log_path_ + ".logged";

        config_.n_read_buffers_   = n_read_buffers_default_;
        config_.n_events_to_read_ = n_events_to_read_default_;
    }

    virtual void TearDown() override {
        close_raw();
    }

    void reset_device() {
        // create instances
        device_.reset();

        file_events_stream_ = nullptr;
        file_events_stream_ = nullptr;
    }

    void open_raw() {
        rawfile_to_log_.reset(new std::ofstream(rawfile_to_log_path_, std::ios::out | std::ios::binary));
        if (!rawfile_to_log_) {
            std::cerr << "Could not open file for writing at " << rawfile_to_log_path_ << std::endl;
            FAIL();
        }
    }

    void close_raw() {
        if (rawfile_to_log_) {
            rawfile_to_log_->close();
            rawfile_to_log_.reset(nullptr);
        }
    }

    PseeRawFileHeader write_random_header() {
        auto header = std::stringstream();
        header << "% Date 2014-02-28 13:37:42" << std::endl
               << "% system_ID " << gen41_system_id << std::endl
               << "% integrator_name Prophesee" << std::endl
               << "% firmware_version 0.0.0" << std::endl
               << "% plugin_name hal_plugin_prophesee" << std::endl
               << "% evt 2.0" << std::endl
               << "% subsystem_ID " << dummy_sub_system_id_ << std::endl
               << "% " << dummy_custom_key_ << " " << dummy_custom_value_ << std::endl
               << "% serial_number " << dummy_serial_ << std::endl;
        PseeRawFileHeader header_to_write(header);

        (*rawfile_to_log_) << header_to_write;

        return header_to_write;
    }

    std::vector<RawEventType> write_ref_data() {
        uint32_t n_buffers_count = 0;
        auto data_to_write       = std::vector<RawEventType>(n_events_per_writen_buffer_default_);
        std::iota(data_to_write.begin(), data_to_write.end(), 1);

        std::vector<RawEventType> data_ref;

        const size_t expected_data_size = n_buffers_default_ * n_events_per_writen_buffer_default_;

        // writes data
        while (++n_buffers_count <= n_buffers_default_) {
            // write data
            data_ref.insert(data_ref.end(), data_to_write.begin(), data_to_write.end());
            rawfile_to_log_->write(reinterpret_cast<char *>(data_to_write.data()),
                                   data_to_write.size() * sizeof(RawEventType));
        }

        EXPECT_EQ(data_ref.size(), expected_data_size);

        return data_ref;
    }

    bool open_file_events_stream(std::string file_to_load = "") {
        reset_device();
        bool ret     = true;
        file_to_load = file_to_load != "" ? file_to_load : rawfile_to_log_path_;

        // Tries to build a regular File Events Stream
        // As it needs to have access to the 'find_neighboor...' function from the decoder, we open the rawfile with
        // device. Then we get the decoder.
        device_ = DeviceDiscovery::open_raw_file(file_to_load, config_);

        if (!device_) {
            // If device couldn't be built, we don't do anything with it and we keep going
            // The idea here is ONLY to test the File Event Stream class
            reset_device();
        } else {
            file_events_stream_ = device_->get_facility<I_EventsStream>();
            file_id_            = device_->get_facility<I_HW_Identification>();
            decoder_            = device_->get_facility<I_EventsStreamDecoder>();

            return (file_events_stream_ != nullptr);
        }

        return true;
    }

    bool open_and_start_file_events_stream(std::string file_to_load = "") {
        if (open_file_events_stream(file_to_load)) {
            file_events_stream_->start();
            return true;
        }

        return false;
    }

public:
    static constexpr uint32_t n_events_per_writen_buffer_default_ = 20, n_buffers_default_ = 5;
    static constexpr uint32_t n_events_to_read_default_ = 15; // fixed size buffers
    static constexpr uint32_t n_read_buffers_default_   = 3;
    static constexpr uint32_t n_events_read_in_last_buffer_ =
        (n_events_per_writen_buffer_default_ * n_buffers_default_) % n_events_to_read_default_;

    static const std::string dummy_serial_;
    static const std::string dummy_events_type_;
    static const std::string dummy_custom_key_;
    static const std::string dummy_custom_value_;

    static constexpr SystemId gen41_system_id   = SystemId::SYSTEM_EVK3_GEN41;
    static constexpr long dummy_sub_system_id_  = 0;
    static constexpr long dummy_system_version_ = 0;

protected:
    std::unique_ptr<Device> device_;
    I_EventsStream *file_events_stream_ = nullptr;
    I_EventsStreamDecoder *decoder_     = nullptr;
    I_HW_Identification *file_id_       = nullptr;

    std::unique_ptr<std::ofstream> rawfile_to_log_;
    std::string rawfile_to_log_path_;
    std::string rawfile_to_log_from_rawfile_path_;
    RawFileConfig config_;
};

constexpr uint32_t FileEventsStream_Gtest::n_events_per_writen_buffer_default_;
constexpr uint32_t FileEventsStream_Gtest::n_read_buffers_default_;
constexpr uint32_t FileEventsStream_Gtest::n_buffers_default_;
constexpr uint32_t FileEventsStream_Gtest::n_events_to_read_default_;
constexpr uint32_t FileEventsStream_Gtest::n_events_read_in_last_buffer_;

constexpr long FileEventsStream_Gtest::dummy_system_version_;
constexpr SystemId FileEventsStream_Gtest::gen41_system_id;
const long FileEventsStream_Gtest::dummy_sub_system_id_;
const std::string FileEventsStream_Gtest::dummy_serial_       = "dummy_serial";
const std::string FileEventsStream_Gtest::dummy_events_type_  = "events_type";
const std::string FileEventsStream_Gtest::dummy_custom_key_   = "custom";
const std::string FileEventsStream_Gtest::dummy_custom_value_ = "field";

TEST_F(FileEventsStream_Gtest, file_does_not_exists) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Check that you can't read a file that doesn't exist

    ASSERT_THROW(open_and_start_file_events_stream(), HalException);
}

TEST_F(FileEventsStream_Gtest, header_expected) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that the read header contains the same information as the written one

    open_raw();
    auto written_header = write_random_header();
    write_ref_data();
    close_raw();

    ASSERT_TRUE(open_and_start_file_events_stream());
    auto a = written_header.get_header_map();
    auto b = file_id_->get_header().get_header_map();
    ASSERT_EQ(a, b);
}

TEST_F(FileEventsStream_Gtest, reading_all_data) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Check that file events streams reads all data

    open_raw();
    write_random_header();
    auto data_ref = write_ref_data();
    close_raw();

    // Loads the file
    ASSERT_TRUE(open_and_start_file_events_stream());

    std::vector<RawEventType> data_read;

    while (file_events_stream_->wait_next_buffer() > 0) {
        auto buffer = file_events_stream_->get_latest_raw_data();
        ASSERT_TRUE(buffer.size() == (n_events_to_read_default_ * decoder_->get_raw_event_size_bytes()) ||
                    buffer.size() == (n_events_read_in_last_buffer_ * decoder_->get_raw_event_size_bytes()));
        data_read.insert(data_read.end(), reinterpret_cast<const RawEventType *>(buffer.data()),
                         reinterpret_cast<const RawEventType *>(buffer.data() + buffer.size()));
    }

    ASSERT_EQ(data_ref, data_read);

    // End of file
    ASSERT_EQ(file_events_stream_->wait_next_buffer(), -1);
}

TEST_F(FileEventsStream_Gtest, do_not_read_if_not_started) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Verify that we can not read until start is called

    open_raw();
    write_random_header();
    write_ref_data();
    close_raw();

    ASSERT_TRUE(open_file_events_stream());

    // fails to read
    ASSERT_EQ(-1, file_events_stream_->wait_next_buffer());

    // Enable the streaming
    file_events_stream_->start();

    // read data successfully
    ASSERT_EQ(1, file_events_stream_->wait_next_buffer());
}

TEST_F(FileEventsStream_Gtest, file_events_stream_read_param) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // If read params

    open_raw();
    write_random_header();
    auto data_ref = write_ref_data();
    close_raw();

    config_.n_events_to_read_ = 0;
    ASSERT_THROW(open_and_start_file_events_stream(), HalException);
}

TEST_F(FileEventsStream_Gtest, rawfile_logging_input_output_same_name_fails) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // You can't log to the same file you are reading

    open_raw();
    write_random_header();
    write_ref_data();
    close_raw();

    // Open should return nullptr
    ASSERT_TRUE(open_file_events_stream());
    ASSERT_FALSE(file_events_stream_->log_raw_data(rawfile_to_log_path_));
}

TEST_F(FileEventsStream_Gtest, rawfile_logging_header_is_same_from_input_to_output) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Expects that the header file of the source RAW file is the same as the header in the output file

    open_raw();
    auto written_header = write_random_header();
    write_ref_data();
    close_raw();

    ASSERT_TRUE(open_file_events_stream());
    ASSERT_TRUE(file_events_stream_->log_raw_data(rawfile_to_log_from_rawfile_path_));
    file_events_stream_->start();

    while (file_events_stream_->wait_next_buffer() > 0) {
        file_events_stream_->get_latest_raw_data();
    }

    reset_device();
    ASSERT_TRUE(open_and_start_file_events_stream(rawfile_to_log_from_rawfile_path_));

    written_header.remove_date();
    auto header = file_id_->get_header();
    header.remove_date();
    ASSERT_EQ(written_header.get_header_map(), header.get_header_map());
}

TEST_F(FileEventsStream_Gtest, rawfile_logging_data_logged_is_data_read) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Expect that what is logged from the rawfile is what is read

    open_raw();
    write_random_header();
    auto data_ref = write_ref_data();
    close_raw();

    ASSERT_TRUE(open_file_events_stream());

    ASSERT_TRUE(file_events_stream_->log_raw_data(rawfile_to_log_from_rawfile_path_));
    file_events_stream_->start();
    while (file_events_stream_->wait_next_buffer() > 0) {
        file_events_stream_->get_latest_raw_data();
    }
    reset_device();

    // Read logged file
    ASSERT_TRUE(open_and_start_file_events_stream(rawfile_to_log_from_rawfile_path_));

    std::vector<RawEventType> data_read;
    while (file_events_stream_->wait_next_buffer() > 0) {
        auto buffer = file_events_stream_->get_latest_raw_data();
        ASSERT_TRUE(buffer.size() == n_events_to_read_default_ * decoder_->get_raw_event_size_bytes() ||
                    buffer.size() == n_events_read_in_last_buffer_ * decoder_->get_raw_event_size_bytes());
        data_read.insert(data_read.end(), reinterpret_cast<const RawEventType *>(buffer.begin()),
                         reinterpret_cast<const RawEventType *>(buffer.end()));
    }

    ASSERT_EQ(data_ref, data_read);

    // End of file
    ASSERT_EQ(-1, file_events_stream_->wait_next_buffer());
}

TEST_F(FileEventsStream_Gtest, reading_from_custom_istream) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Check that file events streams reads all data

    open_raw();
    auto header   = write_random_header();
    auto data_ref = write_ref_data();
    close_raw();

    auto ss = new std::stringstream(std::ios::in | std::ios::out | std::ios::binary);
    std::unique_ptr<std::istream> stream(ss);

    *ss << header;
    ss->write(reinterpret_cast<char *>(data_ref.data()), data_ref.size() * sizeof(RawEventType));

    device_ = DeviceDiscovery::open_stream(std::move(stream), config_);
    ASSERT_NE(nullptr, device_.get());

    file_events_stream_ = device_->get_facility<I_EventsStream>();
    file_id_            = device_->get_facility<I_HW_Identification>();
    decoder_            = device_->get_facility<I_EventsStreamDecoder>();
    ASSERT_NE(nullptr, file_events_stream_);
    ASSERT_NE(nullptr, file_id_);

    file_events_stream_->start();

    std::vector<RawEventType> data_read;
    while (file_events_stream_->wait_next_buffer() > 0) {
        auto buffer = file_events_stream_->get_latest_raw_data();
        ASSERT_TRUE(buffer.size() == n_events_to_read_default_ * decoder_->get_raw_event_size_bytes() ||
                    buffer.size() == n_events_read_in_last_buffer_ * decoder_->get_raw_event_size_bytes());
        data_read.insert(data_read.end(), reinterpret_cast<const RawEventType *>(buffer.begin()),
                         reinterpret_cast<const RawEventType *>(buffer.end()));
    }

    ASSERT_EQ(data_ref, data_read);

    // End of file
    ASSERT_EQ(file_events_stream_->wait_next_buffer(), -1);
}
