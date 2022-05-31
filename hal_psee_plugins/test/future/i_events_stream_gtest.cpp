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

#include <array>
#include <boost/filesystem.hpp>

#include "metavision/utils/gtest/gtest_custom.h"
#include "metavision/hal/facilities/future/i_events_stream.h"
#include "metavision/hal/facilities/future/i_decoder.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_hal_software_info.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/sdk/base/utils/generic_header.h"

using namespace Metavision;

class I_EventsStream_Gtest : public ::testing::Test {
protected:
    virtual void SetUp() override {
        datasets_.push_back(
            (boost::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw")
                .string());
        datasets_.push_back(
            (boost::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt2_hand.raw")
                .string());
        datasets_.push_back(
            (boost::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt3_hand.raw")
                .string());
    }

    virtual void TearDown() override {}

    bool open_dataset(const std::string &dataset_name, Future::RawFileConfig config = Future::RawFileConfig()) {
        device_ = DeviceDiscovery::open_raw_file(dataset_name, config);
        return nullptr != device_.get();
    }

protected:
    std::unique_ptr<Device> device_;
    std::vector<std::string> datasets_;
};

TEST_F_WITH_DATASET(I_EventsStream_Gtest, valid_index_file) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Check that when reading a file, a valid index is generated

    std::array<std::string, 3> sizes     = {"121064697", "71825543", "96786995"};
    std::array<std::string, 3> ts_shifts = {"7451072", "66619456", "14663680"};
    char magic_number_array[] = {'\x54', '\xe2', '\x61', '\xbd', '\x57', '\x4c', '\x0d', '\x30', '\x47', '\x34',
                                 '\x5d', '\xdc', '\x04', '\x49', '\x2a', '\x5a', '\xde', '\x09', '\xd3', '\x61'};
    std::array<char, sizeof(magic_number_array)> magic_number;
    std::copy(magic_number_array, magic_number_array + sizeof(magic_number_array), magic_number.data());

    for (size_t i = 0; i < datasets_.size(); ++i) {
        const auto &dataset = datasets_[i];
        // Remove the index file if any
        std::string path;
        boost::filesystem::remove(dataset + ".tmp_index");
        ASSERT_FALSE(boost::filesystem::exists(dataset + ".tmp_index"));

        // Open the file and generate the index
        ASSERT_TRUE(open_dataset(dataset));

        constexpr uint32_t max_trials = 1000;
        uint32_t trials               = 1;
        // Ensure the index have been built
        auto fes = device_->get_facility<Future::I_EventsStream>();
        ASSERT_NE(nullptr, fes);
        timestamp a, b;
        auto s = fes->get_seek_range(a, b);
        for (trials = 1; s != Metavision::Future::I_EventsStream::IndexStatus::Good && trials < max_trials; ++trials) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            s = fes->get_seek_range(a, b);
        }
        ASSERT_LT(trials, max_trials);

        // Check the index content
        std::ifstream index_file(dataset + ".tmp_index", std::ios::binary);
        ASSERT_TRUE(index_file);
        GenericHeader index_header(index_file);
#if defined _WIN32
        EXPECT_EQ("Windows", index_header.get_field("platform"));
#elif defined __APPLE__
        EXPECT_EQ("Darwin", index_header.get_field("platform"));
#else
        EXPECT_EQ("Linux", index_header.get_field("platform"));
#endif
        EXPECT_EQ(device_->get_facility<I_HALSoftwareInfo>()->get_software_info().get_version(),
                  index_header.get_field("hal_version"));
        EXPECT_EQ(device_->get_facility<I_PluginSoftwareInfo>()->get_software_info().get_version(),
                  index_header.get_field("hal_plugin_version"));
        EXPECT_EQ(sizes[i], index_header.get_field("size"));
        EXPECT_EQ("2000", index_header.get_field("bookmark_period_us"));
        EXPECT_EQ("2.0", index_header.get_field("index_version"));
        EXPECT_EQ(ts_shifts[i], index_header.get_field("ts_shift_us"));

        index_file.clear();
        index_file.seekg(-magic_number.size(), std::ios::end);
        std::array<char, magic_number.size()> buf;
        ASSERT_TRUE(index_file.read(buf.data(), magic_number.size()));
        for (size_t i = 0; i < magic_number.size(); ++i) {
            EXPECT_EQ(magic_number[i], buf[i]);
        }
    }
}

TEST_F_WITH_DATASET(I_EventsStream_Gtest, invalid_index_file) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Check that an invalid index file is replaced by a valid one when reading a file

    char magic_number_array[] = {'\x54', '\xe2', '\x61', '\xbd', '\x57', '\x4c', '\x0d', '\x30', '\x47', '\x34',
                                 '\x5d', '\xdc', '\x04', '\x49', '\x2a', '\x5a', '\xde', '\x09', '\xd3', '\x61'};
    std::array<char, sizeof(magic_number_array)> magic_number;
    std::copy(magic_number_array, magic_number_array + sizeof(magic_number_array), magic_number.data());

    for (size_t i = 0; i < datasets_.size(); ++i) {
        const auto &dataset = datasets_[i];
        std::cout << "\n========================================" << std::endl;
        std::cout << "Path = " << dataset << std::endl;

        // Remove the index file if any
        std::string path;
        boost::filesystem::remove(dataset + ".tmp_index");
        ASSERT_FALSE(boost::filesystem::exists(dataset + ".tmp_index"));

        // Open the file and generate the index
        ASSERT_TRUE(open_dataset(dataset));

        constexpr uint32_t max_trials = 1000;
        uint32_t trials               = 1;
        // Ensure the index have been built
        auto fes = device_->get_facility<Future::I_EventsStream>();
        ASSERT_NE(nullptr, fes);
        timestamp a, b;
        auto s = fes->get_seek_range(a, b);
        for (trials = 1; s != Metavision::Future::I_EventsStream::IndexStatus::Good && trials < max_trials; ++trials) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            s = fes->get_seek_range(a, b);
        }
        ASSERT_LT(trials, max_trials);

        // Get ref index content
        std::ifstream ref_index_file(dataset + ".tmp_index", std::ios::binary);
        GenericHeader ref_index_header(ref_index_file);
        ref_index_header.remove_date(); // this will differ for each comparison
        std::vector<char> ref_index_content((std::istreambuf_iterator<char>(ref_index_file)),
                                            (std::istreambuf_iterator<char>()));

        // Mess up the index file and check that it is automatically recreated with correct content when opening the
        // file again
        const std::unordered_map<std::string, std::string> keys = {
            {"platform", "blorb"},          {"hal_version", "X.Y.Z"},   {"hal_plugin_version", "X.Y.Z"}, {"size", "42"},
            {"bookmark_period_us", "3.14"}, {"index_version", "X.Y.Z"}, {"ts_shift_us", "gna"},
        };
        for (const auto &p : keys) {
            std::cout << "Messing up key " << p.first << std::endl;
            // Mess up the index
            GenericHeader out_index_header(ref_index_header.get_header_map());
            out_index_header.set_field(p.first, p.second);

            // Write index file
            std::ofstream out_index_file(dataset + ".tmp_index", std::ios::binary);
            out_index_file << out_index_header;
            out_index_file.write(ref_index_content.data(), ref_index_content.size());
            out_index_file.close();

            // Open the file and get the index
            ASSERT_TRUE(open_dataset(dataset));

            constexpr uint32_t max_trials = 1000;
            uint32_t trials               = 1;
            // Ensure the index have been built
            auto fes = device_->get_facility<Future::I_EventsStream>();
            ASSERT_NE(nullptr, fes);
            timestamp a, b;
            auto s = fes->get_seek_range(a, b);
            for (trials = 1; s != Metavision::Future::I_EventsStream::IndexStatus::Good && trials < max_trials;
                 ++trials) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                s = fes->get_seek_range(a, b);
            }
            ASSERT_LT(trials, max_trials);

            // Check that the index file has indeed the ref content, not the messed up one
            std::ifstream in_index_file(dataset + ".tmp_index", std::ios::binary);
            ASSERT_TRUE(in_index_file);
            GenericHeader in_index_header(in_index_file);
            in_index_header.remove_date(); // this will differ for each comparison
            std::vector<char> in_index_content((std::istreambuf_iterator<char>(in_index_file)),
                                               (std::istreambuf_iterator<char>()));

            EXPECT_NE(ref_index_header.to_string(), out_index_header.to_string());
            EXPECT_EQ(ref_index_header.to_string(), in_index_header.to_string());
            EXPECT_EQ(ref_index_content.size(), in_index_content.size());
            for (size_t i = 0; i < ref_index_content.size(); ++i) {
                EXPECT_EQ(ref_index_content[i], in_index_content[i]);
            }
        }

        {
            std::cout << "Messing up magic number" << std::endl;
            // Write index file
            std::ofstream out_index_file(dataset + ".tmp_index", std::ios::binary);
            out_index_file << ref_index_header;
            std::vector<char> out_index_content = ref_index_content;
            std::fill(out_index_content.begin() + out_index_content.size() - magic_number.size(),
                      out_index_content.begin() + out_index_content.size(), static_cast<char>(42));
            out_index_file.write(out_index_content.data(), out_index_content.size());
            out_index_file.close();

            // Open the file and get the index
            ASSERT_TRUE(open_dataset(dataset));

            constexpr uint32_t max_trials = 1000;
            uint32_t trials               = 1;
            // Ensure the index have been built
            auto fes = device_->get_facility<Future::I_EventsStream>();
            ASSERT_NE(nullptr, fes);
            timestamp a, b;
            auto s = fes->get_seek_range(a, b);
            for (trials = 1; s != Metavision::Future::I_EventsStream::IndexStatus::Good && trials < max_trials;
                 ++trials) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                s = fes->get_seek_range(a, b);
            }
            ASSERT_LT(trials, max_trials);

            // Check that the index file has indeed the ref content, not the messed up one
            std::ifstream in_index_file(dataset + ".tmp_index", std::ios::binary);
            ASSERT_TRUE(in_index_file);
            GenericHeader in_index_header(in_index_file);
            in_index_header.remove_date(); // this will differ for each comparison
            std::vector<char> in_index_content((std::istreambuf_iterator<char>(in_index_file)),
                                               (std::istreambuf_iterator<char>()));

            EXPECT_EQ(ref_index_header.to_string(), in_index_header.to_string());
            EXPECT_EQ(ref_index_content.size(), in_index_content.size());
            for (size_t i = 0; i < ref_index_content.size(); ++i) {
                EXPECT_EQ(ref_index_content[i], in_index_content[i]);
            }
        }
    }
}

TEST_F_WITH_DATASET(I_EventsStream_Gtest, seek_range) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Check range decoding of the file control

    std::vector<bool> delete_index{{true, false}};
    std::vector<bool> time_shift{{true, false}};

    for (const auto &do_delete : delete_index) {
        MV_HAL_LOG_INFO() << (do_delete ? "Index building from scratch" : "Index loaded from file");
        for (const auto &dataset : datasets_) {
            if (do_delete) {
                std::string path;
                boost::filesystem::remove(dataset + ".tmp_index");
                ASSERT_FALSE(boost::filesystem::exists(dataset + ".tmp_index"));
            }

            MV_HAL_LOG_INFO() << "\tTesting dataset" << dataset;
            for (const auto &do_time_shifting : time_shift) {
                // Builds the device from the dataset
                MV_HAL_LOG_INFO() << "\t\tTime shift:" << (do_time_shifting ? "enabled" : "disabled");
                Future::RawFileConfig config;
                config.do_time_shifting_ = do_time_shifting;
                ASSERT_TRUE(open_dataset(dataset, config));

                // Ensures the index have been built and one can retrieve the timestamp range
                auto fes = device_->get_facility<Future::I_EventsStream>();
                ASSERT_NE(nullptr, fes);

                timestamp first_indexed_event_ts_us, last_indexed_event_ts_us;
                auto index_status = fes->get_seek_range(first_indexed_event_ts_us, last_indexed_event_ts_us);
                constexpr uint32_t max_trials = 1000;
                uint32_t trials               = 1;
                while (index_status != Metavision::Future::I_EventsStream::IndexStatus::Good && trials != max_trials) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    ++trials;
                    index_status = fes->get_seek_range(first_indexed_event_ts_us, last_indexed_event_ts_us);
                }

                ASSERT_LT(trials, max_trials);

                EXPECT_GT(last_indexed_event_ts_us, first_indexed_event_ts_us);

                auto decoder = device_->get_facility<Future::I_Decoder>();
                ASSERT_NE(nullptr, decoder);
                EXPECT_EQ(do_time_shifting, decoder->is_time_shifting_enabled());

                // Decode the full dataset
                // -- First compute the first event timestamp using the decoder
                fes->start();
                timestamp first_decoded_event_ts_us = decoder->get_last_timestamp();
                EXPECT_EQ(timestamp(-1), first_decoded_event_ts_us);
                timestamp last_decoded_event_ts_us;
                long bytes_polled_count;
                while (fes->wait_next_buffer() > 0) {
                    auto data      = fes->get_latest_raw_data(bytes_polled_count);
                    auto data_next = data + decoder->get_raw_event_size_bytes();
                    auto data_end  = data + bytes_polled_count;

                    for (; first_decoded_event_ts_us == decoder->get_last_timestamp() && data != data_end;
                         data = data_next, data_next += decoder->get_raw_event_size_bytes()) {
                        decoder->decode(data, data_next);
                    }

                    if (data != data_end) {
                        first_decoded_event_ts_us = decoder->get_last_timestamp();
                        decoder->decode(data, data_end);
                        break;
                    }
                }
                // -- Then decode the full dataset until the end
                while (fes->wait_next_buffer() > 0) {
                    auto data = fes->get_latest_raw_data(bytes_polled_count);
                    decoder->decode(data, data + bytes_polled_count);
                }
                last_decoded_event_ts_us = decoder->get_last_timestamp();

                // Checks the value correspondence with what has been decoded
                EXPECT_EQ(first_decoded_event_ts_us, first_indexed_event_ts_us);
                EXPECT_GE(last_decoded_event_ts_us, last_indexed_event_ts_us); // Last bookmark is not the last event.

                // If time shift, check the coherence of the first timestamps compare to the timeshift
                timestamp ts_shift_us;
                if (do_time_shifting) {
                    EXPECT_TRUE(decoder->get_timestamp_shift(ts_shift_us));
                    EXPECT_LE(ts_shift_us,
                              first_indexed_event_ts_us +
                                  ts_shift_us); // First_ts is the timestamp of the first non timer high event
                }
            }
        }
    }
}

TEST_F_WITH_DATASET(I_EventsStream_Gtest, file_index_seek) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Check seeking feature of the file control

    Future::RawFileConfig config;
    config.n_events_to_read_ = 10000;

    std::vector<bool> delete_index{{true, false}};
    std::vector<bool> time_shift{{true, false}};

    for (const auto &do_delete : delete_index) {
        MV_HAL_LOG_INFO() << (do_delete ? "Index building from scratch" : "Index loaded from file");
        for (const auto &dataset : datasets_) {
            if (do_delete) {
                std::string path;
                boost::filesystem::remove(dataset + ".tmp_index");
                ASSERT_FALSE(boost::filesystem::exists(dataset + ".tmp_index"));
            }

            MV_HAL_LOG_INFO() << "\tTesting dataset" << dataset;
            for (const auto &do_time_shifting : time_shift) {
                // Builds the device from the dataset
                MV_HAL_LOG_INFO() << "\t\tTime shift:" << (do_time_shifting ? "enabled" : "disabled");
                config.do_time_shifting_ = do_time_shifting;
                ASSERT_TRUE(open_dataset(dataset, config));

                // Ensures the index have been built and one can retrieve the timestamp range
                auto fes = device_->get_facility<Future::I_EventsStream>();
                ASSERT_NE(nullptr, fes);

                timestamp first_indexed_event_ts_us, last_indexed_event_ts_us;
                auto index_status = fes->get_seek_range(first_indexed_event_ts_us, last_indexed_event_ts_us);
                constexpr uint32_t max_trials = 1000;
                uint32_t trials               = 1;
                while (index_status != Metavision::Future::I_EventsStream::IndexStatus::Good && trials != max_trials) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    ++trials;
                    index_status = fes->get_seek_range(first_indexed_event_ts_us, last_indexed_event_ts_us);
                }

                ASSERT_LT(trials, max_trials);
                ASSERT_GT(last_indexed_event_ts_us, first_indexed_event_ts_us);

                auto decoder    = device_->get_facility<Future::I_Decoder>();
                auto cd_decoder = device_->get_facility<I_EventDecoder<EventCD>>();
                ASSERT_NE(nullptr, decoder);
                ASSERT_NE(nullptr, cd_decoder);
                ASSERT_EQ(do_time_shifting, decoder->is_time_shifting_enabled());

                // Start polling data
                fes->start();

                // GTest necessity only:
                // Initialize the decoder (i.e. wait to reach the first decodable event)
                // This is needed so that the seeking capability is validated
                bool valid_event = false;
                long bytes_polled_count;
                cd_decoder->add_event_buffer_callback([&](auto, auto) { valid_event = true; });
                uint8_t *data;
                while (!valid_event) {
                    // Read data from the file
                    ASSERT_TRUE(fes->wait_next_buffer() > 0);
                    data = fes->get_latest_raw_data(bytes_polled_count);
                    decoder->decode(data, data + bytes_polled_count);
                }

                // ------------------------------
                // Check seeking at the beginning
                MV_HAL_LOG_INFO() << "\t\t\tSeek first event";
                timestamp reached_ts;
                ASSERT_EQ(Future::I_EventsStream::SeekStatus::Success,
                          fes->seek(first_indexed_event_ts_us, reached_ts));
                ASSERT_EQ(first_indexed_event_ts_us, reached_ts);
                decoder->reset_timestamp(reached_ts);
                ASSERT_EQ(decoder->get_last_timestamp(), reached_ts);

                ASSERT_TRUE(fes->wait_next_buffer() > 0);
                // Decode a single event and check that the timestamp is correct
                data = fes->get_latest_raw_data(bytes_polled_count);

                // The first event decoded must have a timestamp that is equal to the first event's timestamp
                decoder->decode(data, data + decoder->get_raw_event_size_bytes());
                ASSERT_EQ(decoder->get_last_timestamp(), first_indexed_event_ts_us);

                // ------------------------------
                // Check seeking in the range of possible timestamps
                MV_HAL_LOG_INFO() << "\t\t\tSeek in range of available timestamp";

                std::vector<timestamp> targets;
                const timestamp timestamp_step = (last_indexed_event_ts_us - first_indexed_event_ts_us) / 10;
                for (uint32_t step = 1; step <= 10; ++step) {
                    targets.push_back(first_indexed_event_ts_us + step * timestamp_step);
                }

                // Seeks back and forth in the file
                using SizeType = std::vector<timestamp>::size_type;
                for (SizeType i = 0; i < targets.size(); ++i) {
                    auto target_ts_us = i % 2 ? targets[targets.size() - i] : targets[i];
                    ASSERT_EQ(Future::I_EventsStream::SeekStatus::Success, fes->seek(target_ts_us, reached_ts));
                    decoder->reset_timestamp(reached_ts);
                    ASSERT_LE(reached_ts, target_ts_us);

                    // Read data from the file
                    ASSERT_TRUE(fes->wait_next_buffer() > 0);
                    data = fes->get_latest_raw_data(bytes_polled_count);

                    // The first event decoded must have a timestamp that is equal to the reached timestamp
                    decoder->decode(data, data + decoder->get_raw_event_size_bytes());
                    ASSERT_EQ(decoder->get_last_timestamp(), reached_ts);
                }

                // ------------------------------
                // Check seeking at the end of the file
                MV_HAL_LOG_INFO() << "\t\t\tSeek last event";
                ASSERT_EQ(Future::I_EventsStream::SeekStatus::Success, fes->seek(last_indexed_event_ts_us, reached_ts));
                ASSERT_LE(reached_ts, last_indexed_event_ts_us);
                decoder->reset_timestamp(reached_ts);

                // Read data from the file
                ASSERT_TRUE(fes->wait_next_buffer() > 0);
                data = fes->get_latest_raw_data(bytes_polled_count);

                // The first event decoded must have a timestamp that is equal to the reached timestamp
                decoder->decode(data, data + decoder->get_raw_event_size_bytes());
                ASSERT_EQ(decoder->get_last_timestamp(), reached_ts);
            }
        }
    }
}

TEST_F_WITH_DATASET(I_EventsStream_Gtest, decode_evt3_nevents_monotonous_timestamps) {
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
    auto decoder = device->get_facility<Future::I_Decoder>();
    auto es      = device->get_facility<Future::I_EventsStream>();

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
