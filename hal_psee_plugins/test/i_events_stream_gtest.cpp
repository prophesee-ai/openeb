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
#include <initializer_list>
#include <memory>
#include <thread>
#include <condition_variable>

#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/utils/data_transfer.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/utils/device_builder.h"
#include "metavision/utils/gtest/gtest_custom.h"
#include "metavision/hal/decoders/evt2/evt2_decoder.h"
#include "metavision/psee_hw_layer/boards/rawfile/psee_raw_file_header.h"
#include "geometries/vga_geometry.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "tencoder.h"
#include "gen3CD_device.h"
#include "device_builder_maker.h"

using namespace Metavision;

class MockHWIdentification : public I_HW_Identification {
public:
    MockHWIdentification(const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info, long system_id) :
        I_HW_Identification(plugin_sw_info), system_id_(system_id) {}

    virtual std::string get_serial() const override {
        return dummy_serial_;
    }
    virtual SensorInfo get_sensor_info() const override {
        return SensorInfo();
    }
    virtual std::vector<std::string> get_available_data_encoding_formats() const override {
        std::vector<std::string> available_formats;
        available_formats.push_back(std::string("EVT2"));
        return available_formats;
    }
    virtual std::string get_current_data_encoding_format() const override {
        return "EVT2";
    }
    virtual std::string get_integrator() const override {
        return dummy_camera_integrator_name_;
    }
    virtual SystemInfo get_system_info() const override {
        return SystemInfo();
    }
    virtual std::string get_connection_type() const override {
        return std::string();
    }

    virtual RawFileHeader get_header_impl() const override {
        StreamFormat format(is_evt3 ? "EVT3" : "EVT2");
        format["width"]  = std::to_string(VGAGeometry().get_width());
        format["height"] = std::to_string(VGAGeometry().get_height());
        PseeRawFileHeader header(*this, format);
        header.set_system_id(system_id_);
        header.set_sub_system_id(dummy_sub_system_id_);
        header.set_field(dummy_custom_key_, dummy_custom_value_);
        return header;
    }

    virtual DeviceConfigOptionMap get_device_config_options_impl() const override {
        return {};
    }

    long system_id_;
    static const std::string dummy_serial_;
    static const std::string dummy_camera_integrator_name_;
    static const std::string dummy_custom_key_;
    static const std::string dummy_custom_value_;

    static constexpr bool is_evt3               = false;
    static constexpr long dummy_sub_system_id_  = 3;
    static constexpr long dummy_system_version_ = 0;
};

constexpr long MockHWIdentification::dummy_system_version_;
constexpr long MockHWIdentification::dummy_sub_system_id_;
constexpr bool MockHWIdentification::is_evt3;
const std::string MockHWIdentification::dummy_serial_                 = "dummy_serial";
const std::string MockHWIdentification::dummy_camera_integrator_name_ = "camera_integator_name";
const std::string MockHWIdentification::dummy_custom_key_             = "custom";
const std::string MockHWIdentification::dummy_custom_value_           = "field";

class MockRawDataProducer : public DataTransfer::RawDataProducer {
public:
    MockRawDataProducer() {}

private:
    void start_impl() final {}
    void run_impl(const DataTransfer &data_transfer) final {
        while (!data_transfer.should_stop()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
};

class I_EventsStream_GTest : public GTestWithTmpDir {
public:
    I_EventsStream_GTest(long system_id = metavision_device_traits<Gen3CDDevice>::SYSTEM_ID_DEFAULT) :
        system_id_(system_id) {
        reset();
    }

    virtual ~I_EventsStream_GTest() {}

    void reset() {
        // needed by MockHWIdentification::get_header
        auto plugin_sw_info = std::make_shared<I_PluginSoftwareInfo>(dummy_plugin_integrator_name_, dummy_plugin_name_,
                                                                     SoftwareInfo(0, 0, 0, "", "", "", ""));
        // needed by MockEventsStream::log_raw_data
        hw_identification_ = std::make_shared<MockHWIdentification>(plugin_sw_info, system_id_);

        DeviceBuilder device_builder = make_device_builder();
        auto decoder                 = device_builder.add_facility(std::make_unique<EVT2Decoder>(false));
        device_                      = device_builder();
        events_stream_ = std::make_shared<I_EventsStream>(std::make_unique<MockRawDataProducer>(), hw_identification_,
                                                          std::shared_ptr<I_EventsStreamDecoder>(decoder));
        events_stream_->start();
    }

    static const std::string dummy_plugin_name_;
    static const std::string dummy_plugin_integrator_name_;

    void transfer_data(DataTransfer::DefaultBufferPtr data) const {
        events_stream_->get_data_transfer().transfer_data(data);
    }

protected:
    virtual void SetUp() override {
        datasets_.push_back(
            (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw").string());
        datasets_.push_back(
            (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt2_hand.raw")
                .string());
        datasets_.push_back(
            (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt3_hand.raw")
                .string());
    }

    virtual void TearDown() override {}

    bool open_dataset(const std::string &dataset_name, RawFileConfig config = RawFileConfig()) {
        device_ = DeviceDiscovery::open_raw_file(dataset_name, config);
        return nullptr != device_.get();
    }

    // Needed facilities,
    std::unique_ptr<Device> device_;
    std::shared_ptr<MockHWIdentification> hw_identification_;
    std::shared_ptr<I_EventsStream> events_stream_;
    long system_id_;
    std::vector<std::string> datasets_;
};

const std::string I_EventsStream_GTest::dummy_plugin_name_            = "plugin_name";
const std::string I_EventsStream_GTest::dummy_plugin_integrator_name_ = "plugin_integator_name";
const std::initializer_list<DataTransfer::Data> list                  = {1, 2, 3, 4, 5};

TEST_F(I_EventsStream_GTest, add_sub_system_id_to_header) {
    // Create tmp file
    std::string filename = tmpdir_handler_->get_full_path("log.raw");
    events_stream_->log_raw_data(filename);
    events_stream_->stop_log_raw_data();

    // Now read the subid from the header and verify that it is the one set
    std::ifstream file(filename);
    auto header = PseeRawFileHeader(file);
    ASSERT_EQ(MockHWIdentification::dummy_sub_system_id_, header.get_sub_system_id());
    file.close();
}

TEST_F(I_EventsStream_GTest, poll_buffer) {
    ASSERT_EQ(0, events_stream_->poll_buffer());
    auto data = std::make_shared<DataTransfer::DefaultBufferType>(list);
    transfer_data(data);
    ASSERT_EQ(1, events_stream_->poll_buffer());
}

TEST_F(I_EventsStream_GTest, add_data_triggers_wait_next_buffer) {
    std::condition_variable wait_var;
    std::mutex triggered_mutex;
    bool triggered = false;
    auto thread    = std::thread([this, &wait_var, &triggered_mutex, &triggered]() {
        EXPECT_EQ(1, events_stream_->wait_next_buffer()); // waits for a buffer
        {
            std::unique_lock<std::mutex> wait_lock(triggered_mutex);
            triggered = true;
        }
        wait_var.notify_one();
    });

    while (!thread.joinable()) {} // wait thread to be started

    // trigger with add data
    auto data = std::make_shared<DataTransfer::DefaultBufferType>(list);
    transfer_data(data);

    // Wait for the trigger to be processed (add a timeout to not wait forever)
    auto now = std::chrono::system_clock::now();
    {
        std::unique_lock<std::mutex> wait_lock(triggered_mutex);
        wait_var.wait_until(wait_lock, now + std::chrono::seconds(1), [&triggered]() { return triggered; });
    }

    EXPECT_TRUE(
        triggered); // check that we did trigger the condition and that we did not wake up because of the timeout
    events_stream_->stop();
    thread.join();
}

TEST_F(I_EventsStream_GTest, stop_triggers_wait_next_buffer) {
    std::condition_variable wait_var;
    std::mutex triggered_mutex_;
    bool triggered = false;
    auto thread    = std::thread([this, &wait_var, &triggered_mutex_, &triggered]() {
        events_stream_->wait_next_buffer(); // waits for a trigger
        {
            std::unique_lock<std::mutex> wait_lock(triggered_mutex_);
            triggered = true;
        }
        wait_var.notify_one();
    });

    while (!thread.joinable()) {} // wait thread to be started

    // trigger with stop
    events_stream_->stop();

    // Wait for the trigger to be processed (add a timeout to not wait forever)
    auto now = std::chrono::system_clock::now();
    std::unique_lock<std::mutex> wait_lock(triggered_mutex_);
    wait_var.wait_until(wait_lock, now + std::chrono::seconds(1),
                        [&triggered]() { return triggered; }); // to avoid dead lock

    EXPECT_TRUE(
        triggered); // check that we did trigger the condition and that we did not wake up because of the timeout
    events_stream_->stop();
    thread.join();
}

TEST_F_WITH_DATASET(I_EventsStream_GTest, valid_index_file) {
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
        std::filesystem::remove(dataset + ".tmp_index");
        ASSERT_FALSE(std::filesystem::exists(dataset + ".tmp_index"));

        // Open the file and generate the index
        ASSERT_TRUE(open_dataset(dataset));

        constexpr uint32_t max_trials = 1000;
        uint32_t trials               = 1;
        // Ensure the index have been built
        auto fes = device_->get_facility<I_EventsStream>();
        ASSERT_NE(nullptr, fes);
        timestamp a, b;
        auto s = fes->get_seek_range(a, b);
        for (trials = 1; s != Metavision::I_EventsStream::IndexStatus::Good && trials < max_trials; ++trials) {
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
        index_file.seekg(-static_cast<int>(magic_number.size()), std::ios::end);
        std::array<char, magic_number.size()> buf;
        ASSERT_TRUE(index_file.read(buf.data(), magic_number.size()));
        for (size_t i = 0; i < magic_number.size(); ++i) {
            EXPECT_EQ(magic_number[i], buf[i]);
        }
    }
}

TEST_F_WITH_DATASET(I_EventsStream_GTest, invalid_index_file) {
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
        std::filesystem::remove(dataset + ".tmp_index");
        ASSERT_FALSE(std::filesystem::exists(dataset + ".tmp_index"));

        // Open the file and generate the index
        ASSERT_TRUE(open_dataset(dataset));

        constexpr uint32_t max_trials = 1000;
        uint32_t trials               = 1;
        // Ensure the index have been built
        auto fes = device_->get_facility<I_EventsStream>();
        ASSERT_NE(nullptr, fes);
        timestamp a, b;
        auto s = fes->get_seek_range(a, b);
        for (trials = 1; s != Metavision::I_EventsStream::IndexStatus::Good && trials < max_trials; ++trials) {
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
            auto fes = device_->get_facility<I_EventsStream>();
            ASSERT_NE(nullptr, fes);
            timestamp a, b;
            auto s = fes->get_seek_range(a, b);
            for (trials = 1; s != Metavision::I_EventsStream::IndexStatus::Good && trials < max_trials; ++trials) {
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
            auto fes = device_->get_facility<I_EventsStream>();
            ASSERT_NE(nullptr, fes);
            timestamp a, b;
            auto s = fes->get_seek_range(a, b);
            for (trials = 1; s != Metavision::I_EventsStream::IndexStatus::Good && trials < max_trials; ++trials) {
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

TEST_F_WITH_DATASET(I_EventsStream_GTest, seek_range) {
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
                std::filesystem::remove(dataset + ".tmp_index");
                ASSERT_FALSE(std::filesystem::exists(dataset + ".tmp_index"));
            }

            MV_HAL_LOG_INFO() << "\tTesting dataset" << dataset;
            for (const auto &do_time_shifting : time_shift) {
                // Builds the device from the dataset
                MV_HAL_LOG_INFO() << "\t\tTime shift:" << (do_time_shifting ? "enabled" : "disabled");
                RawFileConfig config;
                config.do_time_shifting_ = do_time_shifting;
                ASSERT_TRUE(open_dataset(dataset, config));

                // Ensures the index have been built and one can retrieve the timestamp range
                auto fes = device_->get_facility<I_EventsStream>();
                ASSERT_NE(nullptr, fes);

                timestamp first_indexed_event_ts_us, last_indexed_event_ts_us;
                auto index_status = fes->get_seek_range(first_indexed_event_ts_us, last_indexed_event_ts_us);
                constexpr uint32_t max_trials = 1000;
                uint32_t trials               = 1;
                while (index_status != Metavision::I_EventsStream::IndexStatus::Good && trials != max_trials) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    ++trials;
                    index_status = fes->get_seek_range(first_indexed_event_ts_us, last_indexed_event_ts_us);
                }

                ASSERT_LT(trials, max_trials);
                EXPECT_GT(last_indexed_event_ts_us, first_indexed_event_ts_us);

                auto decoder = device_->get_facility<I_EventsStreamDecoder>();
                ASSERT_NE(nullptr, decoder);
                EXPECT_EQ(do_time_shifting, decoder->is_time_shifting_enabled());

                // Decode the full dataset
                // -- First compute the first event timestamp using the decoder
                fes->start();
                timestamp first_decoded_event_ts_us = decoder->get_last_timestamp();
                EXPECT_EQ(timestamp(-1), first_decoded_event_ts_us);
                timestamp last_decoded_event_ts_us;
                while (fes->wait_next_buffer() > 0) {
                    auto buffer    = fes->get_latest_raw_data();
                    auto data      = buffer.data();
                    auto data_next = data + decoder->get_raw_event_size_bytes();
                    auto data_end  = buffer.end();

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
                    auto buffer = fes->get_latest_raw_data();
                    decoder->decode(buffer);
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

TEST_F_WITH_DATASET(I_EventsStream_GTest, file_index_seek) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Check seeking feature of the file control

    RawFileConfig config;
    config.n_events_to_read_ = 10000;

    std::vector<bool> delete_index{{true, false}};
    std::vector<bool> time_shift{{true, false}};

    for (const auto &do_delete : delete_index) {
        MV_HAL_LOG_INFO() << (do_delete ? "Index building from scratch" : "Index loaded from file");
        for (const auto &dataset : datasets_) {
            if (do_delete) {
                std::string path;
                std::filesystem::remove(dataset + ".tmp_index");
                ASSERT_FALSE(std::filesystem::exists(dataset + ".tmp_index"));
            }

            MV_HAL_LOG_INFO() << "\tTesting dataset" << dataset;
            for (const auto &do_time_shifting : time_shift) {
                // Builds the device from the dataset
                MV_HAL_LOG_INFO() << "\t\tTime shift:" << (do_time_shifting ? "enabled" : "disabled");
                config.do_time_shifting_ = do_time_shifting;
                ASSERT_TRUE(open_dataset(dataset, config));

                // Ensures the index have been built and one can retrieve the timestamp range
                auto fes = device_->get_facility<I_EventsStream>();
                ASSERT_NE(nullptr, fes);

                timestamp first_indexed_event_ts_us, last_indexed_event_ts_us;
                auto index_status = fes->get_seek_range(first_indexed_event_ts_us, last_indexed_event_ts_us);
                constexpr uint32_t max_trials = 1000;
                uint32_t trials               = 1;
                while (index_status != Metavision::I_EventsStream::IndexStatus::Good && trials != max_trials) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    ++trials;
                    index_status = fes->get_seek_range(first_indexed_event_ts_us, last_indexed_event_ts_us);
                }

                ASSERT_LT(trials, max_trials);
                ASSERT_GT(last_indexed_event_ts_us, first_indexed_event_ts_us);

                auto decoder    = device_->get_facility<I_EventsStreamDecoder>();
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
                cd_decoder->add_event_buffer_callback([&](auto, auto) { valid_event = true; });
                while (!valid_event) {
                    // Read data from the file
                    ASSERT_TRUE(fes->wait_next_buffer() > 0);
                    auto buffer = fes->get_latest_raw_data();
                    decoder->decode(buffer);
                }

                // ------------------------------
                // Check seeking at the beginning
                MV_HAL_LOG_INFO() << "\t\t\tSeek first event";
                timestamp reached_ts;
                ASSERT_EQ(I_EventsStream::SeekStatus::Success, fes->seek(first_indexed_event_ts_us, reached_ts));
                ASSERT_EQ(first_indexed_event_ts_us, reached_ts);
                decoder->reset_last_timestamp(reached_ts);
                ASSERT_EQ(decoder->get_last_timestamp(), reached_ts);

                ASSERT_TRUE(fes->wait_next_buffer() > 0);
                // Decode a single event and check that the timestamp is correct
                auto buffer = fes->get_latest_raw_data();

                // The first event decoded must have a timestamp that is equal to the first event's timestamp
                decoder->decode(buffer.data(), buffer.data() + decoder->get_raw_event_size_bytes());
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
                    ASSERT_EQ(I_EventsStream::SeekStatus::Success, fes->seek(target_ts_us, reached_ts));
                    decoder->reset_last_timestamp(reached_ts);
                    ASSERT_LE(reached_ts, target_ts_us);

                    // Read data from the file
                    ASSERT_TRUE(fes->wait_next_buffer() > 0);
                    buffer = fes->get_latest_raw_data();

                    // The first event decoded must have a timestamp that is equal to the reached timestamp
                    decoder->decode(buffer.data(), buffer.data() + decoder->get_raw_event_size_bytes());
                    ASSERT_EQ(decoder->get_last_timestamp(), reached_ts);
                }

                // ------------------------------
                // Check seeking at the end of the file
                MV_HAL_LOG_INFO() << "\t\t\tSeek last event";
                ASSERT_EQ(I_EventsStream::SeekStatus::Success, fes->seek(last_indexed_event_ts_us, reached_ts));
                ASSERT_LE(reached_ts, last_indexed_event_ts_us);
                decoder->reset_last_timestamp(reached_ts);

                // Read data from the file
                ASSERT_TRUE(fes->wait_next_buffer() > 0);
                buffer = fes->get_latest_raw_data();

                // The first event decoded must have a timestamp that is equal to the reached timestamp
                decoder->decode(buffer.data(), buffer.data() + decoder->get_raw_event_size_bytes());
                ASSERT_EQ(decoder->get_last_timestamp(), reached_ts);
            }
        }
    }
}

TEST_F_WITH_DATASET(I_EventsStream_GTest, decode_evt3_nevents_monotonous_timestamps) {
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

TEST_WITH_DATASET(EventsStream_GTest, stop_on_recording_does_not_drop_buffers) {
    // Read the dataset provided
    std::string dataset_file_path =
        (std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt3_hand.raw").string();

    std::unique_ptr<Metavision::Device> device = Metavision::DeviceDiscovery::open_raw_file(dataset_file_path);
    EXPECT_TRUE(device != nullptr);

    long int n_raw                             = 0;
    Metavision::I_EventsStream *i_eventsstream = device->get_facility<Metavision::I_EventsStream>();
    while (true) {
        i_eventsstream->start();
        // allow reading thread to accumulate some buffers that could be dropped when we stop the events stream
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (i_eventsstream->wait_next_buffer() < 0) {
            break;
        }
        auto ev_buffer = i_eventsstream->get_latest_raw_data();
        n_raw += ev_buffer.size();
        i_eventsstream->stop();
    }

    ASSERT_EQ(96786804, n_raw);
}

template<class Device>
class I_EventsStreamT_GTest : public I_EventsStream_GTest {
public:
    I_EventsStreamT_GTest() : I_EventsStream_GTest(metavision_device_traits<Device>::SYSTEM_ID_DEFAULT) {
        build_events();
    }

    virtual ~I_EventsStreamT_GTest() {}

protected:
    void build_events();

    I_EventsStreamDecoder *create_decoder();

    virtual void SetUp() override {
        I_EventsStream_GTest::SetUp();
    }

    virtual void TearDown() override {
        I_EventsStream_GTest::TearDown();
    }

    std::vector<EventCD> events1_, events2_;
};

template<>
I_EventsStreamDecoder *I_EventsStreamT_GTest<Gen3CDDevice>::create_decoder() {
    static constexpr bool TimeShiftingEnabled = false;

    DeviceBuilder device_builder = make_device_builder();

    auto cd_event_decoder          = device_builder.add_facility(std::make_unique<I_EventDecoder<EventCD>>());
    auto ext_trigger_event_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventExtTrigger>>());
    auto decoder                   = device_builder.add_facility(
        std::make_unique<EVT2Decoder>(TimeShiftingEnabled, cd_event_decoder, ext_trigger_event_decoder));
    device_ = device_builder();

    return decoder.get();
}

template<>
void I_EventsStreamT_GTest<Gen3CDDevice>::build_events() {
    events1_ = {
        {16, 345, 1, 1642},   {3, 360, 0, 3292},    {365, 61, 1, 4977},   {119, 44, 0, 6631},   {24, 349, 1, 8258},
        {41, 339, 1, 9908},   {132, 52, 0, 11577},  {373, 106, 1, 13210}, {2, 334, 1, 14842},   {8, 379, 1, 16516},
        {329, 89, 1, 18179},  {45, 380, 1, 19810},  {22, 350, 0, 21510},  {329, 93, 1, 23207},  {67, 249, 0, 24944},
        {11, 240, 1, 26619},  {12, 353, 1, 28315},  {35, 422, 1, 30047},  {256, 117, 1, 31821}, {14, 311, 0, 33520},
        {128, 66, 1, 35253},  {44, 284, 0, 37005},  {72, 248, 0, 38783},  {9, 369, 1, 40490},   {30, 323, 1, 42220},
        {16, 325, 1, 43976},  {45, 321, 1, 45754},  {26, 322, 1, 47477},  {30, 317, 1, 49236},  {116, 57, 1, 50977},
        {249, 47, 1, 52740},  {122, 47, 0, 54440},  {12, 373, 1, 56164},  {43, 332, 0, 57866},  {38, 369, 1, 59574},
        {2, 328, 0, 61313},   {16, 355, 1, 63035},  {32, 308, 1, 64797},  {296, 109, 0, 66553}, {40, 308, 1, 68284},
        {358, 54, 1, 70002},  {287, 47, 1, 71769},  {326, 28, 1, 73513},  {68, 223, 0, 75246},  {34, 304, 1, 77010},
        {359, 53, 1, 78756},  {72, 222, 0, 80496},  {1, 326, 0, 82262},   {284, 105, 0, 84020}, {123, 36, 1, 85803},
        {142, 54, 1, 87545},  {12, 278, 0, 89275},  {287, 112, 1, 91028}, {278, 102, 0, 92819}, {11, 334, 1, 94596},
        {40, 287, 1, 96320},  {120, 32, 1, 98113},  {78, 223, 0, 99845},  {53, 336, 1, 101530}, {345, 47, 0, 103286},
        {35, 295, 0, 105032}, {8, 321, 1, 106761},  {254, 36, 1, 108485}, {23, 281, 1, 110214}, {36, 296, 0, 111941},
        {4, 279, 1, 113679},  {239, 51, 1, 115390}, {50, 251, 0, 117156}};
}

typedef ::testing::Types<Gen3CDDevice> TestingTypes;

TYPED_TEST_CASE(I_EventsStreamT_GTest, TestingTypes);

TYPED_TEST(I_EventsStreamT_GTest, test_log) {
    // Get tmp file name
    std::string filename(this->tmpdir_handler_->get_full_path("log.raw"));

    // Firmware, serial, system_id and data are added through the
    // common events stream automatically
    this->events_stream_->log_raw_data(filename);

    using EvtFormat = typename metavision_device_traits<TypeParam>::RawEventFormat;

    TEncoder<EvtFormat> encoder;
    auto data = std::make_shared<DataTransfer::DefaultBufferType>();

    encoder.set_encode_event_callback(
        [&data](const uint8_t *ev, const uint8_t *ev_end) { data->insert(data->end(), ev, ev_end); });

    // Encode
    encoder.encode(this->events1_.data(), this->events1_.data() + this->events1_.size());
    encoder.flush();
    this->events_stream_->get_latest_raw_data();
    this->transfer_data(data);

    // REMARK : as of today, in order to log we have to call
    // get_latest_raw_data before add_data and after
    // TODO : remove this following line if we modify the behaviour
    // of log_data (for example if we log when calling add_data or if
    // we log in a separate thread)
    this->events_stream_->get_latest_raw_data();
    this->events_stream_->stop_log_raw_data();

    // Now open the file and verify what is written :
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    ASSERT_TRUE(file.is_open());

    // Get the header
    PseeRawFileHeader header_read(file);
    ASSERT_FALSE(header_read.empty());
    ASSERT_EQ(MockHWIdentification::dummy_serial_, header_read.get_serial());
    ASSERT_EQ(MockHWIdentification::is_evt3 ? "EVT3" : "EVT2", header_read.get_format().name());
    ASSERT_EQ(this->system_id_, header_read.get_system_id());
    ASSERT_EQ(MockHWIdentification::dummy_sub_system_id_, header_read.get_sub_system_id());
    ASSERT_EQ(MockHWIdentification::dummy_camera_integrator_name_, header_read.get_camera_integrator_name());
    ASSERT_EQ(I_EventsStream_GTest::dummy_plugin_integrator_name_, header_read.get_plugin_integrator_name());
    ASSERT_EQ(I_EventsStream_GTest::dummy_plugin_name_, header_read.get_plugin_name());
    ASSERT_EQ(MockHWIdentification::dummy_custom_value_,
              header_read.get_field(MockHWIdentification::dummy_custom_key_));

    // Read the file

    // Check length
    auto position_first_data = file.tellg();                // Get the current position in the file
    file.seekg(0, std::ios::end);                           // Go to the end of the file
    auto n_tot_data = (file.tellg() - position_first_data); // Compute the number of data
    file.seekg(position_first_data);                        // Reset the position at the
                                                            // first event
    char *buffer = new char[n_tot_data];

    // Read data as a block:
    file.read(buffer, n_tot_data);
    ASSERT_EQ(n_tot_data, file.gcount());

    // Verify we have read everything
    file.read(buffer, n_tot_data);
    ASSERT_EQ(0, file.gcount());
    ASSERT_TRUE(file.eof());
    file.close();

    // Decode the buffer received :
    uint8_t *buffer_to_decode = reinterpret_cast<uint8_t *>(buffer);

    // Decoder
    auto decoder = this->create_decoder();

    std::vector<EventCD> decoded_events;
    auto td_decoder = this->device_->template get_facility<I_EventDecoder<EventCD>>();
    td_decoder->add_event_buffer_callback([&decoded_events](const EventCD *begin, const EventCD *end) {
        decoded_events.insert(decoded_events.end(), begin, end);
    });

    decoder->decode(buffer_to_decode, buffer_to_decode + n_tot_data);

    ASSERT_EQ(this->events1_.size(), decoded_events.size());
    auto it_expected = this->events1_.begin();
    auto it          = decoded_events.begin();

    using SizeType = std::vector<EventCD>::size_type;
    for (SizeType i = 0, max_i = this->events1_.size(); i < max_i; ++i, ++it, ++it_expected) {
        EXPECT_EQ(it_expected->x, it->x);
        EXPECT_EQ(it_expected->y, it->y);
        EXPECT_EQ(it_expected->p, it->p);
        EXPECT_EQ(it_expected->t, it->t);
    }

    delete[] buffer;
}
