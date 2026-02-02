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
#include <atomic>
#include <memory>
#include <numeric>

#include "metavision/hal/utils/data_transfer.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/file_raw_data_producer.h"
#include "metavision/utils/gtest/gtest_with_tmp_dir.h"

using namespace Metavision;

class FileRawDataProducer_Gtest : public GTestWithTmpDir {
protected:
    virtual void SetUp() override {
        // Create and open the rawfile
        static int raw_counter = 1;
        rawfile_to_log_path_   = tmpdir_handler_->get_full_path("rawfile_" + std::to_string(++raw_counter) + ".raw");
        rawfile_to_log_from_rawfile_path_ = rawfile_to_log_path_ + ".logged";
    }

    virtual void TearDown() override {
        close_raw();
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

    std::vector<uint8_t> write_ref_data() {
        open_raw();
        uint32_t n_buffers_count = 0;
        auto data_to_write       = std::vector<uint8_t>(bytes_per_written_buffer_default_);
        std::iota(data_to_write.begin(), data_to_write.end(), 1);

        std::vector<uint8_t> data_ref;

        const size_t expected_data_size = n_buffers_default_ * bytes_per_written_buffer_default_;

        // writes data
        while (++n_buffers_count <= n_buffers_default_) {
            // write data
            data_ref.insert(data_ref.end(), data_to_write.begin(), data_to_write.end());
            rawfile_to_log_->write(reinterpret_cast<char *>(data_to_write.data()), data_to_write.size());
        }

        EXPECT_EQ(data_ref.size(), expected_data_size);

        close_raw();
        return data_ref;
    }

    bool open_file_data_transfer(uint32_t raw_events_per_read = raw_events_per_read_default_,
                                 uint32_t read_buffers_count  = read_buffers_count_) {
        auto ifs = std::make_unique<std::ifstream>(rawfile_to_log_path_, std::ios::binary);
        RawFileConfig config;
        config.n_events_to_read_ = raw_events_per_read;
        config.n_read_buffers_   = read_buffers_count;

        file_raw_data_producer_ = std::make_shared<FileRawDataProducer>(std::move(ifs), raw_event_size_bytes_, config);
        file_data_transfer_     = std::make_unique<DataTransfer>(file_raw_data_producer_);
        if (!file_data_transfer_ || !file_data_transfer_) {
            file_data_transfer_.reset(nullptr);
            return false;
        }

        return true;
    }

public:
    static constexpr uint32_t raw_event_size_bytes_             = 4;
    static constexpr uint32_t bytes_per_written_buffer_default_ = 100, n_buffers_default_ = 5;
    static constexpr uint32_t raw_events_per_read_default_ = 23;
    static constexpr uint32_t read_buffers_count_          = 3;

protected:
    std::unique_ptr<DataTransfer> file_data_transfer_;
    std::shared_ptr<DataTransfer::RawDataProducer> file_raw_data_producer_;

    std::unique_ptr<std::ofstream> rawfile_to_log_;
    std::string rawfile_to_log_path_;
    std::string rawfile_to_log_from_rawfile_path_;
};

constexpr uint32_t FileRawDataProducer_Gtest::raw_event_size_bytes_;
constexpr uint32_t FileRawDataProducer_Gtest::read_buffers_count_;
constexpr uint32_t FileRawDataProducer_Gtest::bytes_per_written_buffer_default_;
constexpr uint32_t FileRawDataProducer_Gtest::n_buffers_default_;
constexpr uint32_t FileRawDataProducer_Gtest::raw_events_per_read_default_;

TEST_F(FileRawDataProducer_Gtest, file_does_not_exists) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Check that you can't read a file that doesn't exist

    ASSERT_EQ(nullptr, file_data_transfer_.get());
}

TEST_F(FileRawDataProducer_Gtest, reading_integrity) {
    // GIVEN a RAW file with known content
    auto data_ref = write_ref_data();

    // WHEN opening the data transfer to read the file
    ASSERT_TRUE(open_file_data_transfer());

    // AND WHEN copying the read data in a buffer
    std::vector<DataTransfer::Data> data_read;

    file_data_transfer_->add_new_buffer_callback(
        [&](auto &buffer) { data_read.insert(data_read.end(), buffer.cbegin(), buffer.cend()); });

    // AND WHEN setting a callback on stop
    std::atomic<bool> stopped{false};
    file_data_transfer_->add_status_changed_callback(
        [&](auto status) { stopped = status == DataTransfer::Status::Stopped; });

    file_data_transfer_->start();
    while (!stopped) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // AND THEN the data read must be the same
    ASSERT_EQ(data_ref, data_read);
}

TEST_F(FileRawDataProducer_Gtest, memory_usage) {
    // GIVEN a RAW file with known content
    auto data_ref = write_ref_data();

    // WHEN opening the data transfer to read the file
    ASSERT_TRUE(open_file_data_transfer());

    // AND WHEN keeping the buffers in memory
    std::vector<DataTransfer::Data> data_read;
    std::vector<DataTransfer::BufferPtr> buffers;
    std::mutex buffers_safety;

    file_data_transfer_->add_new_buffer_callback([&](auto &buffer) {
        std::lock_guard<std::mutex> lock(buffers_safety);
        buffers.push_back(buffer);
        // THEN each buffer contains at most the requested RAW events to read count in bytes
        ASSERT_LE(buffer.size(), raw_events_per_read_default_ * raw_event_size_bytes_);
    });

    // AND WHEN setting a callback on stop
    std::atomic<bool> stopped{false};
    file_data_transfer_->add_status_changed_callback(
        [&](auto status) { stopped = status == DataTransfer::Status::Stopped; });

    file_data_transfer_->start();

    // AND THEN the number of buffers transferred can not be greater than the maximum requested (read_buffers_count_)
    constexpr uint32_t max_trials = 10;
    uint32_t trials               = 0;
    while (!stopped) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // Here we expect the RAW file to be read fully in less than 100ms.
        // If more buffer than the number expected are transferred, it should occur in this time interval.
        // If everything goes as expected, the number of buffers is always less or equal to the expected count.
        std::lock_guard<std::mutex> lock(buffers_safety);
        ASSERT_LE(buffers.size(), read_buffers_count_);
        ++trials;
        if (trials >= max_trials) {
            for (auto &buffer : buffers) {
                data_read.insert(data_read.end(), buffer.cbegin(), buffer.cend());
            }
            // release the buffers so that the object pool can reuse them
            buffers.clear();
        }
    }

    // Data read
    ASSERT_EQ(data_ref, data_read);
}

TEST_F(FileRawDataProducer_Gtest, invalid_parameters) {
    // GIVEN a RAW file with known content
    auto data_ref = write_ref_data();

    // WHEN opening the data transfer to read the file with invalid parameters
    // RAW events per read = 0
    ASSERT_THROW(open_file_data_transfer(0), HalException);
}

TEST_F(FileRawDataProducer_Gtest, remove_calback) {
    // GIVEN a RAW file with known content
    auto data_ref = write_ref_data();

    // WHEN opening the data transfer to read the file
    ASSERT_TRUE(open_file_data_transfer());

    bool called = false;
    auto id     = file_data_transfer_->add_status_changed_callback([&](auto status) { called = true; });

    std::atomic<bool> stopped{false};
    file_data_transfer_->add_status_changed_callback(
        [&](auto status) { stopped = status == DataTransfer::Status::Stopped; });

    // remove the callback
    file_data_transfer_->remove_callback(id);

    file_data_transfer_->start();
    while (!stopped) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_FALSE(called);
}
