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
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <gtest/gtest.h>
#include <H5Cpp.h>
#include <hdf5_ecf/ecf_codec.h>
#include <hdf5_ecf/ecf_h5filter.h>

#include "metavision/hal/device/device.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/sdk/base/utils/generic_header.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/ext_trigger.h"
#include "metavision/sdk/stream/dat_event_file_reader.h"
#include "metavision/sdk/stream/raw_event_file_reader.h"
#include "metavision/sdk/stream/raw_event_file_logger.h"
#include "metavision/sdk/stream/hdf5_event_file_reader.h"
#include "metavision/sdk/stream/hdf5_event_file_writer.h"
#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/utils/gtest/gtest_custom.h"

using namespace Metavision;

class HDF5EventFileReader_Gtest : public GTestWithTmpDir {
protected:
    virtual void SetUp() {
        tmp_file_ = tmpdir_handler_->get_full_path("test.hdf5");
    }

    std::filesystem::path tmp_file_;
};

TEST_F_WITH_DATASET(HDF5EventFileReader_Gtest, constructor_invalid) {
    std::string dataset_file_path = "invalid_path.blub";
    std::unique_ptr<HDF5EventFileReader> reader;
    ASSERT_THROW(reader = std::make_unique<HDF5EventFileReader>(dataset_file_path), H5::FileIException);
}

TEST_F_WITH_DATASET(HDF5EventFileReader_Gtest, constructor_valid) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.hdf5";
    std::unique_ptr<HDF5EventFileReader> reader;
    ASSERT_NO_THROW(reader = std::make_unique<HDF5EventFileReader>(dataset_file_path));
}

TEST_F_WITH_DATASET(HDF5EventFileReader_Gtest, get_path) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.hdf5";
    HDF5EventFileReader reader(dataset_file_path);
    ASSERT_EQ(dataset_file_path, reader.get_path());
}

TEST_F_WITH_DATASET(HDF5EventFileReader_Gtest, has_read_callbacks) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.hdf5";
    HDF5EventFileReader reader(dataset_file_path);
    ASSERT_FALSE(reader.has_read_callbacks());
    size_t id = reader.add_read_callback([](const EventCD *, const EventCD *) {});
    ASSERT_TRUE(reader.has_read_callbacks());
    reader.remove_callback(id);
    ASSERT_FALSE(reader.has_read_callbacks());
    reader.add_read_callback([](const EventExtTrigger *, const EventExtTrigger *) {});
    ASSERT_TRUE(reader.has_read_callbacks());
}

TEST_F_WITH_DATASET(HDF5EventFileReader_Gtest, get_metadata_map) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.hdf5";
    HDF5EventFileReader reader(dataset_file_path);
    auto metadata_map = reader.get_metadata_map();
    ASSERT_FALSE(metadata_map.empty());
    auto it = metadata_map.find("integrator_name");
    ASSERT_TRUE(it != metadata_map.end());
    EXPECT_EQ("Prophesee", it->second);
    it = metadata_map.find("serial_number");
    ASSERT_TRUE(it != metadata_map.end());
    EXPECT_EQ("00001621", it->second);
    it = metadata_map.find("generation");
    ASSERT_TRUE(it != metadata_map.end());
    EXPECT_EQ("3.1", it->second);
    it = metadata_map.find("geometry");
    ASSERT_TRUE(it != metadata_map.end());
    EXPECT_EQ("640x480", it->second);
    it = metadata_map.find("version");
    ASSERT_TRUE(it != metadata_map.end());
    EXPECT_EQ("1.0", it->second);
}

TEST_F_WITH_DATASET(HDF5EventFileReader_Gtest, has_seek_callbacks) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.hdf5";
    HDF5EventFileReader reader(dataset_file_path);
    ASSERT_FALSE(reader.has_seek_callbacks());
    size_t id = reader.add_seek_callback([](const auto &) {});
    ASSERT_TRUE(reader.has_seek_callbacks());
    reader.remove_callback(id);
    ASSERT_FALSE(reader.has_seek_callbacks());
}

TEST_F_WITH_DATASET(HDF5EventFileReader_Gtest, get_seek_range) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.hdf5";
    HDF5EventFileReader reader(dataset_file_path);
    timestamp min_t, max_t;
    ASSERT_TRUE(reader.get_seek_range(min_t, max_t));
    EXPECT_EQ(16, min_t);
    EXPECT_EQ(13042000, max_t);
}

TEST_F_WITH_DATASET(HDF5EventFileReader_Gtest, get_duration) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.hdf5";
    HDF5EventFileReader reader(dataset_file_path);
    EXPECT_EQ(13043033, reader.get_duration());
}

TEST_F_WITH_DATASET(HDF5EventFileReader_Gtest, read) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "blinking_gen4_with_ext_triggers.hdf5";
    HDF5EventFileReader reader(dataset_file_path);
    size_t num_cd_events = 0;
    reader.add_read_callback(
        [&num_cd_events](const EventCD *begin, const EventCD *end) { num_cd_events += std::distance(begin, end); });
    size_t num_ext_trigger_events = 0;
    reader.add_read_callback([&num_ext_trigger_events](const EventExtTrigger *begin, const EventExtTrigger *end) {
        num_ext_trigger_events += std::distance(begin, end);
    });
    EXPECT_EQ(0, num_cd_events);
    EXPECT_EQ(0, num_ext_trigger_events);
    while (reader.read()) {
        std::this_thread::yield();
    }
    EXPECT_EQ(2003016, num_cd_events);
    EXPECT_EQ(82, num_ext_trigger_events);
}

TEST_F_WITH_DATASET(HDF5EventFileReader_Gtest, seek) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.hdf5";
    HDF5EventFileReader reader(dataset_file_path);
    timestamp ts = -1;
    reader.add_seek_callback([&ts](const timestamp &t) { ts = t; });
    EXPECT_EQ(-1, ts);
    EXPECT_TRUE(reader.read());
    timestamp min_t, max_t;
    while (!reader.get_seek_range(min_t, max_t)) {
        std::this_thread::yield();
    }
    EXPECT_TRUE(reader.seek(134567));
    EXPECT_EQ(132000, ts);
}

TEST_F(HDF5EventFileReader_Gtest, read_cd_events_written_without_direct_calls) {
    size_t num_expected_events = 2753; // not a multiple of chunk size
    std::vector<Metavision::EventCD> expected_events(num_expected_events);
    std::mt19937 mt_rand; // Mersenne twister
    mt_rand.seed(42);

    std::uniform_int_distribution<int> dt(0, 333827);
    std::vector<timestamp> times(num_expected_events);
    for (size_t i = 0; i < num_expected_events; ++i) {
        times[i] = dt(mt_rand);
    }
    std::uniform_int_distribution<int> dx(0, 1000), dy(0, 800), dp(0, 1);
    std::sort(times.begin(), times.end());
    for (size_t i = 0; i < num_expected_events; ++i) {
        expected_events[i].x = dx(mt_rand);
        expected_events[i].y = dy(mt_rand);
        expected_events[i].t = times[i];
        expected_events[i].p = dp(mt_rand);
    }

    struct Index {
        Index(size_t id = 0, std::int64_t ts = 0) : id(id), ts(ts) {}
        size_t id;
        std::int64_t ts;
    };

    {
        // write HDF5 file using chunking and filtering through HDF5 API (indirect) calls
        // and the ECF plugin
        auto file = H5::H5File(tmp_file_.string(), H5F_ACC_TRUNC);
        file.createGroup("/CD");
        file.createGroup("/EXT_TRIGGER");
        if (ecf_register_h5filter() < 0) {
            FAIL() << "Unable to register ECF Filter";
        }

        hsize_t dims[1]       = {num_expected_events};
        hsize_t chunk_dims[1] = {1024};

        H5::CompType cd_event_dt(sizeof(Metavision::EventCD));
        cd_event_dt.insertMember("x", HOFFSET(Metavision::EventCD, x), H5::PredType::NATIVE_USHORT);
        cd_event_dt.insertMember("y", HOFFSET(Metavision::EventCD, y), H5::PredType::NATIVE_USHORT);
        cd_event_dt.insertMember("p", HOFFSET(Metavision::EventCD, p), H5::PredType::NATIVE_SHORT);
        cd_event_dt.insertMember("t", HOFFSET(Metavision::EventCD, t), H5::PredType::NATIVE_LLONG);
        H5::DataSpace cd_event_ds(1, dims);
        H5::DSetCreatPropList cd_event_ds_prop;
        cd_event_ds_prop.setChunk(1, chunk_dims);
        cd_event_ds_prop.setFilter(H5Z_FILTER_ECF, H5Z_FLAG_OPTIONAL, 0, nullptr);
        H5::DataSet cd_events_dset = file.createDataSet("/CD/events", cd_event_dt, cd_event_ds, cd_event_ds_prop);
        cd_events_dset.write(&expected_events[0], cd_event_dt);

        // create mandatory datasets, even though we don't test their contents
        H5::CompType cd_index_dt(sizeof(Index));
        cd_index_dt.insertMember("id", HOFFSET(Index, id), H5::PredType::NATIVE_ULLONG);
        cd_index_dt.insertMember("ts", HOFFSET(Index, ts), H5::PredType::NATIVE_LLONG);
        dims[0] = 0;
        H5::DataSpace cd_index_ds(1, dims);
        H5::DSetCreatPropList cd_index_ds_prop;
        cd_index_ds_prop.setChunk(1, chunk_dims);
        file.createDataSet("/CD/indexes", cd_index_dt, cd_index_ds, cd_index_ds_prop);

        H5::DataSpace ext_trigger_event_ds(1, dims);
        H5::CompType ext_trigger_event_dt(sizeof(Metavision::EventExtTrigger));
        ext_trigger_event_dt.insertMember("p", HOFFSET(Metavision::EventExtTrigger, p), H5::PredType::NATIVE_SHORT);
        ext_trigger_event_dt.insertMember("t", HOFFSET(Metavision::EventExtTrigger, t), H5::PredType::NATIVE_LLONG);
        ext_trigger_event_dt.insertMember("id", HOFFSET(Metavision::EventExtTrigger, id), H5::PredType::NATIVE_SHORT);
        H5::DSetCreatPropList ext_trigger_event_ds_prop;
        ext_trigger_event_ds_prop.setChunk(1, chunk_dims);

        H5::CompType ext_trigger_index_dt(sizeof(Index));
        ext_trigger_index_dt.insertMember("id", HOFFSET(Index, id), H5::PredType::NATIVE_ULLONG);
        ext_trigger_index_dt.insertMember("ts", HOFFSET(Index, ts), H5::PredType::NATIVE_LLONG);
        H5::DataSpace ext_trigger_index_ds(1, dims);
        H5::DSetCreatPropList ext_trigger_index_ds_prop;
        ext_trigger_index_ds_prop.setChunk(1, chunk_dims);

        file.createDataSet("/EXT_TRIGGER/events", ext_trigger_event_dt, ext_trigger_event_ds,
                           ext_trigger_event_ds_prop);
        file.createDataSet("/EXT_TRIGGER/indexes", ext_trigger_index_dt, ext_trigger_index_ds,
                           ext_trigger_index_ds_prop);
    }
    {
        // read it back and make sure data is correct
        HDF5EventFileReader reader(tmp_file_);
        std::vector<Metavision::EventCD> events;
        reader.add_read_callback(
            [&events](const EventCD *begin, const EventCD *end) { events.insert(events.end(), begin, end); });
        EXPECT_TRUE(events.empty());
        while (reader.read()) {
            std::this_thread::yield();
        }
        ASSERT_EQ(num_expected_events, events.size());
        for (size_t i = 0; i < num_expected_events; ++i) {
            ASSERT_EQ(expected_events[i].x, events[i].x);
            ASSERT_EQ(expected_events[i].y, events[i].y);
            ASSERT_EQ(expected_events[i].p, events[i].p);
            ASSERT_EQ(expected_events[i].t, events[i].t);
        }
    }
}

TEST_WITH_DATASET(RAWEventFileReader_Gtest, constructor_valid) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
    std::unique_ptr<Device> device = DeviceDiscovery::open_raw_file(dataset_file_path);
    std::unique_ptr<RAWEventFileReader> reader;
    ASSERT_NO_THROW(reader = std::make_unique<RAWEventFileReader>(*device, dataset_file_path));
}

TEST_WITH_DATASET(RAWEventFileReader_Gtest, get_path) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
    std::unique_ptr<Device> device = DeviceDiscovery::open_raw_file(dataset_file_path);
    RAWEventFileReader reader(*device, dataset_file_path);
    ASSERT_EQ(dataset_file_path, reader.get_path());
}

TEST_WITH_DATASET(RAWEventFileReader_Gtest, has_read_callbacks) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
    std::unique_ptr<Device> device = DeviceDiscovery::open_raw_file(dataset_file_path);
    RAWEventFileReader reader(*device, dataset_file_path);
    ASSERT_FALSE(reader.has_read_callbacks());
    size_t id = reader.add_read_callback([](const EventCD *, const EventCD *) {});
    ASSERT_TRUE(reader.has_read_callbacks());
    reader.remove_callback(id);
    ASSERT_FALSE(reader.has_read_callbacks());
    reader.add_read_callback([](const EventExtTrigger *, const EventExtTrigger *) {});
    ASSERT_TRUE(reader.has_read_callbacks());
}

TEST_WITH_DATASET(RAWEventFileReader_Gtest, get_metadata_map) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
    std::unique_ptr<Device> device = DeviceDiscovery::open_raw_file(dataset_file_path);
    RAWEventFileReader reader(*device, dataset_file_path);
    auto metadata_map = reader.get_metadata_map();
    ASSERT_FALSE(metadata_map.empty());
    auto it = metadata_map.find("integrator_name");
    ASSERT_TRUE(it != metadata_map.end());
    EXPECT_EQ("Prophesee", it->second);
    it = metadata_map.find("serial_number");
    ASSERT_TRUE(it != metadata_map.end());
    EXPECT_EQ("00001621", it->second);
    it = metadata_map.find("evt");
    ASSERT_TRUE(it != metadata_map.end());
    EXPECT_EQ("2.0", it->second);
}

TEST_WITH_DATASET(RAWEventFileReader_Gtest, has_seek_callbacks) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
    std::unique_ptr<Device> device = DeviceDiscovery::open_raw_file(dataset_file_path);
    RAWEventFileReader reader(*device, dataset_file_path);
    ASSERT_FALSE(reader.has_seek_callbacks());
    size_t id = reader.add_seek_callback([](const auto &) {});
    ASSERT_TRUE(reader.has_seek_callbacks());
    reader.remove_callback(id);
    ASSERT_FALSE(reader.has_seek_callbacks());
}

TEST_WITH_DATASET(RAWEventFileReader_Gtest, get_seek_range) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
    RawFileConfig raw_file_stream_config;
    raw_file_stream_config.build_index_ = true;
    std::unique_ptr<Device> device      = DeviceDiscovery::open_raw_file(dataset_file_path, raw_file_stream_config);
    RAWEventFileReader reader(*device, dataset_file_path);
    timestamp min_t, max_t;
    while (!reader.get_seek_range(min_t, max_t)) {
        std::this_thread::yield();
    }
    EXPECT_EQ(16, min_t);
    EXPECT_EQ(13042000, max_t);
}

TEST_WITH_DATASET(RAWEventFileReader_Gtest, get_duration) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
    std::unique_ptr<Device> device = DeviceDiscovery::open_raw_file(dataset_file_path);
    RAWEventFileReader reader(*device, dataset_file_path);
    EXPECT_EQ(13043033, reader.get_duration());
}

TEST_WITH_DATASET(RAWEventFileReader_Gtest, read) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) /
        "openeb" / "blinking_gen4_with_ext_triggers.raw";
    std::unique_ptr<Device> device = DeviceDiscovery::open_raw_file(dataset_file_path);
    RAWEventFileReader reader(*device, dataset_file_path);
    size_t num_cd_events = 0;
    reader.add_read_callback(
        [&num_cd_events](const EventCD *begin, const EventCD *end) { num_cd_events += std::distance(begin, end); });
    size_t num_ext_trigger_events = 0;
    reader.add_read_callback([&num_ext_trigger_events](const EventExtTrigger *begin, const EventExtTrigger *end) {
        num_ext_trigger_events += std::distance(begin, end);
    });
    EXPECT_EQ(0, num_cd_events);
    EXPECT_EQ(0, num_ext_trigger_events);
    while (reader.read()) {
        std::this_thread::yield();
    }
    EXPECT_EQ(2003016, num_cd_events);
    EXPECT_EQ(82, num_ext_trigger_events);
}

TEST_WITH_DATASET(RAWEventFileReader_Gtest, seek) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
    RawFileConfig raw_file_stream_config;
    raw_file_stream_config.build_index_ = true;
    std::unique_ptr<Device> device      = DeviceDiscovery::open_raw_file(dataset_file_path, raw_file_stream_config);
    RAWEventFileReader reader(*device, dataset_file_path);
    timestamp ts = -1;
    reader.add_seek_callback([&ts](const timestamp &t) { ts = t; });
    EXPECT_EQ(-1, ts);
    EXPECT_TRUE(reader.read());
    timestamp min_t, max_t;
    while (!reader.get_seek_range(min_t, max_t)) {
        std::this_thread::yield();
    }
    EXPECT_TRUE(reader.seek(134567));
    EXPECT_EQ(132000, ts);
}

TEST_WITH_DATASET(DATEventFileReader_Gtest, constructor_valid) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) /
        "openeb" / "core" / "event_io" / "recording_td.dat";
    std::unique_ptr<DATEventFileReader> reader;
    ASSERT_NO_THROW(reader = std::make_unique<DATEventFileReader>(dataset_file_path));
}

TEST_WITH_DATASET(DATEventFileReader_Gtest, get_path) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) /
        "openeb" / "core" / "event_io" / "recording_td.dat";
    DATEventFileReader reader(dataset_file_path);
    ASSERT_EQ(dataset_file_path, reader.get_path());
}

TEST_WITH_DATASET(DATEventFileReader_Gtest, has_read_callbacks) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) /
        "openeb" / "core" / "event_io" / "recording_td.dat";

    DATEventFileReader reader(dataset_file_path);
    ASSERT_FALSE(reader.has_read_callbacks());
    size_t id = reader.add_read_callback([](const EventCD *, const EventCD *) {});
    ASSERT_TRUE(reader.has_read_callbacks());
    reader.remove_callback(id);
    ASSERT_FALSE(reader.has_read_callbacks());
    reader.add_read_callback([](const EventExtTrigger *, const EventExtTrigger *) {});
    ASSERT_TRUE(reader.has_read_callbacks());
}

TEST_WITH_DATASET(DATEventFileReader_Gtest, get_metadata_map) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) /
        "openeb" / "core" / "event_io" / "recording_td.dat";

    DATEventFileReader reader(dataset_file_path);
    auto metadata_map = reader.get_metadata_map();
    ASSERT_FALSE(metadata_map.empty());
    auto it = metadata_map.find("geometry");
    ASSERT_TRUE(it != metadata_map.end());
    EXPECT_EQ("640x480", it->second);
    it = metadata_map.find("generation");
    ASSERT_TRUE(it != metadata_map.end());
    EXPECT_EQ("3.0", it->second);
}

TEST_WITH_DATASET(DATEventFileReader_Gtest, has_seek_callbacks) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) /
        "openeb" / "core" / "event_io" / "recording_td.dat";

    DATEventFileReader reader(dataset_file_path);
    ASSERT_FALSE(reader.has_seek_callbacks());
    size_t id = reader.add_seek_callback([](const auto &) {});
    ASSERT_TRUE(reader.has_seek_callbacks());
    reader.remove_callback(id);
    ASSERT_FALSE(reader.has_read_callbacks());
}

TEST_WITH_DATASET(DATEventFileReader_Gtest, get_seek_range) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) /
        "openeb" / "core" / "event_io" / "recording_td.dat";

    DATEventFileReader reader(dataset_file_path);
    timestamp min_t, max_t;
    ASSERT_FALSE(reader.get_seek_range(min_t, max_t));
}

TEST_WITH_DATASET(DATEventFileReader_Gtest, get_duration) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) /
        "openeb" / "core" / "event_io" / "recording_td.dat";

    DATEventFileReader reader(dataset_file_path);
    EXPECT_EQ(7702845, reader.get_duration());
}

TEST_WITH_DATASET(DATEventFileReader_Gtest, read) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) /
        "openeb" / "core" / "event_io" / "recording_td.dat";
    DATEventFileReader reader(dataset_file_path);
    size_t num_cd_events = 0;
    reader.add_read_callback(
        [&num_cd_events](const EventCD *begin, const EventCD *end) { num_cd_events += std::distance(begin, end); });
    size_t num_ext_trigger_events = 0;
    reader.add_read_callback([&num_ext_trigger_events](const EventExtTrigger *begin, const EventExtTrigger *end) {
        num_ext_trigger_events += std::distance(begin, end);
    });
    EXPECT_EQ(0, num_cd_events);
    EXPECT_EQ(0, num_ext_trigger_events);
    while (reader.read()) {
        std::this_thread::yield();
    }
    EXPECT_EQ(667855, num_cd_events);
    EXPECT_EQ(0, num_ext_trigger_events);
}

TEST_WITH_DATASET(DATEventFileReader_Gtest, read_delayed_recording) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "test_start_after_0.dat";
    DATEventFileReader reader(dataset_file_path);
    size_t num_cd_events = 0;
    reader.add_read_callback(
        [&num_cd_events](const EventCD *begin, const EventCD *end) { num_cd_events += std::distance(begin, end); });
    size_t num_ext_trigger_events = 0;
    reader.add_read_callback([&num_ext_trigger_events](const EventExtTrigger *begin, const EventExtTrigger *end) {
        num_ext_trigger_events += std::distance(begin, end);
    });
    EXPECT_EQ(0, num_cd_events);
    EXPECT_EQ(0, num_ext_trigger_events);
    while (reader.read()) {
        std::this_thread::yield();
    }
    EXPECT_EQ(2045297, num_cd_events);
    EXPECT_EQ(0, num_ext_trigger_events);
}

class RAWEventFileLogger_Gtest : public GTestWithTmpDir {
protected:
    virtual void SetUp() {
        tmp_file_ = tmpdir_handler_->get_full_path("test.raw");
    }

    std::filesystem::path tmp_file_;
};

TEST_F(RAWEventFileLogger_Gtest, empty_constructor) {
    std::unique_ptr<RAWEventFileLogger> writer;
    ASSERT_NO_THROW(writer = std::make_unique<RAWEventFileLogger>());
}

TEST_F(RAWEventFileLogger_Gtest, constructor_invalid) {
    std::unique_ptr<RAWEventFileLogger> writer;
    std::filesystem::path file =
        std::filesystem::path(tmpdir_handler_->get_full_path("inexistent_directory")) / "file.raw";
    ASSERT_THROW(writer = std::make_unique<RAWEventFileLogger>(file), std::runtime_error);
}

TEST_F(RAWEventFileLogger_Gtest, constructor) {
    {
        std::unique_ptr<RAWEventFileLogger> writer;
        ASSERT_NO_THROW(writer = std::make_unique<RAWEventFileLogger>(tmp_file_));
    }
    {
        std::ifstream ifs(tmp_file_);
        ASSERT_TRUE(ifs.is_open());
    }
}

TEST_F(RAWEventFileLogger_Gtest, not_is_open) {
    std::unique_ptr<RAWEventFileLogger> writer;
    writer = std::make_unique<RAWEventFileLogger>();
    ASSERT_FALSE(writer->is_open());
}

TEST_F(RAWEventFileLogger_Gtest, is_open) {
    std::unique_ptr<RAWEventFileLogger> writer;
    writer = std::make_unique<RAWEventFileLogger>(tmp_file_);
    ASSERT_TRUE(writer->is_open());
}

TEST_F(RAWEventFileLogger_Gtest, open) {
    std::unique_ptr<RAWEventFileLogger> writer;
    ASSERT_NO_THROW(writer = std::make_unique<RAWEventFileLogger>());
    ASSERT_NO_THROW(writer->open(tmp_file_));
}

TEST_F(RAWEventFileLogger_Gtest, close) {
    std::unique_ptr<RAWEventFileLogger> writer;
    writer = std::make_unique<RAWEventFileLogger>(tmp_file_);
    ASSERT_NO_THROW(writer->close());
}

TEST_F(RAWEventFileLogger_Gtest, open_close) {
    std::unique_ptr<RAWEventFileLogger> writer;
    ASSERT_NO_THROW(writer = std::make_unique<RAWEventFileLogger>());
    ASSERT_NO_THROW(writer->open(tmp_file_));
    ASSERT_TRUE(writer->is_open());
    ASSERT_NO_THROW(writer->close());
    ASSERT_FALSE(writer->is_open());
    ASSERT_NO_THROW(writer->open(tmp_file_));
}

TEST_F_WITH_DATASET(RAWEventFileLogger_Gtest, constructor_metadata_map) {
    {
        std::filesystem::path dataset_file_path =
            std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
        std::ifstream ifs(dataset_file_path);
        GenericHeader header(ifs);
        auto header_map = header.get_header_map();
        std::unordered_map<std::string, std::string> m(header_map.begin(), header_map.end());
        m["toto"] = "blub";

        std::unique_ptr<RAWEventFileLogger> writer;
        ASSERT_NO_THROW(writer = std::make_unique<RAWEventFileLogger>(tmp_file_, m));
    }
    {
        std::ifstream ifs(tmp_file_);
        GenericHeader header(ifs);
        auto metadata_map = header.get_header_map();
        ASSERT_EQ("blub", metadata_map["toto"]);
    }
}

TEST_F(RAWEventFileLogger_Gtest, get_path) {
    RAWEventFileLogger writer(tmp_file_);
    ASSERT_EQ(tmp_file_, writer.get_path());
}

TEST_F_WITH_DATASET(RAWEventFileLogger_Gtest, add_metadata) {
    {
        std::filesystem::path dataset_file_path =
            std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
        std::ifstream ifs(dataset_file_path);
        GenericHeader header(ifs);
        auto header_map = header.get_header_map();

        RAWEventFileLogger writer(tmp_file_,
                                  std::unordered_map<std::string, std::string>(header_map.begin(), header_map.end()));
        writer.add_metadata("toto", "blub");
    }
    {
        std::ifstream ifs(tmp_file_);
        GenericHeader header(ifs);
        auto metadata_map = header.get_header_map();
        ASSERT_EQ("blub", metadata_map["toto"]);
    }
}

TEST_F_WITH_DATASET(RAWEventFileLogger_Gtest, add_metadata_no_flush_succeed) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
    std::ifstream ifs(dataset_file_path);
    GenericHeader header(ifs);
    auto header_map = header.get_header_map();

    RAWEventFileLogger writer(tmp_file_,
                              std::unordered_map<std::string, std::string>(header_map.begin(), header_map.end()));

    uint8_t buf[1] = {0};
    writer.add_raw_data(buf, 1);

    // can't add metadata to RAW once data has been added
    ASSERT_NO_THROW(writer.add_metadata("toto", "blub"));
}

TEST_F_WITH_DATASET(RAWEventFileLogger_Gtest, add_metadata_flush_raw_fail) {
    (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");

    const std::string expected_output_regex =
#ifdef _WIN32
        "";
#else
        "Unable to modify metadata in RAW";
#endif

    ASSERT_DEATH(
        {
            std::filesystem::path dataset_file_path =
                std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
            std::ifstream ifs(dataset_file_path);
            GenericHeader header(ifs);
            auto header_map = header.get_header_map();

            RAWEventFileLogger writer(
                tmp_file_, std::unordered_map<std::string, std::string>(header_map.begin(), header_map.end()));

            uint8_t buf[1] = {0};
            writer.add_raw_data(buf, 1);
            writer.flush();

            // can't add metadata to RAW once data has been added
            writer.add_metadata("toto", "blub");
        },
        expected_output_regex);
}

TEST_F_WITH_DATASET(RAWEventFileLogger_Gtest, remove_metadata) {
    {
        std::filesystem::path dataset_file_path =
            std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
        std::ifstream ifs(dataset_file_path);
        GenericHeader header(ifs);
        auto header_map = header.get_header_map();

        RAWEventFileLogger writer(tmp_file_,
                                  std::unordered_map<std::string, std::string>(header_map.begin(), header_map.end()));
        writer.add_metadata("toto1", "blub");
        writer.add_metadata("toto2", "blub");
        writer.remove_metadata("toto1");
    }
    {
        std::ifstream ifs(tmp_file_);
        GenericHeader header(ifs);
        auto metadata_map = header.get_header_map();
        ASSERT_EQ("blub", metadata_map["toto2"]);
        ASSERT_TRUE(metadata_map.find("toto1") == metadata_map.end());
    }
}

TEST_F_WITH_DATASET(RAWEventFileLogger_Gtest, add_metadata_map_from_camera) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
    Camera cam = Camera::from_file(dataset_file_path);
    {
        RAWEventFileLogger writer(tmp_file_);
        writer.add_metadata_map_from_camera(cam);
    }
    {
        std::ifstream ifs(tmp_file_);
        GenericHeader header(ifs);
        auto m1 = cam.get_metadata_map();
        auto m2 = header.get_header_map();

        // only test the intersection of fields in both files
        for (auto &p : m1) {
            if (p.first != "Date" && p.first != "date") {
                EXPECT_EQ(p.second, m2[p.first]);
            }
        }
        for (auto &p : m2) {
            if (p.first != "generation" && p.first != "geometry" && p.first != "Date" && p.first != "date") {
                EXPECT_EQ(p.second, m1[p.first]);
            }
        }
    }
}

TEST_F_WITH_DATASET(RAWEventFileLogger_Gtest, write_raw) {
    std::vector<uint8_t> expected_data;
    {
        std::filesystem::path dataset_file_path =
            std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
        Camera cam = Camera::from_file(dataset_file_path);

        RAWEventFileLogger writer(tmp_file_);
        writer.add_metadata_map_from_camera(cam);

        cam.raw_data().add_callback([&writer, &expected_data](const uint8_t *ptr, size_t size) {
            expected_data.insert(expected_data.end(), ptr, ptr + size);
            writer.add_raw_data(ptr, size);
        });

        cam.start();
        while (cam.is_running()) {
            std::this_thread::yield();
        }
    }

    {
        Camera cam        = Camera::from_file(tmp_file_);
        uint8_t *data_ptr = expected_data.data();
        cam.raw_data().add_callback([&data_ptr](const uint8_t *ptr, size_t size) {
            EXPECT_TRUE(std::equal(data_ptr, data_ptr + size, ptr));
            data_ptr += size;
        });

        cam.start();
        while (cam.is_running()) {
            std::this_thread::yield();
        }
    }
}

class HDF5EventFileWriter_Gtest : public GTestWithTmpDir {
protected:
    virtual void SetUp() {
        tmp_file_ = tmpdir_handler_->get_full_path("test.hdf5");
    }

    std::filesystem::path tmp_file_;
};

TEST_F(HDF5EventFileWriter_Gtest, empty_constructor) {
    std::unique_ptr<HDF5EventFileWriter> writer;
    ASSERT_NO_THROW(writer = std::make_unique<HDF5EventFileWriter>());
}

TEST_F(HDF5EventFileWriter_Gtest, constructor_invalid) {
    std::unique_ptr<HDF5EventFileWriter> writer;
    std::filesystem::path file =
        std::filesystem::path(tmpdir_handler_->get_full_path("inexistent_directory")) / "file.hdf5";
    ASSERT_THROW(writer = std::make_unique<HDF5EventFileWriter>(file), H5::FileIException);
}

TEST_F_WITH_DATASET(HDF5EventFileWriter_Gtest, constructor) {
    {
        std::unique_ptr<HDF5EventFileWriter> writer;
        ASSERT_NO_THROW(writer = std::make_unique<HDF5EventFileWriter>(tmp_file_));
    }
    {
        std::ifstream ifs(tmp_file_);
        ASSERT_TRUE(ifs.is_open());
    }
}

TEST_F(HDF5EventFileWriter_Gtest, constructor_check_version) {
    {
        std::unique_ptr<HDF5EventFileWriter> writer;
        ASSERT_NO_THROW(writer = std::make_unique<HDF5EventFileWriter>(tmp_file_));
    }
    {
        HDF5EventFileReader reader(tmp_file_);
        auto metadata_map = reader.get_metadata_map();
        ASSERT_TRUE(metadata_map.find("version") != metadata_map.end());
    }
}

TEST_F(HDF5EventFileWriter_Gtest, not_is_open) {
    std::unique_ptr<HDF5EventFileWriter> writer;
    writer = std::make_unique<HDF5EventFileWriter>();
    ASSERT_FALSE(writer->is_open());
}

TEST_F(HDF5EventFileWriter_Gtest, is_open) {
    std::unique_ptr<HDF5EventFileWriter> writer;
    writer = std::make_unique<HDF5EventFileWriter>(tmp_file_);
    ASSERT_TRUE(writer->is_open());
}

TEST_F(HDF5EventFileWriter_Gtest, open) {
    std::unique_ptr<HDF5EventFileWriter> writer;
    ASSERT_NO_THROW(writer = std::make_unique<HDF5EventFileWriter>());
    ASSERT_NO_THROW(writer->open(tmp_file_));
}

TEST_F(HDF5EventFileWriter_Gtest, close) {
    std::unique_ptr<HDF5EventFileWriter> writer;
    writer = std::make_unique<HDF5EventFileWriter>(tmp_file_);
    ASSERT_NO_THROW(writer->close());
}

TEST_F(HDF5EventFileWriter_Gtest, open_close) {
    std::unique_ptr<HDF5EventFileWriter> writer;
    ASSERT_NO_THROW(writer = std::make_unique<HDF5EventFileWriter>());
    ASSERT_NO_THROW(writer->open(tmp_file_));
    ASSERT_TRUE(writer->is_open());
    ASSERT_NO_THROW(writer->close());
    ASSERT_FALSE(writer->is_open());
    ASSERT_NO_THROW(writer->open(tmp_file_));
}

TEST_F(HDF5EventFileWriter_Gtest, constructor_metadata_map) {
    {
        std::unique_ptr<HDF5EventFileWriter> writer;
        ASSERT_NO_THROW(writer = std::make_unique<HDF5EventFileWriter>(
                            tmp_file_, std::unordered_map<std::string, std::string>{{"toto", "blub"}}));
    }
    {
        HDF5EventFileReader reader(tmp_file_);
        auto metadata_map = reader.get_metadata_map();
        ASSERT_EQ("blub", metadata_map["toto"]);
    }
}

TEST_F(HDF5EventFileWriter_Gtest, get_path) {
    HDF5EventFileWriter writer(tmp_file_);
    ASSERT_EQ(tmp_file_, writer.get_path());
}

TEST_F(HDF5EventFileWriter_Gtest, add_metata) {
    {
        HDF5EventFileWriter writer(tmp_file_);
        writer.add_metadata("toto", "blub");
    }
    {
        HDF5EventFileReader reader(tmp_file_);
        auto metadata_map = reader.get_metadata_map();
        ASSERT_EQ("blub", metadata_map["toto"]);
    }
}

TEST_F(HDF5EventFileWriter_Gtest, remove_metata) {
    {
        HDF5EventFileWriter writer(tmp_file_);
        writer.add_metadata("toto1", "blub");
        writer.add_metadata("toto2", "blub");
        writer.remove_metadata("toto1");
    }
    {
        HDF5EventFileReader reader(tmp_file_);
        auto metadata_map = reader.get_metadata_map();
        ASSERT_EQ("blub", metadata_map["toto2"]);
        ASSERT_TRUE(metadata_map.find("toto1") == metadata_map.end());
    }
}

TEST_F_WITH_DATASET(HDF5EventFileWriter_Gtest, add_metata_map_from_camera) {
    std::filesystem::path dataset_file_path =
        std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
    Camera cam = Camera::from_file(dataset_file_path);
    {
        HDF5EventFileWriter writer(tmp_file_);
        writer.add_metadata_map_from_camera(cam);
    }
    {
        HDF5EventFileReader reader(tmp_file_);
        auto m1 = cam.get_metadata_map();
        auto m2 = reader.get_metadata_map();

        // only test the intersection of fields in both files
        for (auto &p : m1) {
            if (p.first != "Date" && p.first != "date" && p.first != "plugin_name" && p.first != "evt") {
                EXPECT_EQ(p.second, m2[p.first]);
            }
        }
        for (auto &p : m2) {
            if (p.first != "generation" && p.first != "geometry" && p.first != "Date" && p.first != "date" &&
                p.first != "version") {
                EXPECT_EQ(p.second, m1[p.first]);
            }
        }
    }
}

TEST_F(HDF5EventFileWriter_Gtest, simple_write_cds) {
    {
        HDF5EventFileWriter writer(tmp_file_);
        std::vector<EventCD> events{EventCD(0, 0, 0, 0)};
        ASSERT_TRUE(writer.add_events(events.data(), events.data() + 1));
    }

    {
        HDF5EventFileReader reader(tmp_file_);
        size_t num_calls = 0;
        reader.add_read_callback([&num_calls](const EventCD *begin, const EventCD *end) {
            ASSERT_EQ(1, std::distance(begin, end));
            ASSERT_EQ(0, begin->x);
            ASSERT_EQ(0, begin->y);
            ASSERT_EQ(0, begin->t);
            ASSERT_EQ(0, begin->p);
            ++num_calls;
        });
        reader.add_read_callback([](const EventExtTrigger *begin, const EventExtTrigger *end) { FAIL(); });
        while (reader.read()) {
            std::this_thread::yield();
        }
        ASSERT_EQ(1, num_calls);
    }
}

TEST_F(HDF5EventFileWriter_Gtest, simple_write_triggers) {
    {
        HDF5EventFileWriter writer(tmp_file_);
        std::vector<EventExtTrigger> events{EventExtTrigger(0, 0, 0)};
        ASSERT_TRUE(writer.add_events(events.data(), events.data() + 1));
    }

    {
        HDF5EventFileReader reader(tmp_file_);
        size_t num_calls = 0;
        reader.add_read_callback([&num_calls](const EventExtTrigger *begin, const EventExtTrigger *end) {
            ASSERT_EQ(1, std::distance(begin, end));
            ASSERT_EQ(0, begin->p);
            ASSERT_EQ(0, begin->t);
            ASSERT_EQ(0, begin->id);
            ++num_calls;
        });
        reader.add_read_callback([](const EventCD *begin, const EventCD *end) { FAIL(); });
        while (reader.read()) {
            std::this_thread::yield();
        }
        ASSERT_EQ(1, num_calls);
    }
}

TEST_F(HDF5EventFileWriter_Gtest, simple_write_cd_and_triggers) {
    {
        HDF5EventFileWriter writer(tmp_file_);
        std::vector<EventCD> events_cd{EventCD(0, 0, 0, 0)};
        ASSERT_TRUE(writer.add_events(events_cd.data(), events_cd.data() + 1));
        std::vector<EventExtTrigger> events_trigger{EventExtTrigger(0, 0, 0)};
        ASSERT_TRUE(writer.add_events(events_trigger.data(), events_trigger.data() + 1));
    }

    {
        HDF5EventFileReader reader(tmp_file_);
        size_t num_calls_cd = 0;
        reader.add_read_callback([&num_calls_cd](const EventCD *begin, const EventCD *end) {
            ASSERT_EQ(1, std::distance(begin, end));
            ASSERT_EQ(0, begin->x);
            ASSERT_EQ(0, begin->y);
            ASSERT_EQ(0, begin->t);
            ASSERT_EQ(0, begin->p);
            ++num_calls_cd;
        });
        size_t num_calls_trigger = 0;
        reader.add_read_callback([&num_calls_trigger](const EventExtTrigger *begin, const EventExtTrigger *end) {
            ASSERT_EQ(1, std::distance(begin, end));
            ASSERT_EQ(0, begin->p);
            ASSERT_EQ(0, begin->t);
            ASSERT_EQ(0, begin->id);
            ++num_calls_trigger;
        });
        while (reader.read()) {
            std::this_thread::yield();
        }
        ASSERT_EQ(1, num_calls_cd);
        ASSERT_EQ(1, num_calls_trigger);
    }
}

TEST_F(HDF5EventFileWriter_Gtest, invalid_add_events) {
    HDF5EventFileWriter writer(tmp_file_);
    std::vector<EventCD> events_cd{EventCD(0, 0, 0, 1)};
    ASSERT_TRUE(writer.add_events(events_cd.data(), events_cd.data() + 1));
    events_cd[0] = EventCD(0, 0, 0, 0);
    ASSERT_THROW(writer.add_events(events_cd.data(), events_cd.data() + 1), std::runtime_error);
}

TEST_F(HDF5EventFileWriter_Gtest, write_first_ts_is_big) {
    const size_t num_events  = 2;
    const timestamp first_ts = 1000000;
    std::vector<EventCD> expected_events_cd;
    {
        HDF5EventFileWriter writer(tmp_file_);
        std::vector<EventCD> events_cd(num_events);
        for (size_t i = 0; i < num_events; ++i) {
            events_cd[i].x = 0;
            events_cd[i].y = 0;
            events_cd[i].p = 0;
            events_cd[i].t = first_ts + 2000 * i;
        }
        ASSERT_TRUE(writer.add_events(events_cd.data(), events_cd.data() + events_cd.size()));
    }
    {
        std::vector<EventCD> events_cd;
        HDF5EventFileReader reader(tmp_file_);
        reader.add_read_callback(
            [&events_cd](const EventCD *begin, const EventCD *end) { events_cd.insert(events_cd.end(), begin, end); });
        while (reader.read()) {
            std::this_thread::yield();
        }
        ASSERT_EQ(num_events, events_cd.size());
        for (size_t i = 0; i < num_events; ++i) {
            ASSERT_EQ(0, events_cd[i].x);
            ASSERT_EQ(0, events_cd[i].y);
            ASSERT_EQ(0, events_cd[i].p);
            ASSERT_EQ(first_ts + 2000 * i, events_cd[i].t);
        }
    }
    {
        // check that we don't have a long list of Index(0,-1) at the beginning of the indexes table
        H5::H5File file(tmp_file_.string(), H5F_ACC_RDONLY);
        H5::DataSet dset = file.openDataSet("/CD/indexes");
        struct Index {
            Index(size_t id = 0, std::int64_t ts = 0) : id(id), ts(ts) {}
            size_t id;
            std::int64_t ts;
        };
        H5::CompType dt(sizeof(Index));
        dt.insertMember("id", HOFFSET(Index, id), H5::PredType::NATIVE_ULLONG);
        dt.insertMember("ts", HOFFSET(Index, ts), H5::PredType::NATIVE_LLONG);

        const std::vector<Index> expected_indexes{{0, -1}, {0, 0}, {1, 2000}};
        H5::DataSpace ds = dset.getSpace();
        hsize_t dims[1];
        ds.getSimpleExtentDims(dims, nullptr);

        ASSERT_EQ(expected_indexes.size(), dims[0]);

        std::vector<Index> indexes(dims[0]);
        dset.read(indexes.data(), dt);

        for (size_t pos = 0; pos < expected_indexes.size(); ++pos) {
            EXPECT_EQ(expected_indexes[pos].id, indexes[pos].id);
            EXPECT_EQ(expected_indexes[pos].ts, indexes[pos].ts);
        }
    }
}

TEST_F(HDF5EventFileWriter_Gtest, random_writes) {
    std::vector<EventCD> expected_events_cd;
    std::vector<EventExtTrigger> expected_events_trigger;
    {
        HDF5EventFileWriter writer(tmp_file_);
        size_t num_events = 10000;
        std::mt19937 mt_rand; // Mersenne twister
        std::uniform_int_distribution<int> dx(0, 1000), dy(0, 800), dt(0, 3), dp(0, 1);
        mt_rand.seed(42);

        size_t id           = 0;
        timestamp cur_cd_ts = 0, cur_trigger_ts = 0;
        for (size_t j = 0; j < 10; ++j) {
            std::vector<EventCD> events_cd(num_events);
            for (size_t i = 0; i < num_events; ++i) {
                events_cd[i].x = dx(mt_rand);
                events_cd[i].y = dy(mt_rand);
                events_cd[i].p = dp(mt_rand);
                if (i == 0) {
                    events_cd[i].t = cur_cd_ts + dt(mt_rand);
                } else {
                    events_cd[i].t = events_cd[i - 1].t + dt(mt_rand);
                }
            }
            cur_cd_ts = events_cd.back().t;
            ASSERT_TRUE(writer.add_events(events_cd.data(), events_cd.data() + events_cd.size()));
            expected_events_cd.insert(expected_events_cd.end(), events_cd.begin(), events_cd.end());

            std::vector<EventExtTrigger> events_trigger(num_events);
            for (size_t i = 0; i < num_events; ++i) {
                events_trigger[i].id = id++;
                events_trigger[i].p  = dp(mt_rand);
                if (i == 0) {
                    events_trigger[i].t = cur_trigger_ts + dt(mt_rand);
                } else {
                    events_trigger[i].t = events_trigger[i - 1].t + dt(mt_rand);
                }
            }
            cur_trigger_ts = events_trigger.back().t;
            ASSERT_TRUE(writer.add_events(events_trigger.data(), events_trigger.data() + events_trigger.size()));
            expected_events_trigger.insert(expected_events_trigger.end(), events_trigger.begin(), events_trigger.end());
        }
    }
    {
        std::vector<EventCD> events_cd;
        std::vector<EventExtTrigger> events_trigger;
        HDF5EventFileReader reader(tmp_file_);
        size_t id = reader.add_read_callback(
            [&events_cd](const EventCD *begin, const EventCD *end) { events_cd.insert(events_cd.end(), begin, end); });
        reader.add_read_callback([&events_trigger](const EventExtTrigger *begin, const EventExtTrigger *end) {
            events_trigger.insert(events_trigger.end(), begin, end);
        });
        while (reader.read()) {
            std::this_thread::yield();
        }

        ASSERT_EQ(expected_events_cd.size(), events_cd.size());

        for (size_t i = 0; i < expected_events_cd.size(); ++i) {
            ASSERT_EQ(expected_events_cd[i].x, events_cd[i].x);
            ASSERT_EQ(expected_events_cd[i].y, events_cd[i].y);
            ASSERT_EQ(expected_events_cd[i].p, events_cd[i].p);
            ASSERT_EQ(expected_events_cd[i].t, events_cd[i].t);
        }

        ASSERT_EQ(expected_events_trigger.size(), events_trigger.size());

        for (size_t i = 0; i < expected_events_trigger.size(); ++i) {
            ASSERT_EQ(expected_events_trigger[i].p, events_trigger[i].p);
            ASSERT_EQ(expected_events_trigger[i].id, events_trigger[i].id);
            ASSERT_EQ(expected_events_trigger[i].t, events_trigger[i].t);
        }
    }
}
