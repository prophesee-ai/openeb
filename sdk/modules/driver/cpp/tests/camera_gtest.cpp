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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <gtest/gtest-message.h>
#include <gtest/gtest.h>
#include <numeric> // std::iota
#include <sstream>
#include <thread>

#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/utils/raw_file_header.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/driver/camera_error_code.h"
#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/utils/gtest/gtest_custom.h"
#include "metavision/sdk/core/utils/callback_manager.h"
#include "metavision/sdk/driver/camera.h"
#include "metavision/sdk/driver/internal/camera_internal.h"
#include "metavision/sdk/driver/internal/callback_tag_ids.h"
#include "metavision/sdk/driver/camera_exception.h"
#include "metavision/sdk/driver/internal/camera_error_code_internal.h"
#include "encoding_policies.h"
#include "tencoder_gtest_common.h"
#include "device_builder_maker.h"

using namespace Metavision;

class Camera_Gtest : public GTestWithTmpDir {
protected:
    virtual void SetUp() {
        tmp_file_      = tmpdir_handler_->get_full_path("Camera_Gtest.raw");
        bytes_written_ = 0;
    }

    virtual void TearDown() {
        close_file();
    }

    static bool error_message_found_in_exception(int error_code_to_find, const CameraException &e) {
        std::stringstream ss;
        ss << std::hex << error_code_to_find;
        auto what = std::string(e.what());
        return what.find(ss.str()) != std::string::npos;
    }

    void write_header(const RawFileHeader &header) {
        (*log_raw_data_) << header;
    }

    RawFileHeader get_default_header(bool evt2 = true) {
        RawFileHeader header_to_write;
        header_to_write.set_plugin_name(evt2 ? "hal_plugin_gen31_fx3" : "hal_plugin_gen4_fx3");
        header_to_write.set_plugin_integrator_name("Prophesee");
        header_to_write.set_camera_integrator_name("Prophesee");
        header_to_write.set_field("serial_number", dummy_serial_);

        // Prophesee header only. Duplicated what PropheseeRawHeader does to be able to encode then read test RAW
        // file
        header_to_write.set_field("system_ID", std::to_string(dummy_system_id));
        return header_to_write;
    }

    void open_file() {
        log_raw_data_.reset(new std::ofstream(tmp_file_, std::ios::binary));
    }

    void close_file() {
        log_raw_data_.reset(nullptr);
    }

    std::vector<EventCD> write_evt2_raw_cd_events() {
        auto events = build_vector_of_events<Evt2RawFormat, EventCD>();
        TEncoder<Evt2RawFormat, TimerHighRedundancyEvt2Default> encoder;
        encoder.set_encode_event_callback([&](const uint8_t *data, const uint8_t *data_end) {
            log_raw_data_->write(reinterpret_cast<const char *>(data), std::distance(data, data_end));
            bytes_written_ += std::distance(data, data_end);
        });

        encoder.encode(events.cbegin(), events.cend());
        encoder.flush();

        return events;
    }

    std::pair<std::vector<EventCD>, std::vector<EventExtTrigger>> write_evt2_raw_cd_and_ext_trigger_events() {
        auto events         = build_vector_of_events<Evt2RawFormat, EventCD>();
        auto events_trigger = build_vector_of_events<Evt2RawFormat, EventExtTrigger>();
        TEncoder<Evt2RawFormat, TimerHighRedundancyEvt2Default> encoder;
        encoder.set_encode_event_callback([&](const uint8_t *data, const uint8_t *data_end) {
            log_raw_data_->write(reinterpret_cast<const char *>(data), std::distance(data, data_end));
            bytes_written_ += std::distance(data, data_end);
        });

        encoder.encode(events.cbegin(), events.cend(), events_trigger.cbegin(), events_trigger.cend());
        encoder.flush();
        return std::make_pair(events, events_trigger);
    }

    std::vector<EventCD> write_evt2_raw_data() {
        std::vector<EventCD> data;
        open_file();

        write_header(get_default_header());
        data = write_evt2_raw_cd_events();
        close_file();

        return data;
    }

    std::pair<std::vector<EventCD>, std::vector<EventExtTrigger>> write_evt2_raw_data_with_trigger() {
        static bool written = false;
        static std::string evt2_file_name;
        static std::pair<std::vector<EventCD>, std::vector<EventExtTrigger>> data;
        if (written) {
            tmp_file_ = evt2_file_name;
            return data;
        }

        open_file();
        evt2_file_name = tmp_file_;
        written        = true;

        write_header(get_default_header());
        data = write_evt2_raw_cd_and_ext_trigger_events();
        close_file();

        return data;
    }

    enum class DatasetFileType { RAW = 1 << 0, HDF5 = 1 << 1, ALL = 1 << 0 | 1 << 1 };

    std::vector<std::string> get_datasets_paths(DatasetFileType type) {
        std::vector<std::string> datasets;
        if (static_cast<int>(type) & static_cast<int>(DatasetFileType::RAW)) {
            for (const auto &p :
                 {"gen31_timer.raw", "gen4_evt2_hand.raw", "gen4_evt3_hand.raw", "blinking_gen4_with_ext_triggers.raw",
                  "claque_doigt_evt21.raw", "standup_evt21-legacy.raw"}) {
                datasets.push_back(
                    (boost::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / p).string());
            }
        }
#ifdef HAS_HDF5
        if (static_cast<int>(type) & static_cast<int>(DatasetFileType::HDF5)) {
            for (const auto &p :
                 {"gen31_timer.hdf5", "gen4_evt2_hand.hdf5", "gen4_evt3_hand.hdf5",
                  "blinking_gen4_with_ext_triggers.hdf5", "claque_doigt_evt21.hdf5", "standup_evt21-legacy.hdf5"}) {
                datasets.push_back(
                    (boost::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / p).string());
            }
        }
#endif
        return datasets;
    }

    std::string tmp_file_;
    std::unique_ptr<std::ofstream> log_raw_data_;
    size_t bytes_written_{0};

    static const std::string dummy_serial_;
    static const std::string dummy_plugin_name_;

    static constexpr long dummy_system_id = 28; // Prophesee gen31 system id
};

constexpr long Camera_Gtest::dummy_system_id;
const std::string Camera_Gtest::dummy_serial_ = "dummy_serial";

TEST_F_WITH_CAMERA(Camera_Gtest, available_online_sources) {
    auto available_sources = Camera::list_online_sources();
    ASSERT_FALSE(available_sources.empty());

    // Check that we actually found the one we are looking for if we gave the serial number as input arg
    if (!GtestsParameters::instance().serial.empty()) {
        bool found_wanted_camera = false;
        for (auto camera : available_sources) {
            for (auto &serial : camera.second) {
                if (serial == GtestsParameters::instance().serial) {
                    found_wanted_camera = true;
                    break;
                }
            }
            if (found_wanted_camera) {
                break;
            }
        }
        ASSERT_TRUE(found_wanted_camera);
    }
}

TEST_F_WITHOUT_CAMERA(Camera_Gtest, unavailable_online_sources) {
    auto available_sources = Camera::list_online_sources();
    for (auto camera : available_sources) {
        ASSERT_EQ(camera.first, OnlineSourceType::REMOTE);
    }
}

TEST_F(Camera_Gtest, basic_exceptions) {
    Camera camera;
    auto &pimpl = camera.get_pimpl();

    try {
        pimpl.check_initialization();
        std::cout << "Should have thrown exception..." << std::endl;
        FAIL();
    } catch (CameraException &e) { ASSERT_EQ(e.code().value(), CameraErrorCode::CameraNotInitialized); }

    try {
        camera.biases().set_from_file(tmpdir_handler_->get_tmpdir_path() + ".bias");
        std::cout << "Should have thrown exception..." << std::endl;
        FAIL();
    } catch (CameraException &e) { ASSERT_EQ(e.code().value(), CameraErrorCode::CameraNotInitialized); }

    try {
        camera.raw_data().add_callback([](const uint8_t *data, size_t size) {});
        std::cout << "Should have thrown exception..." << std::endl;
        FAIL();
    } catch (CameraException &e) { ASSERT_EQ(e.code().value(), CameraErrorCode::CameraNotInitialized); }

    try {
        camera.cd().add_callback([](const EventCD *, const EventCD *) {});
        std::cout << "Should have thrown exception..." << std::endl;
        FAIL();
    } catch (CameraException &e) { ASSERT_EQ(e.code().value(), CameraErrorCode::CameraNotInitialized); }

    try {
        camera.ext_trigger().add_callback([](const EventExtTrigger *, const EventExtTrigger *) {});
        std::cout << "Should have thrown exception..." << std::endl;
        FAIL();
    } catch (CameraException &e) { ASSERT_EQ(e.code().value(), CameraErrorCode::CameraNotInitialized); }

    try {
        camera.stop_recording();
        std::cout << "Should have thrown exception..." << std::endl;
        FAIL();
    } catch (CameraException &e) { ASSERT_EQ(e.code().value(), CameraErrorCode::CameraNotInitialized); }
}

TEST_F(Camera_Gtest, basic_exceptions_with_additional_info) {
    const std::string additional_info = "PROBLEM";

    try {
        throw(CameraException(CameraErrorCode::CameraNotFound, additional_info));
    } catch (CameraException &e) { ASSERT_TRUE(std::string(e.what()).find(additional_info) != std::string::npos); }
}

TEST_F_WITH_CAMERA(Camera_Gtest, camera_set_biases) {
    //////////////////////////////////////////////////////
    // PURPOSE
    // Checks that you can set biases from files

    Camera camera;
    if (GtestsParameters::instance().serial.empty()) {
        camera = Camera::from_first_available();
    } else {
        camera = Camera::from_serial(GtestsParameters::instance().serial);
    }

    try {
        camera.biases().set_from_file("not_a_bias_file.yolo");
        std::cout << "Should have thrown exception..." << std::endl;
        FAIL();
    } catch (CameraException &e) {
        // internal errors should appear to the user as the main category error
        ASSERT_NE(CameraErrorCode::InvalidArgument, e.code().value());
    }
}

TEST_F_WITHOUT_CAMERA(Camera_Gtest, no_camera_constructors) {
    try {
        Camera camera = Camera::from_first_available();
        std::cout << "Should have thrown exception..." << std::endl;
        FAIL();
    } catch (CameraException &e) { ASSERT_EQ(e.code().value(), CameraErrorCode::CameraNotFound); }

    try {
        Camera camera = Camera::from_source(OnlineSourceType::EMBEDDED);
        std::cout << "Should have thrown exception..." << std::endl;
        FAIL();
    } catch (CameraException &e) { ASSERT_EQ(e.code().value(), CameraErrorCode::CameraNotFound); }

    try {
        Camera camera = Camera::from_source(OnlineSourceType::USB);
        std::cout << "Should have thrown exception..." << std::endl;
    } catch (CameraException &e) { ASSERT_EQ(e.code().value(), CameraErrorCode::CameraNotFound); }

    try {
        Camera camera = Camera::from_serial("tafa");
        std::cout << "Should have thrown exception..." << std::endl;
    } catch (CameraException &e) { ASSERT_EQ(e.code().value(), CameraErrorCode::CameraNotFound); }
}

TEST_F_WITH_CAMERA(Camera_Gtest, camera_constructors) {
    Camera camera;
    if (GtestsParameters::instance().serial.empty()) {
        ASSERT_NO_THROW(camera = Camera::from_first_available());
    } else {
        ASSERT_NO_THROW(camera = Camera::from_serial(GtestsParameters::instance().serial));
    }
    std::string serial = camera.get_camera_configuration().serial_number;
    // important : close the camera before trying to re-open it
    camera = Camera();
    ASSERT_NO_THROW(camera = Camera::from_serial(serial));

    auto online_sources = Camera::list_online_sources();
    int sources;
    sources = 0;
    for (auto serial : online_sources[OnlineSourceType::EMBEDDED]) {
        ASSERT_NO_THROW(camera = Camera::from_source(OnlineSourceType::EMBEDDED, sources));
        // important : close the camera before trying to re-open it
        camera = Camera();
        ASSERT_NO_THROW(camera = Camera::from_serial(serial));
        sources++;
    }
    sources = 0;
    for (auto serial : online_sources[OnlineSourceType::USB]) {
        ASSERT_NO_THROW(camera = Camera::from_source(OnlineSourceType::USB, sources));
        // important : close the camera before trying to re-open it
        camera = Camera();
        ASSERT_NO_THROW(camera = Camera::from_serial(serial));
        sources++;
    }
}

TEST_F_WITH_CAMERA(Camera_Gtest, camera_file_logger_some_events) {
    std::string file = tmpdir_handler_->get_full_path("Camera_Gtest_log.raw");
    Camera camera;
    if (GtestsParameters::instance().serial.empty()) {
        camera = Camera::from_first_available();
    } else {
        camera = Camera::from_serial(GtestsParameters::instance().serial);
    }

    std::atomic<bool> stop(false);
    camera.cd().add_callback([&stop](const EventCD *, const EventCD *) { stop = true; });
    camera.start_recording(file);
    camera.start();

    while (!stop) {
        // busy wait
    }

    camera.stop();
    ASSERT_TRUE(boost::filesystem::exists(file));

    ASSERT_NO_THROW(camera = Camera::from_file(file));
}

template<typename ParamsSet>
class Camera_GtestT : public Camera_Gtest {};

template<bool do_time_shifting>
struct ParamsSet {
    static const bool do_time_shifting_ = do_time_shifting;
};

typedef ::testing::Types<ParamsSet<true>, ParamsSet<false>> TestingTypes;

TYPED_TEST_CASE(Camera_GtestT, TestingTypes);

TYPED_TEST_WITH_CAMERA(Camera_GtestT, camera_file_logger_n_events,
                       camera_params(camera_param().integrator("Prophesee").generation("3.0"),
                                     camera_param().integrator("Prophesee").generation("3.1"))) {
    // TODO MV-233 investigate why this test fails on gen 4.0 and 4.1 cameras
    std::string file = this->tmpdir_handler_->get_full_path("Camera_Gtest_log.raw");
    Camera camera;
    if (GtestsParameters::instance().serial.empty()) {
        camera = Camera::from_first_available();
    } else {
        camera = Camera::from_serial(GtestsParameters::instance().serial);
    }

    std::atomic<bool> stop(false);
    size_t num_events = size_t(0), min_num_events = size_t(1000);
    camera.cd().add_callback([&num_events, min_num_events, &stop](const EventCD *begin, const EventCD *end) {
        num_events += std::distance(begin, end);
        if (num_events > min_num_events)
            stop = true;
    });

    camera.start_recording(file);
    camera.start();

    while (!stop) {
        // busy wait
    }

    camera.stop();
    ASSERT_TRUE(boost::filesystem::exists(file));

    Metavision::FileConfigHints hints = Metavision::FileConfigHints().time_shift(TypeParam::do_time_shifting_);
    ASSERT_NO_THROW(camera = Camera::from_file(file, hints));

    size_t check_num_events = size_t(0);
    camera.cd().add_callback([&check_num_events](const EventCD *begin, const EventCD *end) {
        check_num_events += std::distance(begin, end);
    });
    camera.start();

    while (camera.is_running()) {}
    camera.stop();
    ASSERT_EQ(num_events, check_num_events);
}

TEST_F(Camera_Gtest, raw_default_constructor) {
    write_evt2_raw_data();
    try {
        Camera camera                                   = Camera::from_file(tmp_file_);
        const CameraConfiguration &camera_configuration = camera.get_camera_configuration();
        ASSERT_NE(camera_configuration.serial_number, "");
    } catch (CameraException &e) { FAIL(); }

    try {
        Camera camera = Camera::from_file("non_existing_file.raw");
        std::cout << "Should have thrown exception..." << std::endl;
        FAIL();
    } catch (CameraException &e) { ASSERT_EQ(e.code().value(), CameraErrorCode::FileDoesNotExist); }

    try {
        std::ofstream file(tmp_file_ + ".fake");
        file.close();
        Camera camera = Camera::from_file(tmp_file_ + ".fake");
        std::cout << "Should have thrown exception..." << std::endl;
        FAIL();
    } catch (CameraException &e) { ASSERT_EQ(e.code().value(), CameraErrorCode::WrongExtension); }

    try {
        Camera camera = Camera::from_file(tmp_file_);
        camera.biases().set_from_file(tmpdir_handler_->get_tmpdir_path() + ".bias"); // should throw exception
        FAIL();
    } catch (CameraException &e) {
        // internal errors should appear to the user as the main category error
        ASSERT_NE(UnsupportedFeatureErrors::BiasesUnavailable, e.code().value());
        ASSERT_EQ(CameraErrorCode::UnsupportedFeature, e.code().value());

        // but the real error code must appear in the camera exception message
        ASSERT_TRUE(error_message_found_in_exception(UnsupportedFeatureErrors::BiasesUnavailable, e));
    }

    try {
        Camera camera = Camera::from_file(tmp_file_);
        camera.biases();
        std::cout << "Should have thrown exception..." << std::endl;
        FAIL();
    } catch (CameraException &e) {
        // internal errors should appear to the user as the main category error
        ASSERT_NE(UnsupportedFeatureErrors::BiasesUnavailable, e.code().value());
        ASSERT_EQ(CameraErrorCode::UnsupportedFeature, e.code().value());

        // but the real error code must appear in the camera exception message
        ASSERT_TRUE(error_message_found_in_exception(UnsupportedFeatureErrors::BiasesUnavailable, e));
    }
}

TEST_F(Camera_Gtest, raw_file_logger_invalid_filename) {
    write_evt2_raw_data();

    std::string file =
        (boost::filesystem::path(tmpdir_handler_->get_full_path("inexistent_directory")) / "Camera_Gtest_log.raw")
            .string();
    Camera camera = Camera::from_file(tmp_file_);
    ASSERT_FALSE(boost::filesystem::exists(
        file)); // Just to check that the file does not exists already, otherwise it bias the test

    // REMARK : file is an invalid filename because the directory  tmpdir_ /
    // boost::filesystem::path("inexistent_directory") does not exist

    try {
        camera.start_recording(file);
        std::cout << "Should have thrown exception..." << std::endl;
        FAIL();
    } catch (CameraException &e) {
        // should not be able to open file for log
        EXPECT_EQ(e.code().value(), CameraErrorCode::CouldNotOpenFile);
    }

    EXPECT_FALSE(boost::filesystem::exists(file));
}

TEST_F(Camera_Gtest, raw_file_logger_invalid_filename2) {
    write_evt2_raw_data();

    std::string file = tmpdir_handler_->get_full_path("other_directory.raw");
    if (!boost::filesystem::create_directories(file)) {
        FAIL();
    }
    Camera camera = Camera::from_file(tmp_file_);

    // REMARK : file is an invalid filename because it is the name of a directory
    try {
        camera.start_recording(file);
        std::cout << "Should have thrown exception..." << std::endl;
        FAIL();
    } catch (CameraException &e) {
        // should not be able to open file for log
        EXPECT_EQ(e.code().value(), CameraErrorCode::CouldNotOpenFile);
    }
}

TEST_F(Camera_Gtest, raw_file_logger_permissions) {
    write_evt2_raw_data();

    // Create a directory and set permissions
    boost::filesystem::path tmpdir_perm = tmpdir_handler_->get_full_path("tmp_directory_permissions");
    if (!boost::filesystem::create_directory(tmpdir_perm)) {
        FAIL();
    }

    boost::filesystem::path file = tmpdir_perm / boost::filesystem::path("Camera_Gtest_log.raw");
    ASSERT_FALSE(boost::filesystem::exists(
        file)); // Just to check that the file does not exists already, otherwise it bias the test

    Camera camera = Camera::from_file(tmp_file_);

    // First, before removing permissions, check that we could potentially write the file in the directory :
    ASSERT_NO_THROW(camera.start_recording(file.string()));
    ASSERT_TRUE(boost::filesystem::exists(file));
    // Remove the file
    camera.stop_recording();
    ASSERT_TRUE(boost::filesystem::remove(file));
    // REMOVE PERMISSIONS FROM THE DIRECTORY
    boost::filesystem::permissions(tmpdir_perm, boost::filesystem::perms::owner_all |
                                                    boost::filesystem::perms::group_all |
                                                    boost::filesystem::remove_perms);
    try {
        camera.start_recording(file.string());
        camera.stop();
        // On Windows, the file can be opened but can't be written to (i.e. no header at most)
        EXPECT_EQ(0, boost::filesystem::file_size(file));
    } catch (CameraException &e) { EXPECT_EQ(e.code().value(), CameraErrorCode::CouldNotOpenFile); }

    // ADD BACK THE PERMISSIONS IN ORDER TO BE ABLE TO REMOVE THE DIRECTORY
    boost::filesystem::permissions(tmpdir_perm, boost::filesystem::perms::owner_all |
                                                    boost::filesystem::perms::group_all | boost::filesystem::add_perms);

    // Chek the file does not exists. WARNING : this check has to be done after adding the permissions back, otherwise
    // an exception is thrown EXPECT_FALSE(boost::filesystem:: ::exists(file));
}

TEST_F(Camera_Gtest, raw_file_logger) {
    write_evt2_raw_data();

    std::string file = tmpdir_handler_->get_full_path("Camera_Gtest_log.raw");
    Camera camera    = Camera::from_file(tmp_file_);
    camera.start_recording(file);
    camera.start();

    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    camera.stop();
    camera.stop_recording();
    ASSERT_TRUE(boost::filesystem::exists(file));
}

TEST_F(Camera_Gtest, start_stop) {
    write_evt2_raw_data();

    Camera camera = Camera::from_file(tmp_file_);

    ASSERT_FALSE(camera.is_running());

    std::atomic<bool> wait(true);
    camera.cd().add_callback([&wait](const EventCD *begin, const EventCD *end) {
        while (wait) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    wait = true;
    ASSERT_TRUE(camera.start());
    ASSERT_TRUE(camera.is_running());

    wait = false;
    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    ASSERT_FALSE(camera.is_running());
    camera.stop();
    ASSERT_FALSE(camera.is_running());

    // check that the camera does not provide any event since the file has been read completely
    uint32_t n_events_read = 0;
    camera.cd().add_callback(
        [&n_events_read](const EventCD *begin, const EventCD *end) { n_events_read += end - begin; });
    camera.start();
    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    ASSERT_TRUE(n_events_read == 0);
}

TEST_F_WITH_CAMERA(Camera_Gtest, same_camera_succession_of_start_stop) {
    Camera camera;
    if (GtestsParameters::instance().serial.empty()) {
        ASSERT_NO_THROW(camera = Camera::from_first_available(););
    } else {
        ASSERT_NO_THROW(camera = Camera::from_serial(GtestsParameters::instance().serial););
    }

    for (int i = 0; i < 10; ++i) {
        std::cout << "Start stop test " << i + 1 << "/10\r";
        std::cout.flush();
        ASSERT_FALSE(camera.is_running());
        ASSERT_TRUE(camera.start());
        ASSERT_FALSE(camera.start());
        ASSERT_TRUE(camera.is_running());
        ASSERT_TRUE(camera.stop());
        ASSERT_FALSE(camera.stop());
    }
    std::cout << std::endl;
}

TEST_F_WITH_CAMERA(Camera_Gtest, camera_succession_of_open_start_stop) {
    for (int i = 0; i < 10; ++i) {
        Camera camera;
        if (GtestsParameters::instance().serial.empty()) {
            ASSERT_NO_THROW(camera = Camera::from_first_available(););
        } else {
            ASSERT_NO_THROW(camera = Camera::from_serial(GtestsParameters::instance().serial););
        }
        std::cout << "Start stop test " << i + 1 << "/10\r";
        std::cout.flush();
        ASSERT_FALSE(camera.is_running());
        ASSERT_TRUE(camera.start());
        ASSERT_FALSE(camera.start());
        ASSERT_TRUE(camera.is_running());
        ASSERT_TRUE(camera.stop());
        ASSERT_FALSE(camera.stop());
    }
    std::cout << std::endl;
}

TEST_F(Camera_Gtest, robust_start_stop) {
    auto buffer = write_evt2_raw_data();

    Metavision::FileConfigHints hints = Metavision::FileConfigHints().max_read_per_op(1 * 32);
    Camera camera                     = Camera::from_file(tmp_file_, hints);

    std::atomic<bool> wait(true);
    std::atomic<uint32_t> n_events_read(0);
    camera.cd().add_callback([&wait, &n_events_read](const EventCD *begin, const EventCD *end) {
        n_events_read += end - begin;
        while (wait) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    do {
        uint32_t last_n_events_read = n_events_read;
        wait                        = true;
        ASSERT_TRUE(camera.start());
        ASSERT_TRUE(camera.is_running());

        while (n_events_read == last_n_events_read) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        wait = false;
        ASSERT_TRUE(camera.stop());
        ASSERT_FALSE(camera.is_running());
    } while (n_events_read != buffer.size());
}

TEST_F_WITH_CAMERA(Camera_Gtest, camera_start_stop_keeps_callback) {
    Camera camera;
    if (GtestsParameters::instance().serial.empty()) {
        ASSERT_NO_THROW(camera = Camera::from_first_available(););
    } else {
        ASSERT_NO_THROW(camera = Camera::from_serial(GtestsParameters::instance().serial););
    }

    size_t testing_count_current = 0;
    size_t testing_count_prev    = 0;
    std::atomic<bool> received{false};
    camera.cd().add_callback([&](auto, auto) {
        ++testing_count_current;
        received = true;
    });

    for (int i = 0; i < 10; ++i) {
        std::cout << "Start stop test " << i + 1 << "/10\r";
        std::cout.flush();
        ASSERT_FALSE(camera.is_running());
        ASSERT_TRUE(camera.start());
        ASSERT_TRUE(camera.is_running());
        while (!received) {}
        ASSERT_TRUE(camera.stop());
        received = false;
        ASSERT_GT(testing_count_current, testing_count_prev);
        testing_count_prev = testing_count_current;
    }
    std::cout << std::endl;
}

TEST_F(Camera_Gtest, raw_events_callbacks) {
    const auto ref_events = write_evt2_raw_data();
    Camera camera;
    try {
        camera = Camera::from_file(tmp_file_);
    } catch (CameraException &e) { FAIL(); }

    size_t total_size = 0;
    CallbackId id =
        camera.raw_data().add_callback([&total_size](const uint8_t *data, size_t size) { total_size += size; });
    ASSERT_EQ(id, 0);

    camera.start();

    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    ASSERT_EQ(bytes_written_, total_size);
}

TEST_F(Camera_Gtest, raw_events_callbacks_decoding_check) {
    const auto expected_events = write_evt2_raw_data();

    std::vector<EventCD> received_events;
    auto device                                    = DeviceDiscovery::open_raw_file(tmp_file_);
    I_EventsStreamDecoder *i_events_stream_decoder = device->get_facility<I_EventsStreamDecoder>();
    I_EventDecoder<EventCD> *i_cd_events_decoder   = device->get_facility<I_EventDecoder<EventCD>>();
    ASSERT_TRUE(i_cd_events_decoder);
    i_cd_events_decoder->add_event_buffer_callback(
        [&received_events](const EventCD *begin, const Metavision::EventCD *end) {
            for (const EventCD *ev = begin; ev != end; ++ev) {
                received_events.push_back(*ev);
            }
        });

    Camera camera;
    try {
        camera = Camera::from_file(tmp_file_);
    } catch (CameraException &e) { FAIL(); }

    camera.raw_data().add_callback([i_events_stream_decoder](const uint8_t *data, size_t size) {
        auto raw_data_begin = const_cast<uint8_t *>(data);
        auto raw_data_end   = const_cast<uint8_t *>(data + size);
        i_events_stream_decoder->decode(raw_data_begin, raw_data_end);
    });

    camera.start();

    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    timestamp ts_shift;
    ASSERT_TRUE(i_events_stream_decoder->get_timestamp_shift(ts_shift));

    ASSERT_EQ(expected_events.size(), received_events.size());

    using SizeType = std::vector<EventCD>::size_type;
    for (SizeType i = 0; i < expected_events.size(); ++i) {
        ASSERT_EQ(expected_events[i].x, received_events[i].x);
        ASSERT_EQ(expected_events[i].y, received_events[i].y);
        ASSERT_EQ(expected_events[i].p, received_events[i].p);
        ASSERT_EQ(expected_events[i].t - ts_shift, received_events[i].t);
    }
}

TEST_F(Camera_Gtest, cd_events_callbacks) {
    const auto expected_events = write_evt2_raw_data();
    uint32_t n_events0         = 0;
    uint32_t n_events1         = 0;
    uint32_t n_events2         = 0;

    Camera camera  = Camera::from_file(tmp_file_);
    CallbackId id0 = camera.cd().add_callback(
        [&n_events0](const EventCD *ev_begin, const EventCD *ev_end) { n_events0 += std::distance(ev_begin, ev_end); });
    ASSERT_EQ(id0, 0);

    CallbackId id1 = camera.cd().add_callback(
        [&n_events1](const EventCD *ev_begin, const EventCD *ev_end) { n_events1 += std::distance(ev_begin, ev_end); });
    ASSERT_EQ(id1, 1);
    CallbackId id2 = camera.cd().add_callback(
        [&n_events2](const EventCD *ev_begin, const EventCD *ev_end) { n_events2 += std::distance(ev_begin, ev_end); });
    ASSERT_EQ(id2, 2);

    camera.start();

    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    ASSERT_EQ(expected_events.size(), n_events0);
    ASSERT_EQ(expected_events.size(), n_events1);
    ASSERT_EQ(expected_events.size(), n_events2);
}

TEST_F(Camera_Gtest, no_error_callbacks_called) {
    write_evt2_raw_data();
    Camera camera = Camera::from_file(tmp_file_);

    int error_value = -1;
    camera.add_runtime_error_callback([&error_value](const CameraException &e) { error_value = e.code().value(); });

    camera.start();

    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    ASSERT_EQ(error_value, -1);
}

TEST_F(Camera_Gtest, events_callbacks_add_remove_cd) {
    write_evt2_raw_data();

    Camera camera = Camera::from_file(tmp_file_);

    ASSERT_EQ(0, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    CallbackId id0 = camera.cd().add_callback([](const EventCD *ev_begin, const EventCD *ev_end) {});
    ASSERT_EQ(id0, 0);
    ASSERT_EQ(1, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    CallbackId id1 = camera.cd().add_callback([](const EventCD *ev_begin, const EventCD *ev_end) {});
    ASSERT_EQ(id1, 1);
    ASSERT_EQ(2, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    CallbackId id2 = camera.cd().add_callback([](const EventCD *ev_begin, const EventCD *ev_end) {});
    ASSERT_EQ(id2, 2);
    ASSERT_EQ(3, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    CallbackId id3 = camera.add_runtime_error_callback([](const CameraException &e) {});
    ASSERT_EQ(id3, 3);
    ASSERT_EQ(3, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    ASSERT_TRUE(camera.cd().remove_callback(id1));
    ASSERT_EQ(2, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    ASSERT_FALSE(camera.cd().remove_callback(id0 + id1 + id2));
    ASSERT_EQ(2, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    ASSERT_TRUE(camera.remove_runtime_error_callback(id3));
    ASSERT_EQ(2, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));
}

TEST_F(Camera_Gtest, events_callbacks_add_remove_cd_em) {
    write_evt2_raw_data();

    Camera camera = Camera::from_file(tmp_file_);

    ASSERT_EQ(0, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    CallbackId id0 = camera.cd().add_callback([](const EventCD *ev_begin, const EventCD *ev_end) {});
    ASSERT_EQ(id0, 0);
    ASSERT_EQ(1, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    CallbackId id2 = camera.cd().add_callback([](const EventCD *ev_begin, const EventCD *ev_end) {});
    ASSERT_EQ(id2, 1);
    ASSERT_EQ(2, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    CallbackId id3 = camera.add_runtime_error_callback([](const CameraException &e) {});
    ASSERT_EQ(id3, 2);
    ASSERT_EQ(2, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    ASSERT_TRUE(camera.cd().remove_callback(id0));
    ASSERT_EQ(1, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    ASSERT_TRUE(camera.remove_runtime_error_callback(id3));
    ASSERT_EQ(1, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    ASSERT_FALSE(camera.cd().remove_callback(210));
    ASSERT_EQ(1, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    CallbackId id4 = camera.add_runtime_error_callback([](const CameraException &e) {});
    ASSERT_EQ(id4, 3);
    ASSERT_EQ(1, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    CallbackId id5 = camera.cd().add_callback([](const EventCD *ev_begin, const EventCD *ev_end) {});
    ASSERT_EQ(id5, 4);
    ASSERT_EQ(2, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));
}

TEST_F(Camera_Gtest, events_callbacks_add_remove_ext_trigger) {
    write_evt2_raw_data_with_trigger();

    Camera camera = Camera::from_file(tmp_file_);

    ASSERT_EQ(0, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    CallbackId id0 =
        camera.ext_trigger().add_callback([](const EventExtTrigger *ev_begin, const EventExtTrigger *ev_end) {});
    ASSERT_EQ(id0, 0);
    ASSERT_EQ(1, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    CallbackId id1 =
        camera.ext_trigger().add_callback([](const EventExtTrigger *ev_begin, const EventExtTrigger *ev_end) {});
    ASSERT_EQ(id1, 1);
    ASSERT_EQ(2, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    CallbackId id2 =
        camera.ext_trigger().add_callback([](const EventExtTrigger *ev_begin, const EventExtTrigger *ev_end) {});
    ASSERT_EQ(id2, 2);
    ASSERT_EQ(3, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    CallbackId id3 = camera.add_runtime_error_callback([](const CameraException &e) {});
    ASSERT_EQ(id3, 3);
    ASSERT_EQ(3, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    ASSERT_TRUE(camera.ext_trigger().remove_callback(id0));
    ASSERT_EQ(2, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    ASSERT_FALSE(camera.ext_trigger().remove_callback(321));
    ASSERT_EQ(2, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));

    ASSERT_TRUE(camera.remove_runtime_error_callback(id3));
    ASSERT_EQ(2, camera.get_pimpl().index_manager_.counter_map_.tag_count(CallbackTagIds::DECODE_CALLBACK_TAG_ID));
}

TEST_F(Camera_Gtest, multiple_events_callbacks) {
    write_evt2_raw_data();
    Camera camera = Camera::from_file(tmp_file_);

    uint32_t n_cd_decoded = 0;
    camera.cd().add_callback(
        [&n_cd_decoded](const EventCD *ev_begin, const EventCD *ev_end) { n_cd_decoded += ev_end - ev_begin; });

    camera.cd().add_callback(
        [&n_cd_decoded](const EventCD *ev_begin, const EventCD *ev_end) { n_cd_decoded -= ev_end - ev_begin; });

    camera.start();

    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    camera.stop();
    ASSERT_LE(n_cd_decoded, 0);
}

TEST_F_WITHOUT_CAMERA(Camera_Gtest, roi_with_file) {
    //////////////////////////////////////////////////////
    // PURPOSE
    // Checks that you can't get the roi instance when running a camera from a file

    write_evt2_raw_data();
    Camera camera = Camera::from_file(tmp_file_);

    try {
        camera.roi();
        std::cout << "Should have thrown exception..." << std::endl;
        FAIL();
    } catch (CameraException &e) {
        // internal errors should appear to the user as the main category error
        ASSERT_EQ(CameraErrorCode::UnsupportedFeature, e.code().value());
        // but the real error code must appear in the camera exception message
        ASSERT_TRUE(error_message_found_in_exception(UnsupportedFeatureErrors::RoiUnavailable, e));
    }
}

TEST_F_WITH_CAMERA(Camera_Gtest, roi_with_camera) {
    //////////////////////////////////////////////////////
    // PURPOSE
    // Checks that no events are generated outside the ROI

    static constexpr uint16_t roi_x = 10, roi_y = 20, roi_width = 10, roi_height = 10;
    static constexpr size_t max_cd_count = 1000;
    Camera camera;
    if (GtestsParameters::instance().serial.empty()) {
        camera = Camera::from_first_available();
    } else {
        camera = Camera::from_serial(GtestsParameters::instance().serial);
    }
    std::mutex wait_mutex;
    std::condition_variable wait_cond;

    size_t n_cd_counts = 0;
    bool roi_set       = false;
    camera.cd().add_callback(
        [&n_cd_counts, &roi_set, &wait_mutex, &wait_cond](const EventCD *it_begin, const EventCD *it_end) {
            if (!roi_set) {
                return;
            }

            if (n_cd_counts >= max_cd_count) {
                std::unique_lock<std::mutex> lock(wait_mutex);
                wait_cond.notify_all();
                return;
            }

            n_cd_counts += (it_end - it_begin);
            for (; it_begin != it_end; ++it_begin) {
                EXPECT_TRUE((it_begin->x >= roi_x) && (it_begin->x < (roi_x + roi_width)));
                EXPECT_TRUE((it_begin->y >= roi_y) && (it_begin->y < (roi_y + roi_height)));
            }
        });

    ASSERT_TRUE(camera.start());
    ASSERT_NO_THROW(camera.roi().set({roi_x, roi_y, roi_width, roi_height}));
    // wait for the ROI to be set
    std::this_thread::sleep_for(std::chrono::seconds(1));
    roi_set = true;
    {
        std::unique_lock<std::mutex> lock(wait_mutex);
        wait_cond.wait(lock, [&n_cd_counts]() { return n_cd_counts >= max_cd_count; });
    }
    ASSERT_TRUE(camera.stop());
}

TEST_F_WITH_CAMERA(Camera_Gtest, roi_advanced_bitmap_fit_dimension_with_camera) {
    //////////////////////////////////////////////////////
    // PURPOSE
    // Checks that events are generated outside the disabled lines and thus inside the enabled ones

    static constexpr size_t max_cd_count = 1000;
    Camera camera;
    if (GtestsParameters::instance().serial.empty()) {
        camera = Camera::from_first_available();
    } else {
        camera = Camera::from_serial(GtestsParameters::instance().serial);
    }

    std::vector<bool> rows_to_enable(camera.geometry().height(), true), cols_to_enable(camera.geometry().width(), true);

    std::vector<size_t> rows_disabled(100, 0);
    std::vector<size_t> cols_disabled(100, 0);

    std::iota(rows_disabled.begin(), rows_disabled.begin() + 50, 20);  // disabled line from 20 to 69
    std::iota(rows_disabled.begin() + 50, rows_disabled.end(), 124);   // disabled line from 124 to 173
    std::iota(cols_disabled.begin(), cols_disabled.begin() + 50, 100); // disabled cols from 100 to 149
    std::iota(cols_disabled.begin() + 50, cols_disabled.end(), 200);   // disabled cols from 200 to 249

    for (auto row_to_disable : rows_disabled) {
        rows_to_enable[row_to_disable] = false; // disable the line
    }

    for (auto col_to_disable : cols_disabled) {
        cols_to_enable[col_to_disable] = false; // disable the line
    }

    std::mutex wait_mutex;
    std::condition_variable wait_cond;
    size_t n_cd_counts = 0;
    bool roi_set       = false;
    camera.cd().add_callback([&n_cd_counts, &roi_set, &wait_mutex, &wait_cond, &rows_disabled,
                              &cols_disabled](const EventCD *it_begin, const EventCD *it_end) {
        if (!roi_set) {
            return;
        }

        if (n_cd_counts >= max_cd_count) {
            std::unique_lock<std::mutex> lock(wait_mutex);
            wait_cond.notify_all();
            return;
        }

        n_cd_counts += (it_end - it_begin);
        for (; it_begin != it_end; ++it_begin) {
            for (auto row_to_disable : rows_disabled) {
                EXPECT_TRUE((it_begin->y != row_to_disable));
            }

            for (auto col_to_disable : cols_disabled) {
                EXPECT_TRUE((it_begin->x != col_to_disable));
            }
        }
    });

    ASSERT_TRUE(camera.start());
    ASSERT_NO_THROW(camera.roi().set(cols_to_enable, rows_to_enable));
    // wait for the ROI to be set
    std::this_thread::sleep_for(std::chrono::seconds(1));
    roi_set = true;
    {
        std::unique_lock<std::mutex> lock(wait_mutex);
        wait_cond.wait(lock, [&n_cd_counts]() { return n_cd_counts >= max_cd_count; });
    }
    ASSERT_TRUE(camera.stop());
}

TEST_F_WITH_CAMERA(Camera_Gtest, roi_advanced_bitmap_wrong_dimension_with_camera) {
    //////////////////////////////////////////////////////
    // PURPOSE
    // Checks that you can't set ROI from bitmap when these bitmaps don't have the correct dimensions

    Camera camera;
    if (GtestsParameters::instance().serial.empty()) {
        camera = Camera::from_first_available();
    } else {
        camera = Camera::from_serial(GtestsParameters::instance().serial);
    }

    ASSERT_TRUE(camera.start());
    try {
        camera.roi().set({false}, {false}); // bool vector of size 1
        FAIL();
    } catch (CameraException &e) { ASSERT_EQ(e.code().value(), CameraErrorCode::RoiError); }
    ASSERT_TRUE(camera.stop());
}

TEST_F(Camera_Gtest, trigger_out_unsupported_with_file) {
    //////////////////////////////////////////////////////
    // PURPOSE
    // From file trigger out does not exist

    write_evt2_raw_data();
    Camera camera = Camera::from_file(tmp_file_);
    ASSERT_THROW(camera.trigger_out(), CameraException);
    try {
        camera.trigger_out();
        std::cerr << "Expected throw..." << std::endl;
        FAIL();
    } catch (CameraException &e) {
        ASSERT_TRUE(
            error_message_found_in_exception(static_cast<int>(UnsupportedFeatureErrors::TriggerOutUnavailable), e));
    }
}

TEST_F(Camera_Gtest, test_afk_unsupported_on_rawfile) {
    write_evt2_raw_data();
    Camera camera = Camera::from_file(tmp_file_);
    ASSERT_THROW(camera.antiflicker_module(), CameraException);
    ASSERT_THROW(camera.event_trail_filter_module(), CameraException);
}

TEST_F(Camera_Gtest, decode_evt2_data) {
    const auto expected_events = write_evt2_raw_data();
    std::vector<EventCD> received_events;
    Camera camera;
    try {
        camera = Camera::from_file(tmp_file_, FileConfigHints().real_time_playback(false));
    } catch (CameraException &e) { FAIL(); }

    camera.cd().add_callback(
        [&](auto ev_begin, auto ev_end) { received_events.insert(received_events.end(), ev_begin, ev_end); });

    camera.start();

    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    ASSERT_EQ(expected_events.size(), received_events.size());
    timestamp time_shift = -1;

    using SizeType = std::vector<EventCD>::size_type;
    for (SizeType i = 0, i_end = expected_events.size(); i < i_end; ++i) {
        ASSERT_EQ(expected_events[i].x, received_events[i].x);
        ASSERT_EQ(expected_events[i].y, received_events[i].y);
        ASSERT_EQ(expected_events[i].p, received_events[i].p);
        if (time_shift == -1) {
            time_shift = expected_events[i].t - received_events[i].t;
        } else {
            ASSERT_EQ(time_shift, expected_events[i].t - received_events[i].t);
        }
    }
}

TEST_F_WITH_DATASET(Camera_Gtest, decode_evt3_data) {
    // Read the dataset provided
    std::string dataset_file_path = (boost::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" /
                                     "blinking_gen4_with_ext_triggers.raw")
                                        .string();

    Camera camera = Camera::from_file(dataset_file_path, FileConfigHints().real_time_playback(false));

    // Read the file
    uint32_t n_cd_decoded = 0;
    camera.cd().add_callback(
        [&n_cd_decoded](const EventCD *ev_begin, const EventCD *ev_end) { n_cd_decoded += ev_end - ev_begin; });
    uint32_t n_ext_triggers_decoded = 0;
    camera.ext_trigger().add_callback(
        [&n_ext_triggers_decoded](const EventExtTrigger *ev_begin, const EventExtTrigger *ev_end) {
            n_ext_triggers_decoded += ev_end - ev_begin;
        });
    uint32_t n_raw = 0;
    camera.raw_data().add_callback([&n_raw](const uint8_t *, size_t size) { n_raw += size; });

    camera.start();

    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    camera.stop();
    ASSERT_EQ(2003016, n_cd_decoded);
    ASSERT_EQ(82, n_ext_triggers_decoded);
    ASSERT_EQ(11924600, n_raw);
}

#ifdef HAS_HDF5
TEST_F_WITH_DATASET(Camera_Gtest, decode_hdf5_data) {
    // Read the dataset provided
    std::string dataset_file_path = (boost::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" /
                                     "blinking_gen4_with_ext_triggers.hdf5")
                                        .string();

    Camera camera = Camera::from_file(dataset_file_path, FileConfigHints().real_time_playback(false));

    // Read the file
    uint32_t n_cd_decoded = 0;
    camera.cd().add_callback(
        [&n_cd_decoded](const EventCD *ev_begin, const EventCD *ev_end) { n_cd_decoded += ev_end - ev_begin; });
    uint32_t n_ext_triggers_decoded = 0;
    camera.ext_trigger().add_callback(
        [&n_ext_triggers_decoded](const EventExtTrigger *ev_begin, const EventExtTrigger *ev_end) {
            n_ext_triggers_decoded += ev_end - ev_begin;
        });
    ASSERT_THROW(camera.raw_data().add_callback([](const uint8_t *, size_t) {}), CameraException);

    camera.start();

    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    camera.stop();
    ASSERT_EQ(2003016, n_cd_decoded);
    ASSERT_EQ(82, n_ext_triggers_decoded);
}
#endif

TEST_F_WITH_DATASET(Camera_Gtest, decode_evt3_erccounter_data) {
    // Read the dataset provided
    std::string dataset_file_path =
        (boost::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt3_hand.raw").string();

    Camera camera = Camera::from_file(dataset_file_path, FileConfigHints().real_time_playback(false));

    // Read the file
    uint32_t count_in_total  = 0;
    uint32_t count_out_total = 0;
    camera.erc_counter().add_callback(
        [&count_in_total, &count_out_total](const EventERCCounter *ev_begin, const EventERCCounter *ev_end) {
            ASSERT_EQ(1, std::distance(ev_begin, ev_end));
            if (ev_begin->is_output)
                count_out_total += ev_begin->event_count;
            else
                count_in_total += ev_begin->event_count;
        });
    camera.start();

    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    camera.stop();
    ASSERT_EQ(18158913, count_in_total);
    ASSERT_EQ(18095375, count_out_total);
}

TEST_F_WITH_DATASET(Camera_Gtest, offline_streaming_control_not_ready) {
    std::vector<std::string> datasets = get_datasets_paths(DatasetFileType::RAW);

    for (const auto &dataset : datasets) {
        boost::filesystem::remove(dataset + "_index");
        ASSERT_FALSE(boost::filesystem::exists(dataset + "_index"));

        // With this function, index building is not requested, so OSC is never ready
        Metavision::FileConfigHints hints;
        hints.set("index", false);
        Camera camera = Camera::from_file(dataset, hints);

        ASSERT_NO_THROW(camera.offline_streaming_control());

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        ASSERT_FALSE(camera.offline_streaming_control().is_ready());
    }
}

TEST_F_WITH_DATASET(Camera_Gtest, offline_streaming_control_ready_raw_files) {
    std::vector<std::string> datasets = get_datasets_paths(DatasetFileType::RAW);

    for (const auto &dataset : datasets) {
        boost::filesystem::remove(dataset + "_index");
        ASSERT_FALSE(boost::filesystem::exists(dataset + "_index"));

        // With this function, index building is requested, so OSC should be ready
        Camera camera = Camera::from_file(dataset);

        ASSERT_NO_THROW(camera.offline_streaming_control());

        bool ready     = false;
        int max_trials = 1000;
        for (int i = 0; i < max_trials; ++i) {
            if (camera.offline_streaming_control().is_ready()) {
                ready = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        ASSERT_TRUE(ready);
    }
}

TEST_F_WITH_DATASET(Camera_Gtest, offline_streaming_control_ready_hdf5_files) {
    std::vector<std::string> datasets = get_datasets_paths(DatasetFileType::HDF5);

    for (const auto &dataset : datasets) {
        Metavision::FileConfigHints hints = Metavision::FileConfigHints().real_time_playback(false);
        Camera camera                     = Camera::from_file(dataset, hints);
        ASSERT_NO_THROW(camera.offline_streaming_control());
        ASSERT_TRUE(camera.offline_streaming_control().is_ready());
    }
}

TEST_F_WITH_DATASET(Camera_Gtest, offline_streaming_control_seek_range) {
    std::vector<std::string> datasets = get_datasets_paths(DatasetFileType::ALL);

    std::vector<std::pair<Metavision::timestamp, Metavision::timestamp>> ranges;
    // RAW
    ranges.insert(ranges.end(),
                  {{16, 13042000}, {49, 10442000}, {5714, 15441920}, {19, 4194001}, {33, 2782000}, {5, 5834034}});
    // HDF5
    ranges.insert(ranges.end(),
                  {{16, 13042000}, {49, 10442000}, {5714, 15000000}, {19, 4194001}, {33, 2782000}, {5, 5834034}});
    size_t i = 0;
    for (const auto &dataset : datasets) {
        // With this function, index building is requested, so OSC should be ready
        Metavision::FileConfigHints hints = Metavision::FileConfigHints().real_time_playback(false);
        Camera camera                     = Camera::from_file(dataset, hints);

        int max_trials = 1000;
        for (int i = 0; i < max_trials; ++i) {
            if (camera.offline_streaming_control().is_ready()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        EXPECT_EQ(ranges[i].first, camera.offline_streaming_control().get_seek_start_time());
        EXPECT_EQ(ranges[i].second, camera.offline_streaming_control().get_seek_end_time());
        ++i;
    }
}

TEST_F_WITH_DATASET(Camera_Gtest, offline_streaming_control_seeks) {
    std::vector<std::string> datasets = get_datasets_paths(DatasetFileType::ALL);

    std::vector<std::pair<Metavision::timestamp, Metavision::timestamp>> ranges;
    // RAW
    ranges.insert(ranges.end(),
                  {{16, 13042000}, {49, 10442000}, {5714, 15441920}, {19, 4194001}, {33, 2782000}, {5, 5834034}});
    // HDF5
    ranges.insert(ranges.end(),
                  {{16, 13042000}, {49, 10442000}, {5714, 15000000}, {19, 4194001}, {33, 2782000}, {5, 5834034}});
    size_t i = 0;
    for (const auto &dataset : datasets) {
        Metavision::FileConfigHints hints = Metavision::FileConfigHints().real_time_playback(false);
        Camera camera                     = Camera::from_file(dataset, hints);

        std::atomic<bool> decoded{false};
        Metavision::timestamp ts, last_ts = 0;
        camera.cd().add_callback([&](const EventCD *begin, const EventCD *end) {
            if (!decoded && std::abs(begin->t - last_ts) > 1000) {
                // this mean we've seeked
                ts      = begin->t;
                decoded = true;
            }
            last_ts = std::prev(end)->t;
        });

        int max_trials = 1000;
        for (int i = 0; i < max_trials; ++i) {
            if (camera.offline_streaming_control().is_ready()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // a valid seek while the camera is not started must succeed
        ASSERT_TRUE(camera.offline_streaming_control().seek((ranges[i].first + ranges[i].second) / 2));
        camera.start();
        while (!decoded) {}
        ASSERT_GE((ranges[i].first + ranges[i].second) / 2, ts);
        camera.stop();

        // out of bounds seek must fail when camera is not started
        ASSERT_FALSE(camera.offline_streaming_control().seek(-1));
        ASSERT_FALSE(camera.offline_streaming_control().seek(ranges[i].second + 100000));

        camera.start();
        // out of bounds seek must fail when camera is started
        ASSERT_FALSE(camera.offline_streaming_control().seek(-1));
        ASSERT_FALSE(camera.offline_streaming_control().seek(ranges[i].second + 100000));
        camera.stop();

        std::vector<timestamp> targets;
        const timestamp timestamp_step = (ranges[i].second - ranges[i].first) / 10;
        for (uint32_t step = 0; step < 10; ++step) {
            targets.push_back(ranges[i].first + step * timestamp_step);
        }

        using SizeType = std::vector<timestamp>::size_type;
        for (SizeType j = 0; j < targets.size(); ++j) {
            auto target = j % 2 ? targets[targets.size() - j] : targets[j];
            decoded     = false;

            ASSERT_TRUE(camera.offline_streaming_control().seek(target));

            // some seek will reach the end of file
            // make sure we restart the camera before seeking
            if (!camera.is_running()) {
                camera.stop();
                camera.start();
            }

            while (!decoded) {}
            ASSERT_GE(target, ts);
        }
        ++i;
    }
}
