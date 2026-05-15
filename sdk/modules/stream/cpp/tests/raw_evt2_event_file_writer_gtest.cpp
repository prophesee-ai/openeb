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
#include <iterator>
#include <thread>
#include <gtest/gtest.h>
#include "metavision/sdk/base/utils/generic_header.h"
#include "metavision/sdk/base/utils/log.h"
#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/raw_event_file_reader.h"
#include "metavision/sdk/stream/raw_evt2_event_file_writer.h"
#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/utils/gtest/gtest_custom.h"

using namespace Metavision;

class RAWEvt2EventFileWriter_Gtest : public GTestWithTmpDir {
protected:
    virtual void SetUp() {
        std::string tmp_file = tmpdir_handler_->get_full_path("test.raw");
        tmp_file_path_       = std::filesystem::path(tmp_file);
    }

    std::filesystem::path tmp_file_path_;
};

TEST_F(RAWEvt2EventFileWriter_Gtest, empty_constructor) {
    std::unique_ptr<RAWEvt2EventFileWriter> writer;
    ASSERT_NO_THROW(writer = std::make_unique<RAWEvt2EventFileWriter>(640, 480));
}

TEST_F(RAWEvt2EventFileWriter_Gtest, constructor_invalid) {
    std::unique_ptr<RAWEvt2EventFileWriter> writer;
    const auto filepath = std::filesystem::path(tmpdir_handler_->get_full_path("inexistent_directory")) / "file.raw";
    ASSERT_THROW(writer = std::make_unique<RAWEvt2EventFileWriter>(640, 480, filepath), std::runtime_error);
}

TEST_F(RAWEvt2EventFileWriter_Gtest, constructor) {
    {
        std::unique_ptr<RAWEvt2EventFileWriter> writer;
        ASSERT_NO_THROW(writer = std::make_unique<RAWEvt2EventFileWriter>(640, 480, tmp_file_path_));
    }
    {
        std::ifstream ifs(tmp_file_path_.string());
        ASSERT_TRUE(ifs.is_open());
    }
}

TEST_F(RAWEvt2EventFileWriter_Gtest, not_is_open) {
    std::unique_ptr<RAWEvt2EventFileWriter> writer;
    writer = std::make_unique<RAWEvt2EventFileWriter>(640, 480);
    ASSERT_FALSE(writer->is_open());
}

TEST_F(RAWEvt2EventFileWriter_Gtest, is_open) {
    std::unique_ptr<RAWEvt2EventFileWriter> writer;
    writer = std::make_unique<RAWEvt2EventFileWriter>(640, 480, tmp_file_path_);
    ASSERT_TRUE(writer->is_open());
}

TEST_F(RAWEvt2EventFileWriter_Gtest, open) {
    std::unique_ptr<RAWEvt2EventFileWriter> writer;
    ASSERT_NO_THROW(writer = std::make_unique<RAWEvt2EventFileWriter>(640, 480));
    ASSERT_NO_THROW(writer->open(tmp_file_path_.string()));
}

TEST_F(RAWEvt2EventFileWriter_Gtest, close) {
    std::unique_ptr<RAWEvt2EventFileWriter> writer;
    writer = std::make_unique<RAWEvt2EventFileWriter>(640, 480, tmp_file_path_);
    ASSERT_NO_THROW(writer->close());
}

TEST_F(RAWEvt2EventFileWriter_Gtest, open_close) {
    std::unique_ptr<RAWEvt2EventFileWriter> writer;
    ASSERT_NO_THROW(writer = std::make_unique<RAWEvt2EventFileWriter>(640, 480));
    ASSERT_NO_THROW(writer->open(tmp_file_path_.string()));
    ASSERT_TRUE(writer->is_open());
    ASSERT_NO_THROW(writer->close());
    ASSERT_FALSE(writer->is_open());
    ASSERT_NO_THROW(writer->open(tmp_file_path_.string()));
}

TEST_F_WITH_DATASET(RAWEvt2EventFileWriter_Gtest, constructor_metadata_map) {
    {
        const auto dataset_file_path =
            std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen31_timer.raw";
        std::ifstream ifs(dataset_file_path.string());
        GenericHeader header(ifs);
        auto header_map = header.get_header_map();
        std::unordered_map<std::string, std::string> m(header_map.begin(), header_map.end());
        m["toto"] = "blub";

        std::unique_ptr<RAWEvt2EventFileWriter> writer;
        ASSERT_NO_THROW(writer = std::make_unique<RAWEvt2EventFileWriter>(640, 480, tmp_file_path_, false, m));
    }
    {
        std::ifstream ifs(tmp_file_path_.string());
        GenericHeader header(ifs);
        auto metadata_map = header.get_header_map();
        ASSERT_EQ("blub", metadata_map["toto"]);
    }
}

TEST_F(RAWEvt2EventFileWriter_Gtest, get_path) {
    RAWEvt2EventFileWriter writer(640, 480, tmp_file_path_);
    ASSERT_EQ(tmp_file_path_.string(), writer.get_path());
}

TEST_F_WITH_DATASET(RAWEvt2EventFileWriter_Gtest, add_metadata) {
    {
        RAWEvt2EventFileWriter writer(640, 480, tmp_file_path_);
        writer.add_metadata("toto", "blub");
    }
    {
        std::ifstream ifs(tmp_file_path_.string());
        GenericHeader header(ifs);
        auto metadata_map = header.get_header_map();
        ASSERT_EQ("blub", metadata_map["toto"]);
    }
}

TEST_F_WITH_DATASET(RAWEvt2EventFileWriter_Gtest, add_metadata_flush_raw_fail) {
    (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");

    const std::string expected_output_regex =
#ifdef _WIN32
        "";
#else
        "Unable to modify metadata in RAW";
#endif

    ASSERT_DEATH(
        {
            RAWEvt2EventFileWriter writer(640, 480, tmp_file_path_);

            std::vector<EventCD> events = {EventCD(0, 0, 0, 0)};
            writer.add_events(events.data(), events.data() + 1);
            writer.flush();

            // can't add metadata to RAW once data has been added
            writer.add_metadata("toto", "blub");
        },
        expected_output_regex);
}

TEST_F_WITH_DATASET(RAWEvt2EventFileWriter_Gtest, remove_metadata) {
    {
        RAWEvt2EventFileWriter writer(640, 480, tmp_file_path_);
        writer.add_metadata("toto1", "blub");
        writer.add_metadata("toto2", "blub");
        writer.remove_metadata("toto1");
    }
    {
        std::ifstream ifs(tmp_file_path_.string());
        GenericHeader header(ifs);
        auto metadata_map = header.get_header_map();
        ASSERT_EQ("blub", metadata_map["toto2"]);
        ASSERT_TRUE(metadata_map.find("toto1") == metadata_map.end());
    }
}

TEST_F_WITH_DATASET(RAWEvt2EventFileWriter_Gtest, write_raw_cd_only) {
    std::vector<EventCD> expected_data_cd;
    expected_data_cd.reserve(10000);
    {
        const auto dataset_file_path = std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" /
                                       "blinking_gen4_with_ext_triggers.raw";
        Camera cam =
            Camera::from_file(dataset_file_path.string(), Metavision::FileConfigHints().real_time_playback(false));

        const int sensor_width  = cam.geometry().get_width();
        const int sensor_height = cam.geometry().get_height();
        RAWEvt2EventFileWriter writer(sensor_width, sensor_height, tmp_file_path_, false);

        cam.cd().add_callback([&writer, &expected_data_cd](const EventCD *begin, const EventCD *end) {
            expected_data_cd.insert(expected_data_cd.end(), begin, end);
            writer.add_events(begin, end);
        });

        cam.start();
        while (cam.is_running()) {
            std::this_thread::yield();
        }
        cam.stop();
    }

    {
        Camera cam =
            Camera::from_file(tmp_file_path_.string(), Metavision::FileConfigHints().real_time_playback(false));
        auto data_cd_it = expected_data_cd.cbegin();
        cam.cd().add_callback([&data_cd_it](const EventCD *begin, const EventCD *end) {
            while (begin != end) {
                EXPECT_EQ(begin->t, data_cd_it->t);
                EXPECT_EQ(begin->p, data_cd_it->p);
                EXPECT_EQ(begin->x, data_cd_it->x);
                EXPECT_EQ(begin->y, data_cd_it->y);
                ++begin;
                ++data_cd_it;
            }
        });

        cam.start();
        while (cam.is_running()) {
            std::this_thread::yield();
        }
        cam.stop();
        EXPECT_EQ(expected_data_cd.size(), std::distance(expected_data_cd.cbegin(), data_cd_it));
    }
}

TEST_F_WITH_DATASET(RAWEvt2EventFileWriter_Gtest, write_raw_cd_trigger) {
    std::vector<EventCD> expected_data_cd;
    expected_data_cd.reserve(10000);
    std::vector<EventExtTrigger> expected_data_trigger;
    expected_data_trigger.reserve(1000);
    {
        const auto dataset_file_path = std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" /
                                       "blinking_gen4_with_ext_triggers.raw";
        Camera cam =
            Camera::from_file(dataset_file_path.string(), Metavision::FileConfigHints().real_time_playback(false));

        const int sensor_width  = cam.geometry().get_width();
        const int sensor_height = cam.geometry().get_height();
        RAWEvt2EventFileWriter writer(sensor_width, sensor_height, tmp_file_path_, true);

        cam.cd().add_callback([&writer, &expected_data_cd](const EventCD *begin, const EventCD *end) {
            expected_data_cd.insert(expected_data_cd.end(), begin, end);
            writer.add_events(begin, end);
        });

        cam.ext_trigger().add_callback(
            [&writer, &expected_data_trigger](const EventExtTrigger *begin, const EventExtTrigger *end) {
                expected_data_trigger.insert(expected_data_trigger.end(), begin, end);
                writer.add_events(begin, end);
            });

        cam.start();
        while (cam.is_running()) {
            std::this_thread::yield();
        }
        cam.stop();
    }

    {
        Camera cam =
            Camera::from_file(tmp_file_path_.string(), Metavision::FileConfigHints().real_time_playback(false));
        auto data_cd_it = expected_data_cd.cbegin();
        cam.cd().add_callback([&data_cd_it](const EventCD *begin, const EventCD *end) {
            while (begin != end) {
                ASSERT_EQ(begin->t, data_cd_it->t);
                ASSERT_EQ(begin->p, data_cd_it->p);
                ASSERT_EQ(begin->x, data_cd_it->x);
                ASSERT_EQ(begin->y, data_cd_it->y);
                ++begin;
                ++data_cd_it;
            }
        });
        auto data_trig_it = expected_data_trigger.cbegin();
        cam.ext_trigger().add_callback([&data_trig_it](const EventExtTrigger *begin, const EventExtTrigger *end) {
            while (begin != end) {
                EXPECT_EQ(begin->t, data_trig_it->t);
                EXPECT_EQ(begin->p, data_trig_it->p);
                EXPECT_EQ(begin->id, data_trig_it->id);
                ++begin;
                ++data_trig_it;
            }
        });

        cam.start();
        while (cam.is_running()) {
            std::this_thread::yield();
        }
        cam.stop();
        EXPECT_EQ(expected_data_cd.size(), std::distance(expected_data_cd.cbegin(), data_cd_it));
        EXPECT_EQ(expected_data_trigger.size(), std::distance(expected_data_trigger.cbegin(), data_trig_it));
    }
}

TEST_F_WITH_DATASET(RAWEvt2EventFileWriter_Gtest, write_raw_cd_trigger_1s_max_add_latency) {
    std::vector<EventCD> expected_data_cd;
    expected_data_cd.reserve(10000);
    std::vector<EventExtTrigger> expected_data_trigger;
    expected_data_trigger.reserve(1000);
    {
        const auto dataset_file_path = std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" /
                                       "blinking_gen4_with_ext_triggers.raw";
        Camera cam =
            Camera::from_file(dataset_file_path.string(), Metavision::FileConfigHints().real_time_playback(false));

        const int sensor_width  = cam.geometry().get_width();
        const int sensor_height = cam.geometry().get_height();
        RAWEvt2EventFileWriter writer(sensor_width, sensor_height, tmp_file_path_, true, {}, 1'000'000);

        cam.cd().add_callback([&writer, &expected_data_cd](const EventCD *begin, const EventCD *end) {
            expected_data_cd.insert(expected_data_cd.end(), begin, end);
            writer.add_events(begin, end);
        });

        cam.ext_trigger().add_callback(
            [&writer, &expected_data_trigger](const EventExtTrigger *begin, const EventExtTrigger *end) {
                expected_data_trigger.insert(expected_data_trigger.end(), begin, end);
                writer.add_events(begin, end);
            });

        cam.start();
        while (cam.is_running()) {
            std::this_thread::yield();
        }
        cam.stop();
    }

    {
        Camera cam =
            Camera::from_file(tmp_file_path_.string(), Metavision::FileConfigHints().real_time_playback(false));
        auto data_cd_it = expected_data_cd.cbegin();
        cam.cd().add_callback([&data_cd_it](const EventCD *begin, const EventCD *end) {
            while (begin != end) {
                ASSERT_EQ(begin->t, data_cd_it->t);
                ASSERT_EQ(begin->p, data_cd_it->p);
                ASSERT_EQ(begin->x, data_cd_it->x);
                ASSERT_EQ(begin->y, data_cd_it->y);
                ++begin;
                ++data_cd_it;
            }
        });
        auto data_trig_it = expected_data_trigger.cbegin();
        cam.ext_trigger().add_callback([&data_trig_it](const EventExtTrigger *begin, const EventExtTrigger *end) {
            while (begin != end) {
                EXPECT_EQ(begin->t, data_trig_it->t);
                EXPECT_EQ(begin->p, data_trig_it->p);
                EXPECT_EQ(begin->id, data_trig_it->id);
                ++begin;
                ++data_trig_it;
            }
        });

        cam.start();
        while (cam.is_running()) {
            std::this_thread::yield();
        }
        cam.stop();
        EXPECT_EQ(expected_data_cd.size(), std::distance(expected_data_cd.cbegin(), data_cd_it));
        EXPECT_EQ(expected_data_trigger.size(), std::distance(expected_data_trigger.cbegin(), data_trig_it));
    }
}
