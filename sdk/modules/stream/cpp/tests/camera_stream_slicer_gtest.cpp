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
#include <gtest/gtest.h>

#include "metavision/sdk/stream/camera_stream_slicer.h"
#include "metavision/utils/gtest/gtest_custom.h"

using namespace Metavision;

namespace fs = std::filesystem;

class CameraStreamSlicerTest : public ::testing::Test {
protected:
    void SetUp() final {
        dataset_dir_ = GtestsParameters::instance().dataset_dir;
    }

    std::string dataset_dir_;
};

TEST_F(CameraStreamSlicerTest, from_file_returns_valid_iterator) {
    // GIVEN a record file
    const auto record_path = fs::path(dataset_dir_) / "openeb" / "gen4_evt3_hand.raw";

    // WHEN we create a slicer from this file
    auto camera = Camera::from_file(record_path.string());
    CameraStreamSlicer slicer(std::move(camera));

    // THEN the iterator is valid
    auto begin = slicer.begin();
    auto end   = slicer.end();

    ASSERT_NE(begin, end);
}

TEST_F(CameraStreamSlicerTest, from_file_returns_valid_slices) {
    // GIVEN
    // - a record file
    // - a slicing condition based on the number of events and the number of us
    static constexpr int kNevents = 20000;
    static constexpr int kNus     = 20000;

    const auto record_path = fs::path(dataset_dir_) / "openeb" / "gen4_evt3_hand.raw";

    // WHEN we create a slicer from this file based on the N us condition
    auto camera            = Camera::from_file(record_path.string());
    auto slicing_condition = CameraStreamSlicer::SliceCondition::make_n_us(kNus);
    CameraStreamSlicer slicer(std::move(camera), slicing_condition);

    for (const auto &slice : slicer) {
        ASSERT_TRUE(slice.status == Detail::ReslicingConditionStatus::MET_N_US ||
                    slice.status == Detail::ReslicingConditionStatus::MET_AUTOMATIC);

        if (slice.status == Detail::ReslicingConditionStatus::MET_N_US) {
            ASSERT_TRUE(slice.t % kNus == 0);
        }
    }

    // WHEN we create a slicer from this file based on the N events condition
    slicing_condition = CameraStreamSlicer::SliceCondition::make_n_events(kNevents);
    camera            = Camera::from_file(record_path.string());
    slicer            = CameraStreamSlicer(std::move(camera), slicing_condition);

    // THEN the slice is created every N events (except for the last slice)
    for (const auto &slice : slicer) {
        ASSERT_TRUE(slice.status == Detail::ReslicingConditionStatus::MET_N_EVENTS ||
                    slice.status == Detail::ReslicingConditionStatus::MET_AUTOMATIC);

        if (slice.status == Detail::ReslicingConditionStatus::MET_N_EVENTS) {
            ASSERT_EQ(slice.events->size(), kNevents);
        }
    }
}

TEST_F(CameraStreamSlicerTest, throw_if_camera_is_already_running) {
    const auto record_path = fs::path(dataset_dir_) / "openeb" / "gen4_evt3_hand.raw";
    auto camera            = Camera::from_file(record_path.string());
    camera.start();

    ASSERT_THROW(CameraStreamSlicer slicer(std::move(camera)), std::runtime_error);
}

TEST_F_WITH_ANY_CAMERA(CameraStreamSlicerTest, from_device_returns_valid_iterator) {
    // GIVEN a slicer created from a device
    auto camera = Camera::from_first_available();
    CameraStreamSlicer slicer(std::move(camera));

    auto begin = slicer.begin();
    auto end   = slicer.end();

    // THEN the iterator is valid
    ASSERT_NE(begin, end);
}

TEST_F_WITH_ANY_CAMERA(CameraStreamSlicerTest, from_device_returns_valid_slices) {
    // GIVEN a slicing condition based on the number of events and the number of us

    static constexpr std::uint8_t kNSlicesToTest = 10;
    static constexpr int kNevents                = 20000;
    static constexpr int kNus                    = 20000;

    // WHEN we create a slicer from a device based on the N us condition
    auto slicing_condition = CameraStreamSlicer::SliceCondition::make_n_us(kNus);
    auto camera            = Camera::from_first_available();

    {
        CameraStreamSlicer slicer(std::move(camera), slicing_condition);

        std::uint8_t num_slices = 0;
        for (const auto &slice : slicer) {
            ASSERT_TRUE(slice.status == Detail::ReslicingConditionStatus::MET_N_US ||
                        slice.status == Detail::ReslicingConditionStatus::MET_AUTOMATIC);

            if (slice.status == Detail::ReslicingConditionStatus::MET_N_US) {
                ASSERT_TRUE(slice.t % kNus == 0);
            }

            if (++num_slices >= kNSlicesToTest) {
                break;
            }
        }
    }

    // WHEN we create a slicer from a device based on the N events condition
    slicing_condition = CameraStreamSlicer::SliceCondition::make_n_events(kNevents);
    camera            = Camera::from_first_available();

    {
        CameraStreamSlicer slicer(std::move(camera), slicing_condition);
        std::uint8_t num_slices = 0;
        for (const auto &slice : slicer) {
            ASSERT_TRUE(slice.status == Detail::ReslicingConditionStatus::MET_N_EVENTS ||
                        slice.status == Detail::ReslicingConditionStatus::MET_AUTOMATIC);

            if (slice.status == Detail::ReslicingConditionStatus::MET_N_EVENTS) {
                ASSERT_EQ(slice.events->size(), kNevents);
            }

            if (++num_slices >= kNSlicesToTest) {
                break;
            }
        }
    }
}