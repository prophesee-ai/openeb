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

#include "metavision/sdk/stream/synced_camera_streams_slicer.h"
#include "metavision/utils/gtest/gtest_custom.h"

using namespace Metavision;

namespace fs = std::filesystem;

class SyncedCameraStreamSlicerTest : public ::testing::Test {
protected:
    void SetUp() final {
        const auto dataset_dir = GtestsParameters::instance().dataset_dir;
        master_record_path_    = (fs::path(dataset_dir) / "openeb/synced/recording_master.raw").string();
        slave_1_record_path_   = (fs::path(dataset_dir) / "openeb/synced/recording_slave_0.raw").string();
        slave_2_record_path_   = (fs::path(dataset_dir) / "openeb/synced/recording_slave_1.raw").string();
    }

    std::string master_record_path_;
    std::string slave_1_record_path_;
    std::string slave_2_record_path_;
};

TEST_F(SyncedCameraStreamSlicerTest, from_files_returns_valid_iterator) {
    // GIVEN a master record and two slave records
    Camera master_camera = Camera::from_file(master_record_path_);
    std::vector<Camera> slave_cameras;
    slave_cameras.emplace_back(Camera::from_file(slave_1_record_path_));
    slave_cameras.emplace_back(Camera::from_file(slave_2_record_path_));

    // WHEN we create a slicer from these files
    SyncedCameraStreamsSlicer slicer(std::move(master_camera), std::move(slave_cameras));

    // THEN the iterator is valid
    auto begin = slicer.begin();
    auto end   = slicer.end();

    ASSERT_NE(begin, end);
}

TEST_F(SyncedCameraStreamSlicerTest, from_files_returns_valid_slices) {
    // GIVEN
    // - a master record and two slave records
    // - a slicing condition based on the number of events and the number of us

    static constexpr int kNevents = 20000;
    static constexpr int kNus     = 20000;

    Camera master_camera = Camera::from_file(master_record_path_);
    std::vector<Camera> slave_cameras;
    slave_cameras.emplace_back(Camera::from_file(slave_1_record_path_));
    slave_cameras.emplace_back(Camera::from_file(slave_2_record_path_));

    auto slicing_condition = SyncedCameraStreamsSlicer::SliceCondition::make_n_us(kNus);

    // WHEN we create a slicer from these files based on the N us condition
    SyncedCameraStreamsSlicer slicer(std::move(master_camera), std::move(slave_cameras), slicing_condition);

    // THEN
    // - the master slice is created every N us (except for the last slice)
    // - the slave slices don't last longer than the master slice
    for (const auto &slice : slicer) {
        ASSERT_TRUE(slice.status == Detail::ReslicingConditionStatus::MET_N_US ||
                    slice.status == Detail::ReslicingConditionStatus::MET_AUTOMATIC);

        if (slice.status == Detail::ReslicingConditionStatus::MET_N_US) {
            ASSERT_TRUE(slice.t % kNus == 0);

            for (const auto &slave_slice : slice.slave_events) {
                if (slave_slice->empty())
                    continue;

                ASSERT_TRUE(slave_slice->back().t < slice.t);
            }
        }
    }

    // WHEN we create a slicer from these files based on the N events condition
    master_camera = Camera::from_file(master_record_path_);
    slave_cameras.clear();
    slave_cameras.emplace_back(Camera::from_file(slave_1_record_path_));
    slave_cameras.emplace_back(Camera::from_file(slave_2_record_path_));

    slicing_condition = SyncedCameraStreamsSlicer::SliceCondition::make_n_events(kNevents);
    slicer = SyncedCameraStreamsSlicer(std::move(master_camera), std::move(slave_cameras), slicing_condition);

    // THEN
    // - the master slice is created every N events (except for the last slice)
    // - the slave slices don't last longer than the master slice
    for (const auto &slice : slicer) {
        ASSERT_TRUE(slice.status == Detail::ReslicingConditionStatus::MET_N_EVENTS ||
                    slice.status == Detail::ReslicingConditionStatus::MET_AUTOMATIC);

        if (slice.status == Detail::ReslicingConditionStatus::MET_N_EVENTS) {
            ASSERT_EQ(slice.master_events->size(), kNevents);
        }

        for (const auto &slave_slice : slice.slave_events) {
            if (slave_slice->empty())
                continue;

            ASSERT_TRUE(slave_slice->back().t < slice.t);
        }
    }
}