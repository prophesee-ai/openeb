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

#include "metavision/utils/gtest/gtest_custom.h"
#include "metavision/sdk/stream/camera_generation.h"
#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/internal/camera_generation_internal.h"

using namespace Metavision;

class CameraGeneration_GTest : public ::testing::Test {
public:
    CameraGeneration_GTest() {}

    virtual ~CameraGeneration_GTest() {}

protected:
    virtual void SetUp() override {}

    virtual void TearDown() override {}

    void run_test(const std::string &file_basename, short version_major_expected, short version_minor_expected) {
        std::filesystem::path dataset_file_path =
            std::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / file_basename;

        Camera camera;

        ASSERT_NO_THROW(camera = Camera::from_file(dataset_file_path));

        auto &camera_generation = camera.generation();
        ASSERT_EQ(version_major_expected, camera_generation.version_major());
        ASSERT_EQ(version_minor_expected, camera_generation.version_minor());
    }
};

TEST_F(CameraGeneration_GTest, construction) {
    std::unique_ptr<CameraGeneration> gen31(CameraGeneration::Private::build(3, 1));
    ASSERT_EQ(3, gen31->version_major());
    ASSERT_EQ(1, gen31->version_minor());

    std::unique_ptr<CameraGeneration> gen40(CameraGeneration::Private::build(4, 0));
    ASSERT_EQ(4, gen40->version_major());
    ASSERT_EQ(0, gen40->version_minor());

    std::unique_ptr<CameraGeneration> gen41(CameraGeneration::Private::build(4, 1));
    ASSERT_EQ(4, gen41->version_major());
    ASSERT_EQ(1, gen41->version_minor());
}

TEST_F(CameraGeneration_GTest, operators) {
    std::unique_ptr<CameraGeneration> gen31(CameraGeneration::Private::build(3, 1));
    std::unique_ptr<CameraGeneration> gen31bis(CameraGeneration::Private::build(3, 1));
    std::unique_ptr<CameraGeneration> gen4(CameraGeneration::Private::build(4, 0));
    std::unique_ptr<CameraGeneration> gen4bis(CameraGeneration::Private::build(4, 0));

    {
        std::vector<std::pair<CameraGeneration *, CameraGeneration *>> same_versions = {{gen31.get(), gen31bis.get()},
                                                                                        {gen4.get(), gen4bis.get()}};

        for (auto &couple : same_versions) {
            EXPECT_TRUE(*couple.first == *couple.first);
            EXPECT_TRUE(*couple.first == *couple.second);
            EXPECT_FALSE(*couple.first != *couple.first);
            EXPECT_FALSE(*couple.first != *couple.second);
            EXPECT_FALSE(*couple.first < *couple.second);
            EXPECT_TRUE(*couple.first <= *couple.second);
            EXPECT_FALSE(*couple.first > *couple.second);
            EXPECT_TRUE(*couple.first >= *couple.second);
        }
    }

    {
        std::vector<std::pair<CameraGeneration *, CameraGeneration *>> ordered_couples = {{gen31.get(), gen4.get()}};

        for (auto &couple : ordered_couples) {
            EXPECT_FALSE(*couple.first == *couple.second);
            EXPECT_TRUE(*couple.first != *couple.second);
            EXPECT_TRUE(*couple.first < *couple.second);
            EXPECT_TRUE(*couple.first <= *couple.second);
            EXPECT_FALSE(*couple.first > *couple.second);
            EXPECT_FALSE(*couple.first >= *couple.second);
            EXPECT_FALSE(*couple.second < *couple.first);
            EXPECT_FALSE(*couple.second <= *couple.first);
            EXPECT_TRUE(*couple.second > *couple.first);
            EXPECT_TRUE(*couple.second >= *couple.first);
        }
    }
}

TEST_F_WITH_DATASET(CameraGeneration_GTest, rawfile_gen31) {
    run_test("gen31_timer.raw", 3, 1);
}

TEST_F_WITH_DATASET(CameraGeneration_GTest, rawfile_gen4_evt2) {
    run_test("gen4_evt2_hand.raw", 4, 0);
}

TEST_F_WITH_DATASET(CameraGeneration_GTest, rawfile_gen4_evt3) {
    run_test("gen4_evt3_hand.raw", 4, 0);
}
