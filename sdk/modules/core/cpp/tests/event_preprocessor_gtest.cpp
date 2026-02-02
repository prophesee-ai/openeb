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

#include <gtest/gtest.h>
#include <atomic>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/core/preprocessors/event_preprocessor.h"

#include "metavision/sdk/core/preprocessors/diff_processor.h"
#include "metavision/sdk/core/preprocessors/event_cube_processor.h"
#include "metavision/sdk/core/preprocessors/hardware_diff_processor.h"
#include "metavision/sdk/core/preprocessors/hardware_histo_processor.h"
#include "metavision/sdk/core/preprocessors/histo_processor.h"
#include "metavision/sdk/core/preprocessors/time_surface_processor.h"

using EventCD = Metavision::EventCD;

class EventPreprocessor_GTest : public ::testing::Test {
public:
    EventPreprocessor_GTest() {}

    virtual ~EventPreprocessor_GTest() {}

protected:
    virtual void SetUp() override {}

    virtual void TearDown() override {}
};

TEST_F(EventPreprocessor_GTest, invalid_arguments) {
    // clip_value_after_normalization = 0
    ASSERT_THROW(new Metavision::HistoProcessor<EventCD *>(480, 240, 5.f, 0.f), std::invalid_argument);
    // event width = -1
    ASSERT_THROW(new Metavision::HistoProcessor<EventCD *>(-1, 240, 5.f, 1.f), std::invalid_argument);
    // event height = -1
    ASSERT_THROW(new Metavision::HistoProcessor<EventCD *>(480, -1, 5.f, 1.f), std::invalid_argument);
}

TEST_F(EventPreprocessor_GTest, valid_arguments) {
    // check that the processing can be instantiated
    std::unique_ptr<Metavision::EventPreprocessor<EventCD *>> processing(
        new Metavision::HistoProcessor<EventCD *>(240, 120, 5.f, 1.f));
    EXPECT_TRUE(processing);
    processing.reset(new Metavision::DiffProcessor<EventCD *>(240, 120, 5.f, 1.f));
    EXPECT_TRUE(processing);
    processing.reset(new Metavision::EventCubeProcessor<EventCD *>(240, 120, 60, 5, true, 255.f, 1.f));
    EXPECT_TRUE(processing);
    processing.reset(new Metavision::HardwareDiffProcessor<EventCD *>(120, 100, -128, 127, true));
    EXPECT_TRUE(processing);
    processing.reset(new Metavision::HardwareHistoProcessor<EventCD *>(120, 100, 255, 255));
    EXPECT_TRUE(processing);
    processing.reset(new Metavision::TimeSurfaceProcessor<EventCD *>(120, 100));
    EXPECT_TRUE(processing);
}
