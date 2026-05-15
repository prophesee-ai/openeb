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

#include "metavision/sdk/core/utils/counter_map.h"

namespace Metavision {

class CounterMap_GTest : public ::testing::Test {
public:
    CounterMap_GTest() {}

    virtual ~CounterMap_GTest() {}

protected:
    virtual void SetUp() override {}

    virtual void TearDown() override {}

    CounterMap<std::string> counter_map_;
};

TEST_F(CounterMap_GTest, tag_untag) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Tagger method tag untag works as expected: tagging increments, untagging decrements
    //

    const std::string KEY1      = "TEST1";
    const std::string KEY2      = "TEST2";
    const std::string KEY3      = "TEST3";
    const std::string RANDOMKEY = "RANDOM";

    ASSERT_EQ(1, counter_map_.tag(KEY1));
    ASSERT_EQ(1, counter_map_.tag(KEY2));
    ASSERT_EQ(2, counter_map_.tag(KEY2));
    ASSERT_EQ(1, counter_map_.tag(KEY3));
    ASSERT_EQ(2, counter_map_.tag(KEY3));
    ASSERT_EQ(3, counter_map_.tag(KEY3));
    ASSERT_EQ(1, counter_map_.tag_count(KEY1));
    ASSERT_EQ(2, counter_map_.tag_count(KEY2));
    ASSERT_EQ(3, counter_map_.tag_count(KEY3));

    // decrease twice. First time goes from 1 to 0
    // second time remain at 0 as it should not exist anymore
    ASSERT_EQ(0, counter_map_.untag(KEY1));
    ASSERT_EQ(0, counter_map_.tag_count(KEY1));
    ASSERT_EQ(0, counter_map_.untag(KEY1));
    ASSERT_EQ(0, counter_map_.tag_count(KEY1));

    ASSERT_EQ(1, counter_map_.untag(KEY2));
    ASSERT_EQ(1, counter_map_.tag_count(KEY2));

    ASSERT_EQ(2, counter_map_.untag(KEY3));
    ASSERT_EQ(2, counter_map_.tag_count(KEY3));

    ASSERT_EQ(0, counter_map_.tag_count(RANDOMKEY));
}

} // namespace Metavision
