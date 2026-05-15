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

#include "metavision/sdk/base/utils/software_info.h"

TEST(SoftwareInfo_GTest, software_info_getters_dev) {
    Metavision::SoftwareInfo si(3, 5, 1, "dev", "branchname", "commithash", "2017119");

    EXPECT_EQ(3, si.get_version_major());
    EXPECT_EQ(5, si.get_version_minor());
    EXPECT_EQ(1, si.get_version_patch());
    EXPECT_EQ("dev", si.get_version_suffix());
    EXPECT_EQ("3.5.1-dev", si.get_version());
    EXPECT_EQ("branchname", si.get_vcs_branch());
    EXPECT_EQ("commithash", si.get_vcs_commit());
    EXPECT_EQ("2017119", si.get_vcs_date());
}

TEST(SoftwareInfo_GTest, software_info_getters_official) {
    Metavision::SoftwareInfo si(3, 5, 1, "", "branchname", "commithash", "2017119");

    EXPECT_EQ(3, si.get_version_major());
    EXPECT_EQ(5, si.get_version_minor());
    EXPECT_EQ(1, si.get_version_patch());
    EXPECT_EQ("", si.get_version_suffix());
    EXPECT_EQ("3.5.1", si.get_version());
    EXPECT_EQ("branchname", si.get_vcs_branch());
    EXPECT_EQ("commithash", si.get_vcs_commit());
    EXPECT_EQ("2017119", si.get_vcs_date());
}
