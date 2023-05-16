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

#include "metavision/sdk/base/utils/detail/bitinstructions.h"

using namespace Metavision;

TEST(BitInstructions_Gtest, should_clz_with_32b_input) {
    EXPECT_EQ(clz(uint32_t(0xFFFFFFFF)), 0);
    EXPECT_EQ(clz(uint32_t(0xFFFFF000)), 0);
    EXPECT_EQ(clz(uint32_t(0x0FFFFFFF)), 4);
    EXPECT_EQ(clz(uint32_t(1)), 31);
    EXPECT_EQ(clz(uint32_t(0)), 32);
}

TEST(BitInstructions_Gtest, should_clz_with_64b_input) {
    EXPECT_EQ(clz(uint64_t(1)), 63);
    EXPECT_EQ(clz(uint64_t(1LL << 63)), 0);
    EXPECT_EQ(clz(uint64_t(1LL << 62)), 1);
    EXPECT_EQ(clz(uint64_t(0xFFFFFFFF)), 32);
    EXPECT_EQ(clz(uint64_t(0)), 64);
}

TEST(BitInstructions_Gtest, should_clz_with_signed_types) {
    // EXPECT_EQ(clz<short int>(-1), 0); // Should fail at compile time
    EXPECT_EQ(clz<int>(-1), 0);
    EXPECT_EQ(clz<long int>(-1), 0);
    EXPECT_EQ(clz<long int>(0), Metavision::detail::bit_size<long int>);
    EXPECT_EQ(clz<long long int>(-1), 0);
}

TEST(BitInstructions_Gtest, should_ctz_with_32b_input) {
    EXPECT_EQ(ctz(uint32_t(0xFFFFFFFF)), 0);
    EXPECT_EQ(ctz(uint32_t(0xFFFFF000)), 12);
    EXPECT_EQ(ctz(uint32_t(0x0FFFFFFF)), 0);
    EXPECT_EQ(ctz(uint32_t(1)), 0);
    EXPECT_EQ(ctz(uint32_t(0)), 32);
}

TEST(BitInstructions_Gtest, should_ctz_with_64b_input) {
    EXPECT_EQ(ctz(uint64_t(1)), 0);
    EXPECT_EQ(ctz(uint64_t(1LL << 63)), 63);
    EXPECT_EQ(ctz(uint64_t(1LL << 62)), 62);
    EXPECT_EQ(ctz(uint64_t(0xFFFFFFFF)), 0);
    EXPECT_EQ(ctz(uint64_t(0)), 64);
}

TEST(BitInstructions_Gtest, should_ctz_with_signed_types) {
    // EXPECT_EQ(ctz<short int>(-1), 0); // Should fail at compile time
    EXPECT_EQ(ctz<int>(-1), 0);
    EXPECT_EQ(ctz<long int>(-1), 0);
    EXPECT_EQ(ctz<long int>(0), Metavision::detail::bit_size<long int>);
    EXPECT_EQ(ctz<long long int>(-1), 0);
}
