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

#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "metavision/psee_hw_layer/utils/register_map.h"
#include "metavision/psee_hw_layer/utils/regmap_data.h"

using namespace Metavision;
using namespace ::testing;

namespace {
const char path_to_register[] = "path/to/register";
}

static RegmapElement testRegmap[] = {
    // clang-format off
    {R, {{path_to_register, 123}}},
    {F, {{"Garfield", 0, 4, 0b1001}}},
    {F, {{"Cornfield", 4, 8}}},
    // clang-format on
};
static uint32_t testRegmapSize = sizeof(testRegmap) / sizeof(testRegmap[0]);

class RegisterAccessTest : public ::testing::Test {
public:
    RegisterMap regmap;
    RegisterAccessTest() : regmap({std::make_tuple(testRegmap, testRegmapSize, "", 0)}) {}

    MockFunction<void(uint32_t addr, uint32_t val)> write_mock;
    MockFunction<uint32_t(uint32_t addr)> read_mock;

    void SetUp() {
        regmap.set_write_cb(write_mock.AsStdFunction());
        regmap.set_read_cb(read_mock.AsStdFunction());
    }
};

TEST_F(RegisterAccessTest, should_construct_register) {
    RegisterMap::Register reg("blah", 42);
    EXPECT_EQ(reg.get_name(), std::string("blah"));
    EXPECT_EQ(reg.get_address(), 42);
}

TEST_F(RegisterAccessTest, should_get_same_register_by_name_or_address) {
    EXPECT_EQ(regmap[path_to_register], regmap[123]);

    const auto const_regmap = regmap;
    EXPECT_EQ(const_regmap[path_to_register], const_regmap[123]);
}

TEST_F(RegisterAccessTest, should_get_register_field_access) {
    RegisterMap::FieldAccess fieldaccess = regmap[path_to_register]["Garfield"];

    uint32_t bitfield = 0;
    fieldaccess.get_field()->set_default_bitfield_in_value(bitfield);
    EXPECT_EQ(bitfield, 0b1001);
}

TEST_F(RegisterAccessTest, should_read_write_register_field_from_callback) {
    RegisterMap::RegisterAccess reg      = regmap[path_to_register];
    RegisterMap::FieldAccess fieldaccess = reg["Garfield"];

    EXPECT_CALL(write_mock, Call(123, 0xF));
    EXPECT_CALL(read_mock, Call(123)).WillOnce(Return(0xF));
    fieldaccess.write_value(0xF);

    EXPECT_CALL(read_mock, Call(123)).WillOnce(Return(0xFF));
    EXPECT_EQ(fieldaccess.read_value(), 0xF);

    EXPECT_CALL(read_mock, Call(123)).WillOnce(Return(0xFF));
    EXPECT_EQ(reg.read_value(), 0xFF);
}

template<class T>
bool is_const() {
    return std::is_const<T>().value;
}

TEST_F(RegisterAccessTest, should_get_const_accesses_from_const_regmap) {
    const auto const_regmap = regmap;

    EXPECT_TRUE(is_const<decltype(const_regmap["path"])>());
    EXPECT_TRUE(is_const<decltype(const_regmap["path"]["register"])>());
    EXPECT_TRUE(is_const<decltype(const_regmap[0x123]["register"])>());
}
