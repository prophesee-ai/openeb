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

#include "metavision/hal/utils/device_config.h"

using namespace Metavision;

TEST(DeviceConfigOptionTest, empty_ctor) {
    DeviceConfigOption opt;
    EXPECT_EQ(DeviceConfigOption::Type::Invalid, opt.type());
}

TEST(DeviceConfigOptionTest, boolean_ctor) {
    DeviceConfigOption opt(true);
    EXPECT_EQ(DeviceConfigOption::Type::Boolean, opt.type());
    EXPECT_EQ(true, opt.get_default_value<bool>());
}

TEST(DeviceConfigOptionTest, boolean_copy_ctor) {
    DeviceConfigOption opt(true);
    DeviceConfigOption opt2(opt);
    EXPECT_EQ(DeviceConfigOption::Type::Boolean, opt2.type());
    EXPECT_EQ(true, opt.get_default_value<bool>());
}

TEST(DeviceConfigOptionTest, boolean_copy_assign_to_boolean_option) {
    DeviceConfigOption opt(true);
    DeviceConfigOption opt2(false);
    opt2 = opt;
    EXPECT_EQ(DeviceConfigOption::Type::Boolean, opt2.type());
    EXPECT_EQ(true, opt2.get_default_value<bool>());
}

TEST(DeviceConfigOptionTest, boolean_copy_assign_to_int_option) {
    DeviceConfigOption opt(true);
    DeviceConfigOption opt2(3, 5, 4);
    opt2 = opt;
    EXPECT_EQ(DeviceConfigOption::Type::Boolean, opt2.type());
    EXPECT_EQ(true, opt2.get_default_value<bool>());
}

TEST(DeviceConfigOptionTest, boolean_copy_assign_to_double_option) {
    DeviceConfigOption opt(true);
    DeviceConfigOption opt2(3.0, 5.0, 4.0);
    opt2 = opt;
    EXPECT_EQ(DeviceConfigOption::Type::Boolean, opt2.type());
    EXPECT_EQ(true, opt2.get_default_value<bool>());
}

TEST(DeviceConfigOptionTest, boolean_copy_assign_to_string_option) {
    DeviceConfigOption opt(true);
    DeviceConfigOption opt2({"a", "b"}, "b");
    opt2 = opt;
    EXPECT_EQ(DeviceConfigOption::Type::Boolean, opt2.type());
    EXPECT_EQ(true, opt2.get_default_value<bool>());
}

TEST(DeviceConfigOptionTest, int_ctor) {
    DeviceConfigOption opt(3, 5, 4);
    EXPECT_EQ(DeviceConfigOption::Type::Int, opt.type());
    EXPECT_EQ(std::make_pair(3, 5), opt.get_range<int>());
}

TEST(DeviceConfigOptionTest, int_ctor_invalid) {
    EXPECT_THROW(DeviceConfigOption opt(3, 5, 12), std::runtime_error);
}

TEST(DeviceConfigOptionTest, int_copy_ctor) {
    DeviceConfigOption opt(3, 5, 4);
    DeviceConfigOption opt2(opt);
    EXPECT_EQ(DeviceConfigOption::Type::Int, opt.type());
    EXPECT_EQ(std::make_pair(3, 5), opt2.get_range<int>());
}

TEST(DeviceConfigOptionTest, int_copy_assign_to_boolean_option) {
    DeviceConfigOption opt(3, 5, 4);
    DeviceConfigOption opt2(false);
    opt2 = opt;
    EXPECT_EQ(DeviceConfigOption::Type::Int, opt2.type());
    EXPECT_EQ(std::make_pair(3, 5), opt2.get_range<int>());
}

TEST(DeviceConfigOptionTest, int_copy_assign_to_int_option) {
    DeviceConfigOption opt(3, 5, 4);
    DeviceConfigOption opt2(6, 8, 7);
    opt2 = opt;
    EXPECT_EQ(DeviceConfigOption::Type::Int, opt2.type());
    EXPECT_EQ(std::make_pair(3, 5), opt2.get_range<int>());
}

TEST(DeviceConfigOptionTest, int_copy_assign_to_double_option) {
    DeviceConfigOption opt(3, 5, 4);
    DeviceConfigOption opt2(3.0, 5.0, 4.0);
    opt2 = opt;
    EXPECT_EQ(DeviceConfigOption::Type::Int, opt2.type());
    EXPECT_EQ(std::make_pair(3, 5), opt2.get_range<int>());
}

TEST(DeviceConfigOptionTest, int_copy_assign_to_string_option) {
    DeviceConfigOption opt(3, 5, 4);
    DeviceConfigOption opt2({"a", "b"}, "b");
    opt2 = opt;
    EXPECT_EQ(DeviceConfigOption::Type::Int, opt2.type());
    EXPECT_EQ(std::make_pair(3, 5), opt2.get_range<int>());
}

TEST(DeviceConfigOptionTest, double_ctor) {
    DeviceConfigOption opt(3.0, 5.0, 4.0);
    EXPECT_EQ(DeviceConfigOption::Type::Double, opt.type());
    EXPECT_EQ(std::make_pair(3.0, 5.0), opt.get_range<double>());
}

TEST(DeviceConfigOptionTest, double_ctor_invalid) {
    EXPECT_THROW(DeviceConfigOption opt(3.0, 5.0, 12.0), std::runtime_error);
}

TEST(DeviceConfigOptionTest, double_copy_ctor) {
    DeviceConfigOption opt(3.0, 5.0, 4.0);
    DeviceConfigOption opt2(opt);
    EXPECT_EQ(DeviceConfigOption::Type::Double, opt2.type());
    EXPECT_EQ(std::make_pair(3.0, 5.0), opt2.get_range<double>());
}

TEST(DeviceConfigOptionTest, double_copy_assign_to_boolean_option) {
    DeviceConfigOption opt(3.0, 5.0, 4.0);
    DeviceConfigOption opt2(false);
    opt2 = opt;
    EXPECT_EQ(DeviceConfigOption::Type::Double, opt2.type());
    EXPECT_EQ(std::make_pair(3.0, 5.0), opt2.get_range<double>());
}

TEST(DeviceConfigOptionTest, double_copy_assign_to_int_option) {
    DeviceConfigOption opt(3.0, 5.0, 4.0);
    DeviceConfigOption opt2(6, 8, 7);
    opt2 = opt;
    EXPECT_EQ(DeviceConfigOption::Type::Double, opt2.type());
    EXPECT_EQ(std::make_pair(3.0, 5.0), opt2.get_range<double>());
}

TEST(DeviceConfigOptionTest, double_copy_assign_to_double_option) {
    DeviceConfigOption opt(3.0, 5.0, 4.0);
    DeviceConfigOption opt2(6.0, 8.0, 7.0);
    opt2 = opt;
    EXPECT_EQ(DeviceConfigOption::Type::Double, opt2.type());
    EXPECT_EQ(std::make_pair(3.0, 5.0), opt2.get_range<double>());
}

TEST(DeviceConfigOptionTest, double_copy_assign_to_string_option) {
    DeviceConfigOption opt(3.0, 5.0, 4.0);
    DeviceConfigOption opt2({"a", "b"}, "b");
    opt2 = opt;
    EXPECT_EQ(DeviceConfigOption::Type::Double, opt2.type());
    EXPECT_EQ(std::make_pair(3.0, 5.0), opt2.get_range<double>());
}

TEST(DeviceConfigOptionTest, string_ctor) {
    DeviceConfigOption opt({"a", "b"}, "b");
    EXPECT_EQ(DeviceConfigOption::Type::String, opt.type());
    EXPECT_EQ(std::vector<std::string>({"a", "b"}), opt.get_values());
}

TEST(DeviceConfigOptionTest, string_ctor_invalid) {
    EXPECT_THROW(DeviceConfigOption opt({"a", "b"}, "c"), std::runtime_error);
}

TEST(DeviceConfigOptionTest, string_copy_ctor) {
    DeviceConfigOption opt({"a", "b"}, "b");
    DeviceConfigOption opt2(opt);
    EXPECT_EQ(DeviceConfigOption::Type::String, opt2.type());
    EXPECT_EQ(std::vector<std::string>({"a", "b"}), opt2.get_values());
}

TEST(DeviceConfigOptionTest, string_copy_assign_to_boolean_option) {
    DeviceConfigOption opt({"a", "b"}, "b");
    DeviceConfigOption opt2(false);
    opt2 = opt;
    EXPECT_EQ(DeviceConfigOption::Type::String, opt2.type());
    EXPECT_EQ(std::vector<std::string>({"a", "b"}), opt2.get_values());
}

TEST(DeviceConfigOptionTest, string_copy_assign_to_int_option) {
    DeviceConfigOption opt({"a", "b"}, "b");
    DeviceConfigOption opt2(6, 8, 7);
    opt2 = opt;
    EXPECT_EQ(DeviceConfigOption::Type::String, opt2.type());
    EXPECT_EQ(std::vector<std::string>({"a", "b"}), opt2.get_values());
}

TEST(DeviceConfigOptionTest, string_copy_assign_to_double_option) {
    DeviceConfigOption opt({"a", "b"}, "b");
    DeviceConfigOption opt2(6.0, 8.0, 7.0);
    opt2 = opt;
    EXPECT_EQ(DeviceConfigOption::Type::String, opt2.type());
    EXPECT_EQ(std::vector<std::string>({"a", "b"}), opt2.get_values());
}

TEST(DeviceConfigOptionTest, string_copy_assign_to_string_option) {
    DeviceConfigOption opt({"a", "b"}, "b");
    DeviceConfigOption opt2({"c", "d"}, "c");
    opt2 = opt;
    EXPECT_EQ(DeviceConfigOption::Type::String, opt2.type());
    EXPECT_EQ(std::vector<std::string>({"a", "b"}), opt2.get_values());
}

TEST(DeviceConfigOptionTest, get_default_value_for_boolean_option) {
    DeviceConfigOption opt(true);
    EXPECT_NO_THROW(opt.get_default_value<bool>());
    EXPECT_THROW(opt.get_default_value<int>(), std::runtime_error);
    EXPECT_THROW(opt.get_default_value<double>(), std::runtime_error);
    EXPECT_THROW(opt.get_default_value<std::string>(), std::runtime_error);
}

TEST(DeviceConfigOptionTest, get_default_value_for_int_option) {
    DeviceConfigOption opt(3, 5, 4);
    EXPECT_THROW(opt.get_default_value<bool>(), std::runtime_error);
    EXPECT_NO_THROW(opt.get_default_value<int>());
    EXPECT_THROW(opt.get_default_value<double>(), std::runtime_error);
    EXPECT_THROW(opt.get_default_value<std::string>(), std::runtime_error);
}

TEST(DeviceConfigOptionTest, get_default_value_for_double_option) {
    DeviceConfigOption opt(3.0, 5.0, 4.0);
    EXPECT_THROW(opt.get_default_value<bool>(), std::runtime_error);
    EXPECT_THROW(opt.get_default_value<int>(), std::runtime_error);
    EXPECT_NO_THROW(opt.get_default_value<double>());
    EXPECT_THROW(opt.get_default_value<std::string>(), std::runtime_error);
}

TEST(DeviceConfigOptionTest, get_default_value_for_string_option) {
    DeviceConfigOption opt({"a", "b"}, "b");
    EXPECT_THROW(opt.get_default_value<bool>(), std::runtime_error);
    EXPECT_THROW(opt.get_default_value<int>(), std::runtime_error);
    EXPECT_THROW(opt.get_default_value<double>(), std::runtime_error);
    EXPECT_NO_THROW(opt.get_default_value<std::string>());
}

TEST(DeviceConfigOptionTest, get_range_for_boolean_option) {
    DeviceConfigOption opt(true);
    EXPECT_THROW(opt.get_range<int>(), std::runtime_error);
    EXPECT_THROW(opt.get_range<double>(), std::runtime_error);
}

TEST(DeviceConfigOptionTest, get_range_for_int_option) {
    DeviceConfigOption opt(3, 5, 4);
    EXPECT_NO_THROW(opt.get_range<int>());
    EXPECT_THROW(opt.get_range<double>(), std::runtime_error);
}

TEST(DeviceConfigOptionTest, get_range_for_double_option) {
    DeviceConfigOption opt(3.0, 5.0, 4.0);
    EXPECT_THROW(opt.get_range<int>(), std::runtime_error);
    EXPECT_NO_THROW(opt.get_range<double>());
}

TEST(DeviceConfigOptionTest, get_range_for_string_option) {
    DeviceConfigOption opt({"a", "b"}, "b");
    EXPECT_THROW(opt.get_range<int>(), std::runtime_error);
    EXPECT_THROW(opt.get_range<double>(), std::runtime_error);
}

TEST(DeviceConfigOptionTest, get_values_for_boolean_option) {
    DeviceConfigOption opt(true);
    EXPECT_THROW(opt.get_values(), std::runtime_error);
}

TEST(DeviceConfigOptionTest, get_values_for_int_option) {
    DeviceConfigOption opt(3, 5, 4);
    EXPECT_THROW(opt.get_values(), std::runtime_error);
}

TEST(DeviceConfigOptionTest, get_values_for_double_option) {
    DeviceConfigOption opt(3.0, 5.0, 4.0);
    EXPECT_THROW(opt.get_values(), std::runtime_error);
}

TEST(DeviceConfigOptionTest, get_values_for_string_option) {
    DeviceConfigOption opt({"a", "b"}, "b");
    EXPECT_NO_THROW(opt.get_values());
}

TEST(DeviceConfigTest, set_format) {
    DeviceConfig config;
    config.set_format("blub");
    EXPECT_EQ("blub", config.format());
    EXPECT_EQ("blub", config.get(DeviceConfig::get_format_key()));
    EXPECT_EQ("blub", config.get<std::string>(DeviceConfig::get_format_key()));
}

TEST(DeviceConfigTest, set_format_via_key) {
    DeviceConfig config;
    config.set(DeviceConfig::get_format_key(), "blub");
    EXPECT_EQ("blub", config.format());
    EXPECT_EQ("blub", config.get(DeviceConfig::get_format_key()));
    EXPECT_EQ("blub", config.get<std::string>(DeviceConfig::get_format_key()));
}

TEST(DeviceConfigTest, enable_biases_range_check_bypass) {
    DeviceConfig config;
    config.enable_biases_range_check_bypass(true);
    EXPECT_TRUE(config.biases_range_check_bypass());
    EXPECT_TRUE(config.get<bool>(DeviceConfig::get_biases_range_check_bypass_key()));
}

TEST(DeviceConfigTest, enable_biases_range_check_bypass_via_key) {
    DeviceConfig config;
    config.set(DeviceConfig::get_biases_range_check_bypass_key(), true);
    EXPECT_TRUE(config.biases_range_check_bypass());
    EXPECT_TRUE(config.get<bool>(DeviceConfig::get_biases_range_check_bypass_key()));
}