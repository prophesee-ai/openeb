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
#include <gmock/gmock.h>
#include <memory>

#include "gtest_utils/i_hw_register_mock.h"
#include "metavision/hal/utils/device_config.h"
#include "metavision/psee_hw_layer/devices/gen41/gen41_ll_biases.h"

using namespace Metavision;
using namespace Metavision::Test;
using namespace testing;

class Gen41_LL_Biases_Fixture : public ::testing::Test {
public:
    DeviceConfig device_conf;
    std::shared_ptr<HW_Register_Mock> hw_register = std::make_shared<HW_Register_Mock>();
};

TEST_F(Gen41_LL_Biases_Fixture, should_get_all_biases) {
    EXPECT_CALL(*hw_register, read_register(Matcher<const std::string &>(_))).Times(AnyNumber());

    Gen41_LL_Biases gen41_ll_biases(device_conf, hw_register, "prefix/");
    EXPECT_GT(gen41_ll_biases.get_all_biases().size(), 0);
}

TEST_F(Gen41_LL_Biases_Fixture, should_not_set_saturated_bias_diff_on_by_default) {
    EXPECT_CALL(*hw_register, read_register(Matcher<const std::string &>(_)))
        .Times(AtLeast(2))
        .WillRepeatedly(Return(50));

    EXPECT_CALL(*hw_register, write_register("prefix/bias/bias_diff_on", _)).Times(AtLeast(1));

    Gen41_LL_Biases gen41_ll_biases(device_conf, hw_register, "prefix/");
    EXPECT_FALSE(gen41_ll_biases.set("bias_diff_on", 51));
    EXPECT_TRUE(gen41_ll_biases.set("bias_diff_on", 75));
}

TEST_F(Gen41_LL_Biases_Fixture, should_not_set_saturated_bias_diff_off_by_default) {
    EXPECT_CALL(*hw_register, read_register(Matcher<const std::string &>(_)))
        .Times(AtLeast(2))
        .WillRepeatedly(Return(50));

    EXPECT_CALL(*hw_register, write_register("prefix/bias/bias_diff_off", _)).Times(AtLeast(1));

    Gen41_LL_Biases gen41_ll_biases(device_conf, hw_register, "prefix/");
    EXPECT_FALSE(gen41_ll_biases.set("bias_diff_off", 49));
    EXPECT_TRUE(gen41_ll_biases.set("bias_diff_off", 35));
}

TEST_F(Gen41_LL_Biases_Fixture, should_not_saturate_bias_on_bypass) {
    EXPECT_CALL(*hw_register, write_register(Matcher<const std::string &>(_), _)).Times(AtLeast(4));

    device_conf.enable_biases_range_check_bypass(true);
    Gen41_LL_Biases gen41_ll_biases(device_conf, hw_register, "prefix/");

    EXPECT_TRUE(gen41_ll_biases.set("bias_diff_on", 51));
    EXPECT_TRUE(gen41_ll_biases.set("bias_diff_on", 75));
    EXPECT_TRUE(gen41_ll_biases.set("bias_diff_off", 49));
    EXPECT_TRUE(gen41_ll_biases.set("bias_diff_off", 35));
}
