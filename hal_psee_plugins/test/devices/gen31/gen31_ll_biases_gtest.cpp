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
#include "metavision/psee_hw_layer/devices/gen31/gen31_ll_biases.h"

using namespace Metavision::Test;
using namespace ::testing;

using StringMatcher = Matcher<const std::string &>;

class Gen31_LL_Biases_Fixture : public ::testing::Test {
public:
    Metavision::DeviceConfig device_conf;
    std::shared_ptr<HW_Register_Mock> hw_register = std::make_shared<HW_Register_Mock>();
};

TEST_F(Gen31_LL_Biases_Fixture, should_get_all_biases) {
    Metavision::Gen31_LL_Biases gen31_ll_biases(device_conf, hw_register, "prefix/");

    EXPECT_CALL(*hw_register, read_register(StringMatcher(_))).Times(AnyNumber());
    EXPECT_GT(gen31_ll_biases.get_all_biases().size(), 0);
}

// Encoded CCam3BiasEncoding with values:
//  * bias_diff = 1524
//  * bias_fo = -1
constexpr int encoded_bias_diff_of_1524 = 1358964736;

TEST_F(Gen31_LL_Biases_Fixture, should_not_set_saturated_diff_on_bias_by_default) {
    EXPECT_CALL(*hw_register, read_register(Matcher<const std::string &>(_)))
        .Times(AtLeast(2))
        .WillRepeatedly(Return(encoded_bias_diff_of_1524));

    Metavision::Gen31_LL_Biases gen31_ll_biases(device_conf, hw_register, "prefix/");
    EXPECT_FALSE(gen31_ll_biases.set("bias_diff_on", 1));

    EXPECT_CALL(*hw_register, write_register(StringMatcher(_), _)).Times(AtLeast(1));
    EXPECT_TRUE(gen31_ll_biases.set("bias_diff_on", 1619));
}

TEST_F(Gen31_LL_Biases_Fixture, should_not_set_saturated_diff_off_bias_by_default) {
    EXPECT_CALL(*hw_register, read_register(Matcher<const std::string &>(_)))
        .Times(AtLeast(2))
        .WillRepeatedly(Return(encoded_bias_diff_of_1524));

    Metavision::Gen31_LL_Biases gen31_ll_biases(device_conf, hw_register, "prefix/");
    EXPECT_FALSE(gen31_ll_biases.set("bias_diff_off", 1440));

    EXPECT_CALL(*hw_register, write_register(StringMatcher(_), _)).Times(AtLeast(1));
    EXPECT_TRUE(gen31_ll_biases.set("bias_diff_off", 1439));
}

TEST_F(Gen31_LL_Biases_Fixture, should_not_saturate_bias_on_bypass) {
    EXPECT_CALL(*hw_register, read_register(Matcher<const std::string &>(_)))
        .Times(AnyNumber())
        .WillRepeatedly(Return(encoded_bias_diff_of_1524));
    EXPECT_CALL(*hw_register, write_register(Matcher<const std::string &>(_), _)).Times(AtLeast(4));

    device_conf.enable_biases_range_check_bypass(true);
    Metavision::Gen31_LL_Biases gen31_ll_biases(device_conf, hw_register, "prefix/");

    EXPECT_TRUE(gen31_ll_biases.set("bias_diff_on", 1));
    EXPECT_TRUE(gen31_ll_biases.set("bias_diff_on", 1619));
    EXPECT_TRUE(gen31_ll_biases.set("bias_diff_off", 1439));
    EXPECT_TRUE(gen31_ll_biases.set("bias_diff_off", 1440));
}
