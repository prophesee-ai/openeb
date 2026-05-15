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
#include "metavision/psee_hw_layer/devices/imx636/imx636_ll_biases.h"

using namespace Metavision;
using namespace Metavision::Test;
using namespace ::testing;

using StringMatcher = Matcher<const std::string &>;

class Imx636_LL_Biases_Fixture : public ::testing::Test {
public:
    DeviceConfig device_conf;
    std::shared_ptr<HW_Register_Mock> hw_register = std::make_shared<HW_Register_Mock>();

    void SetUp() {
        EXPECT_CALL(*hw_register, read_register(StringMatcher(_))).Times(AnyNumber());
    }
};

TEST_F(Imx636_LL_Biases_Fixture, should_get_all_biases) {
    Imx636_LL_Biases imx636_ll_biases(device_conf, hw_register, "prefix/");

    EXPECT_GT(imx636_ll_biases.get_all_biases().size(), 0);
}

TEST_F(Imx636_LL_Biases_Fixture, should_not_saturate_bias_by_default) {
    EXPECT_CALL(*hw_register, write_register(Matcher<const std::string &>(_), _)).Times(AtLeast(6));

    Imx636_LL_Biases imx636_ll_biases(device_conf, hw_register, "prefix/");

    EXPECT_TRUE(imx636_ll_biases.set("bias_diff_on", 49));
    EXPECT_TRUE(imx636_ll_biases.set("bias_diff_on", 50));
    EXPECT_TRUE(imx636_ll_biases.set("bias_diff_on", 51));

    EXPECT_TRUE(imx636_ll_biases.set("bias_diff_off", 49));
    EXPECT_TRUE(imx636_ll_biases.set("bias_diff_off", 50));
    EXPECT_TRUE(imx636_ll_biases.set("bias_diff_off", 51));
}

TEST_F(Imx636_LL_Biases_Fixture, should_not_saturate_bias_on_bypass) {
    EXPECT_CALL(*hw_register, write_register(Matcher<const std::string &>(_), _)).Times(AtLeast(6));

    device_conf.enable_biases_range_check_bypass(true);
    Imx636_LL_Biases imx636_ll_biases(device_conf, hw_register, "prefix/");

    EXPECT_TRUE(imx636_ll_biases.set("bias_diff_on", 49));
    EXPECT_TRUE(imx636_ll_biases.set("bias_diff_on", 50));
    EXPECT_TRUE(imx636_ll_biases.set("bias_diff_on", 51));

    EXPECT_TRUE(imx636_ll_biases.set("bias_diff_off", 49));
    EXPECT_TRUE(imx636_ll_biases.set("bias_diff_off", 50));
    EXPECT_TRUE(imx636_ll_biases.set("bias_diff_off", 51));
}
