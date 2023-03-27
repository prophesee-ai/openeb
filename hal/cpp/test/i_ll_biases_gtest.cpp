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

#include <map>
#include <sstream>
#include <gmock/gmock.h>

#include "metavision/hal/device/device.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/facilities/i_ll_biases.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/utils/gtest/gtest_custom.h"

#include "utils/device_test.h"

class I_LL_Biases_GTest : public Metavision::testing::DeviceTest {
public:
    void on_opened_device(Metavision::Device &device) override {
        ll_biases_ = device.get_facility<Metavision::I_LL_Biases>();
        ASSERT_NE(nullptr, ll_biases_);
    }

protected:
    Metavision::I_LL_Biases *ll_biases_{nullptr};
};

TEST_F_WITH_CAMERA(I_LL_Biases_GTest, get_existing_biases) {
    for (auto &bias : ll_biases_->get_all_biases()) {
        ASSERT_NO_THROW(ll_biases_->get(bias.first));
    }
}

TEST_F_WITH_CAMERA(I_LL_Biases_GTest, get_non_existing_bias) {
    ASSERT_THROW(ll_biases_->get("bad_bias_name"), Metavision::HalException);
}

TEST_F_WITH_CAMERA(I_LL_Biases_GTest, set_non_existing_bias) {
    ASSERT_THROW(ll_biases_->set("bad_bias_name", 0), Metavision::HalException);
}

TEST_F_WITH_CAMERA(I_LL_Biases_GTest, get_non_existing_bias_info) {
    Metavision::LL_Bias_Info bi;
    ASSERT_THROW(ll_biases_->get_bias_info("bad_bias_name", bi), Metavision::HalException);
}

TEST_F_WITH_CAMERA(I_LL_Biases_GTest, get_existing_biases_info) {
    for (auto &bias : ll_biases_->get_all_biases()) {
        Metavision::LL_Bias_Info bi;
        ASSERT_NO_THROW(ll_biases_->get_bias_info(bias.first, bi));
    }
}

TEST_F_WITH_CAMERA(I_LL_Biases_GTest, set_non_modifiable_bias) {
    for (auto &bias : ll_biases_->get_all_biases()) {
        Metavision::LL_Bias_Info bi;
        ll_biases_->get_bias_info(bias.first, bi);
        if (!bi.is_modifiable()) {
            ASSERT_THROW(ll_biases_->set(bias.first, 0), Metavision::HalException);
        }
    }
}

TEST_F_WITH_CAMERA(I_LL_Biases_GTest, set_existing_biases_valid_value) {
    for (auto &bias : ll_biases_->get_all_biases()) {
        Metavision::LL_Bias_Info bi;
        ll_biases_->get_bias_info(bias.first, bi);
        if (bi.is_modifiable()) {
            ASSERT_NO_THROW(ll_biases_->set(bias.first, bi.get_bias_range().first));
        }
    }
}

TEST_F_WITH_CAMERA(I_LL_Biases_GTest, set_existing_biases_outofrange_value) {
    for (auto &bias : ll_biases_->get_all_biases()) {
        Metavision::LL_Bias_Info bi;
        ll_biases_->get_bias_info(bias.first, bi);
        if (bi.is_modifiable()) {
            ASSERT_THROW(ll_biases_->set(bias.first, bi.get_bias_range().second + 1), Metavision::HalException);
        }
    }
}

namespace {

class Mock_LL_Biases : public Metavision::I_LL_Biases {
public:
    Mock_LL_Biases(const Metavision::DeviceConfig &device_config = Metavision::DeviceConfig()) :
        Metavision::I_LL_Biases(device_config) {}
    ~Mock_LL_Biases() override {}

#if defined(MOCK_METHOD)
    MOCK_METHOD(bool, set_impl, (const std::string &, int), (override));
    MOCK_METHOD(int, get_impl, (const std::string &), (override));
    MOCK_METHOD((std::map<std::string, int>), get_all_biases, (), (override));
    MOCK_METHOD(bool, get_bias_info_impl, (const std::string &, Metavision::LL_Bias_Info &), (const, override));
#else
#define GMOCK_DOES_NOT_SUPPORT_LAMBDA
    MOCK_METHOD2(set_impl, bool(const std::string &, int));
    MOCK_METHOD1(get_impl, int(const std::string &));
    MOCK_METHOD0(get_all_biases, std::map<std::string, int>());
    MOCK_CONST_METHOD2(get_bias_info_impl, bool(const std::string &, Metavision::LL_Bias_Info &));
#endif

private:
    std::map<std::string, int> biases_map_;
};

#ifdef GMOCK_DOES_NOT_SUPPORT_LAMBDA
bool MakeModifiableLLBiasInfoSameRange(testing::Unused, Metavision::LL_Bias_Info &bias_info) {
    bias_info = Metavision::LL_Bias_Info(0, 10, "", true, "");
    return true;
}
bool MakeModifiableLLBiasInfoDiffRange(testing::Unused, Metavision::LL_Bias_Info &bias_info) {
    bias_info = Metavision::LL_Bias_Info(-100, 100, 0, 10, "", true, "");
    return true;
}
#endif
} // namespace

using namespace testing;

TEST(Mock_LL_Biases_GTest, get_valid_bias_info) {
    const std::string bias_name = "bias";
    Mock_LL_Biases ll_biases;
    EXPECT_CALL(ll_biases, get_bias_info_impl(bias_name, _)).WillOnce(Return(true));
    Metavision::LL_Bias_Info info;
    EXPECT_NO_THROW(ll_biases.get_bias_info(bias_name, info));
}

TEST(Mock_LL_Biases_GTest, get_invalid_bias_info) {
    const std::string bias_name = "bias";
    Mock_LL_Biases ll_biases;
    EXPECT_CALL(ll_biases, get_bias_info_impl(bias_name, _)).WillOnce(Return(false));
    Metavision::LL_Bias_Info info;
    EXPECT_THROW(ll_biases.get_bias_info(bias_name, info), Metavision::HalException);
}

TEST(Mock_LL_Biases_GTest, get_valid_bias) {
    const std::string bias_name = "bias";
    Mock_LL_Biases ll_biases;
    EXPECT_CALL(ll_biases, get_bias_info_impl(bias_name, _)).WillOnce(Return(true));
    EXPECT_CALL(ll_biases, get_impl(bias_name)).WillOnce(Return(1));
    EXPECT_NO_THROW(EXPECT_EQ(1, ll_biases.get("bias")));
}

TEST(Mock_LL_Biases_GTest, get_invalid_bias) {
    const std::string bias_name = "bias";
    Mock_LL_Biases ll_biases;
    EXPECT_CALL(ll_biases, get_bias_info_impl(bias_name, _)).WillOnce(Return(false));
    EXPECT_CALL(ll_biases, get_impl(bias_name)).Times(0);
    EXPECT_THROW(ll_biases.get(bias_name), Metavision::HalException);
}

TEST(Mock_LL_Biases_GTest, set_valid_bias) {
    const std::string bias_name = "bias";
    const int bias_value        = 3;
    Mock_LL_Biases ll_biases;
    EXPECT_CALL(ll_biases, get_bias_info_impl(bias_name, _))
        .WillRepeatedly(
#ifdef GMOCK_DOES_NOT_SUPPORT_LAMBDA
            Invoke(MakeModifiableLLBiasInfoSameRange)
#else
            [](auto, auto &bias_info) {
                bias_info = Metavision::LL_Bias_Info(0, 10, "", true, "");
                return true;
            }
#endif
        );
    EXPECT_CALL(ll_biases, set_impl(bias_name, bias_value)).WillOnce(Return(true));
    EXPECT_CALL(ll_biases, get_impl(bias_name)).WillOnce(Return(3));
    EXPECT_NO_THROW(EXPECT_TRUE(ll_biases.set(bias_name, bias_value)));
    EXPECT_EQ(3, ll_biases.get(bias_name));
}

TEST(Mock_LL_Biases_GTest, set_invalid_bias) {
    const std::string bias_name = "bias";
    const int bias_value        = 3;
    Mock_LL_Biases ll_biases;
    EXPECT_CALL(ll_biases, get_bias_info_impl(bias_name, _)).WillOnce(Return(false));
    EXPECT_CALL(ll_biases, set_impl(bias_name, bias_value)).Times(0);
    EXPECT_THROW(ll_biases.set(bias_name, bias_value), Metavision::HalException);
}

TEST(Mock_LL_Biases_GTest, set_no_bypass_bias_range_check_same_range) {
    const std::string bias_name = "bias";
    const int bias_value1 = -50, bias_value2 = 50;
    Mock_LL_Biases ll_biases;
    EXPECT_CALL(ll_biases, get_bias_info_impl(bias_name, _))
        .WillRepeatedly(
#ifdef GMOCK_DOES_NOT_SUPPORT_LAMBDA
            Invoke(MakeModifiableLLBiasInfoSameRange)
#else
            [](auto, auto &bias_info) {
                bias_info = Metavision::LL_Bias_Info(0, 10, "", true, "");
                return true;
            }
#endif
        );
    EXPECT_CALL(ll_biases, set_impl(bias_name, bias_value1)).Times(0);
    EXPECT_CALL(ll_biases, set_impl(bias_name, bias_value2)).Times(0);
    EXPECT_THROW(ll_biases.set(bias_name, bias_value1), Metavision::HalException);
    EXPECT_THROW(ll_biases.set(bias_name, bias_value2), Metavision::HalException);
    Metavision::LL_Bias_Info info;
    EXPECT_TRUE(ll_biases.get_bias_info(bias_name, info));
    EXPECT_EQ(0, info.get_bias_range().first);
    EXPECT_EQ(10, info.get_bias_range().second);
    EXPECT_EQ(0, info.get_bias_allowed_range().first);
    EXPECT_EQ(10, info.get_bias_allowed_range().second);
    EXPECT_EQ(0, info.get_bias_recommended_range().first);
    EXPECT_EQ(10, info.get_bias_recommended_range().second);
}

TEST(Mock_LL_Biases_GTest, set_bypass_bias_range_check_same_range) {
    const std::string bias_name = "bias";
    const int bias_value1 = -50, bias_value2 = 50;
    Metavision::DeviceConfig config;
    config.enable_biases_range_check_bypass(true);
    Mock_LL_Biases ll_biases(config);
    EXPECT_CALL(ll_biases, get_bias_info_impl(bias_name, _))
        .WillRepeatedly(
#ifdef GMOCK_DOES_NOT_SUPPORT_LAMBDA
            Invoke(MakeModifiableLLBiasInfoSameRange)
#else
            [](auto, auto &bias_info) {
                bias_info = Metavision::LL_Bias_Info(0, 10, "", true, "");
                return true;
            }
#endif
        );
    EXPECT_CALL(ll_biases, set_impl(bias_name, bias_value1)).WillOnce(Return(true));
    EXPECT_CALL(ll_biases, set_impl(bias_name, bias_value2)).WillOnce(Return(true));
    EXPECT_CALL(ll_biases, get_impl(bias_name)).WillOnce(Return(bias_value1)).WillOnce(Return(bias_value2));
    EXPECT_NO_THROW(ll_biases.set(bias_name, bias_value1));
    EXPECT_EQ(bias_value1, ll_biases.get(bias_name));
    EXPECT_NO_THROW(ll_biases.set(bias_name, bias_value2));
    EXPECT_EQ(bias_value2, ll_biases.get(bias_name));
    Metavision::LL_Bias_Info info;
    EXPECT_TRUE(ll_biases.get_bias_info(bias_name, info));
    EXPECT_EQ(0, info.get_bias_range().first);
    EXPECT_EQ(10, info.get_bias_range().second);
    EXPECT_EQ(0, info.get_bias_allowed_range().first);
    EXPECT_EQ(10, info.get_bias_allowed_range().second);
    EXPECT_EQ(0, info.get_bias_recommended_range().first);
    EXPECT_EQ(10, info.get_bias_recommended_range().second);
}

TEST(Mock_LL_Biases_GTest, set_no_bypass_bias_range_check_diff_range) {
    const std::string bias_name = "bias";
    const int bias_value1 = -50, bias_value2 = 50;
    Mock_LL_Biases ll_biases;
    EXPECT_CALL(ll_biases, get_bias_info_impl(bias_name, _))
        .WillRepeatedly(
#ifdef GMOCK_DOES_NOT_SUPPORT_LAMBDA
            Invoke(MakeModifiableLLBiasInfoDiffRange)
#else
            [](auto, auto &bias_info) {
                bias_info = Metavision::LL_Bias_Info(-100, 100, 0, 10, "", true, "");
                return true;
            }
#endif
        );
    EXPECT_CALL(ll_biases, set_impl(bias_name, bias_value1)).Times(0);
    EXPECT_CALL(ll_biases, set_impl(bias_name, bias_value2)).Times(0);
    EXPECT_THROW(ll_biases.set(bias_name, bias_value1), Metavision::HalException);
    EXPECT_THROW(ll_biases.set(bias_name, bias_value2), Metavision::HalException);
    Metavision::LL_Bias_Info info;
    EXPECT_TRUE(ll_biases.get_bias_info(bias_name, info));
    EXPECT_EQ(0, info.get_bias_range().first);
    EXPECT_EQ(10, info.get_bias_range().second);
    EXPECT_EQ(-100, info.get_bias_allowed_range().first);
    EXPECT_EQ(100, info.get_bias_allowed_range().second);
    EXPECT_EQ(0, info.get_bias_recommended_range().first);
    EXPECT_EQ(10, info.get_bias_recommended_range().second);
}

TEST(Mock_LL_Biases_GTest, set_bypass_bias_range_check_diff_range) {
    const std::string bias_name = "bias";
    const int bias_value1 = -50, bias_value2 = 50;
    Metavision::DeviceConfig config;
    config.enable_biases_range_check_bypass(true);
    Mock_LL_Biases ll_biases(config);
    EXPECT_CALL(ll_biases, get_bias_info_impl(bias_name, _))
        .WillRepeatedly(
#ifdef GMOCK_DOES_NOT_SUPPORT_LAMBDA
            Invoke(MakeModifiableLLBiasInfoDiffRange)
#else
            [](auto, auto &bias_info) {
                bias_info = Metavision::LL_Bias_Info(-100, 100, 0, 10, "", true, "");
                return true;
            }
#endif
        );
    EXPECT_CALL(ll_biases, set_impl(bias_name, bias_value1)).WillOnce(Return(true));
    EXPECT_CALL(ll_biases, set_impl(bias_name, bias_value2)).WillOnce(Return(true));
    EXPECT_CALL(ll_biases, get_impl(bias_name)).WillOnce(Return(bias_value1)).WillOnce(Return(bias_value2));
    EXPECT_NO_THROW(ll_biases.set(bias_name, bias_value1));
    EXPECT_EQ(bias_value1, ll_biases.get(bias_name));
    EXPECT_NO_THROW(ll_biases.set(bias_name, bias_value2));
    EXPECT_EQ(bias_value2, ll_biases.get(bias_name));
    Metavision::LL_Bias_Info info;
    EXPECT_TRUE(ll_biases.get_bias_info(bias_name, info));
    EXPECT_EQ(-100, info.get_bias_range().first);
    EXPECT_EQ(100, info.get_bias_range().second);
    EXPECT_EQ(-100, info.get_bias_allowed_range().first);
    EXPECT_EQ(100, info.get_bias_allowed_range().second);
    EXPECT_EQ(0, info.get_bias_recommended_range().first);
    EXPECT_EQ(10, info.get_bias_recommended_range().second);
}
