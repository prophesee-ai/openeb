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

#include <filesystem>
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
    MOCK_METHOD(int, get_impl, (const std::string &), (const, override));
    MOCK_METHOD((std::map<std::string, int>), get_all_biases, (), (const, override));
    MOCK_METHOD(bool, get_bias_info_impl, (const std::string &, Metavision::LL_Bias_Info &), (const, override));
#else
#define GMOCK_DOES_NOT_SUPPORT_LAMBDA
    MOCK_METHOD2(set_impl, bool(const std::string &, int));
    MOCK_CONST_METHOD1(get_impl, int(const std::string &));
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

class Dummy_LL_Biases : public Metavision::I_LL_Biases {
public:
    Dummy_LL_Biases(const Metavision::DeviceConfig &device_config, const std::map<std::string, int> &biases_map) :
        Metavision::I_LL_Biases(device_config), biases_map_(biases_map) {}

    ~Dummy_LL_Biases() override {}

    bool set_impl(const std::string &bias_name, int bias_value) override {
        biases_map_[bias_name] = bias_value;
        return true;
    }

    int get_impl(const std::string &bias_name) const override {
        return biases_map_.find(bias_name)->second;
    }

    std::map<std::string, int> get_all_biases() const override {
        return biases_map_;
    }

    bool get_bias_info_impl(const std::string &bias_name, Metavision::LL_Bias_Info &bias_info) const override {
        auto it = biases_map_.find(bias_name);
        if (it == biases_map_.end()) {
            return false;
        }
        bias_info =
            Metavision::LL_Bias_Info(std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), "", true, "");
        return true;
    }

    void set_biases_map(const std::map<std::string, int> &biases_map) {
        biases_map_ = biases_map;
    }

private:
    std::map<std::string, int> biases_map_;
};

class BiasFile_GTest : public Metavision::GTestWithTmpDir {
public:
    void write_file(const std::string &filename, const std::string &contents) {
        std::ofstream file_out(filename);

        ASSERT_TRUE(file_out.is_open());
        file_out << contents;
        file_out.close();
    }

    void compare_biases(const std::map<std::string, int> &expected_biases, const std::map<std::string, int> &biases) {
        ASSERT_EQ(expected_biases.size(), biases.size());
        for (auto it_exp = expected_biases.begin(), it_exp_end = expected_biases.end(), it = biases.begin();
             it_exp != it_exp_end; ++it_exp, ++it) {
            EXPECT_EQ(it_exp->first, it->first);
            EXPECT_EQ(it_exp->second, it->second);
        }
    }

protected:
    virtual void SetUp() override {
        std::map<std::string, int> biases_map = {{"bias_diff", -1}, {"bias_diff_off", -1}, {"bias_diff_on", -1},
                                                 {"bias_fo", -1},   {"bias_hpf", -1},      {"bias_pr", -1},
                                                 {"bias_refr", -1}};
        i_ll_biases_ = std::make_unique<Dummy_LL_Biases>(Metavision::DeviceConfig(), biases_map);
    }

    std::unique_ptr<Metavision::I_LL_Biases> i_ll_biases_;
};


TEST_F(BiasFile_GTest, load_from_file_compatible_with_legacy_format) {
    // GIVEN a bias file with the legacy format
    std::string contents = "% gen 3.1 CD standard biases\n"
                           "% characterization release 1.4\n"
                           "% subsystem_ID 2418019330\n"
                           "% gen 3.1 CD standard biases\n"
                           "% system_ID 28\n"
                           "300   % bias_diff           %   v\n"
                           "222   % bias_diff_off       %   v\n"
                           "385   % bias_diff_on        %   v\n"
                           "1480  % bias_fo             %   i\n"
                           "1450  % bias_hpf            %   i\n"
                           "1250  % bias_pr             %   i\n"
                           "1500  % bias_refr           %   i\n";
    std::string filename = tmpdir_handler_->get_full_path("input.bias");
    write_file(filename, contents);

    // WHEN we set the biases from file
    try {
        i_ll_biases_->load_from_file(filename);
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
    ASSERT_NO_THROW(i_ll_biases_->load_from_file(filename));

    // THEN the biases set in the HAL facility have the values written in the file
    std::map<std::string, int> biases_set      = i_ll_biases_->get_all_biases();
    std::map<std::string, int> expected_biases = {{"bias_diff", 300}, {"bias_diff_off", 222}, {"bias_diff_on", 385},
                                                  {"bias_fo", 1480},  {"bias_hpf", 1450},     {"bias_pr", 1250},
                                                  {"bias_refr", 1500}};

    compare_biases(expected_biases, biases_set);
}

TEST_F(BiasFile_GTest, load_from_file_with_current_format) {
    // GIVEN a bias file like the ones we provide at installation
    std::string contents = "299  % bias_diff\n"
                           "228  % bias_diff_off\n"
                           "370  % bias_diff_on\n"
                           "1507 % bias_fo\n"
                           "1499 % bias_hpf\n"
                           "1250 % bias_pr\n"
                           "1500 % bias_refr\n";
    std::string filename = tmpdir_handler_->get_full_path("input.bias");
    write_file(filename, contents);

    // WHEN we set the biases from file
    ASSERT_NO_THROW(i_ll_biases_->load_from_file(filename));

    // THEN the biases set in the HAL facility have the values written in the file
    std::map<std::string, int> biases_set      = i_ll_biases_->get_all_biases();
    std::map<std::string, int> expected_biases = {{"bias_diff", 299}, {"bias_diff_off", 228}, {"bias_diff_on", 370},
                                                  {"bias_fo", 1507},  {"bias_hpf", 1499},     {"bias_pr", 1250},
                                                  {"bias_refr", 1500}};
    compare_biases(expected_biases, biases_set);
}

TEST_F(BiasFile_GTest, load_from_file_exa) {
    // GIVEN a bias file with values in hexadecimal
    std::string contents = "0x97 % bias_pr\n"
                           "0x17 % bias_fo\n"
                           "0x30 % bias_hpf\n"
                           "0x70 % bias_diff_on\n"
                           "0x45 % bias_diff\n"
                           "0x34 % bias_diff_off\n"
                           "0x2D % bias_refr\n";
    std::string filename = tmpdir_handler_->get_full_path("input.bias");
    write_file(filename, contents);

    // WHEN we set the biases from file
    ASSERT_NO_THROW(i_ll_biases_->load_from_file(filename));

    // THEN the biases set in the HAL facility have the values written in the file
    std::map<std::string, int> biases_set      = i_ll_biases_->get_all_biases();
    std::map<std::string, int> expected_biases = {{"bias_pr", 151},      {"bias_fo", 23},   {"bias_hpf", 48},
                                                  {"bias_diff_on", 112}, {"bias_diff", 69}, {"bias_diff_off", 52},
                                                  {"bias_refr", 45}};
    compare_biases(expected_biases, biases_set);
}

TEST_F(BiasFile_GTest, load_from_file_wrong_extension) {
    // GIVEN a bias file with wrong extension
    std::string filename = tmpdir_handler_->get_full_path("input.txt");
    write_file(filename, "299  % bias_diff");

    try {
        // WHEN we set the biases from the given file
        i_ll_biases_->load_from_file(filename);
        FAIL() << "Expected exception Metavision::HalErrorCode::InvalidArgument";
    } catch (const Metavision::HalException &err) {
        // THEN it throws an exception
        EXPECT_EQ(err.code().value(), Metavision::HalErrorCode::InvalidArgument);
    }
}

TEST_F(BiasFile_GTest, load_from_file_nonexistent_file) {
    // GIVEN a bias file that doesn't exist
    std::string filename = tmpdir_handler_->get_full_path("nonexistent.bias");

    try {
        // WHEN we set the biases from this nonexistent file
        i_ll_biases_->load_from_file(filename);
        FAIL() << "Expected exception Metavision::HalErrorCode::InvalidArgument";
    } catch (const Metavision::HalException &err) {
        // THEN it throws an exception
        EXPECT_EQ(err.code().value(), Metavision::HalErrorCode::InvalidArgument);
    }
}

TEST_F(BiasFile_GTest, load_from_file_wrong_format) {
    // GIVEN a bias file with the wrong format
    std::string contents = "300   bias_diff\n"
                           "222   bias_diff_off\n"
                           "385   bias_diff_on\n"
                           "1480  bias_fo\n"
                           "1450  bias_hpf\n"
                           "1250  bias_pr\n"
                           "1500  bias_refr\n";
    std::string filename = tmpdir_handler_->get_full_path("input.bias");
    write_file(filename, contents);

    try {
        // WHEN we set the biases from file
        i_ll_biases_->load_from_file(filename);
        FAIL() << "Expected exception Metavision::HalErrorCode::InvalidArgument";
    } catch (const Metavision::HalException &err) {
        // THEN it throws an exception
        EXPECT_EQ(err.code().value(), Metavision::HalErrorCode::InvalidArgument);
    }
}

TEST_F(BiasFile_GTest, load_from_file_with_incompatible_biases) {
    // GIVEN a bias file like the ones we provide at installation
    std::string contents = "299  % bias_diff\n"
                           "228  % bias_diff_off\n"
                           "370  % bias_diff_on\n"
                           "1507 % bias_fo_n\n" // this is the incompatible bias
                           "1499 % bias_hpf\n"
                           "1250 % bias_pr\n"
                           "1500 % bias_refr\n";
    std::string filename = tmpdir_handler_->get_full_path("input.bias");
    write_file(filename, contents);

    try {
        // WHEN we set the biases from file
        i_ll_biases_->load_from_file(filename);
        FAIL() << "Expected exception Metavision::HalErrorCode::NonExistingValue";
    } catch (const Metavision::HalException &err) {
        // THEN it throws an exception
        EXPECT_EQ(err.code().value(), Metavision::HalErrorCode::NonExistingValue);
    }
}

TEST_F(BiasFile_GTest, load_from_file_with_two_different_values_for_same_bias) {
    // GIVEN a bias file containing different values for a same bias
    std::string contents = "299  % bias_diff\n"
                           "228  % bias_diff_off\n"
                           "370  % bias_diff_on\n"
                           "1507 % bias_fo\n"
                           "1499 % bias_hpf\n"
                           "1250 % bias_pr\n"
                           "1500 % bias_fo\n"
                           "1500 % bias_refr\n";
    std::string filename = tmpdir_handler_->get_full_path("input.bias");
    write_file(filename, contents);

    try {
        // WHEN we set the biases from file
        i_ll_biases_->load_from_file(filename);
        FAIL() << "Expected exception Metavision::HalErrorCode::InvalidArgument";
    } catch (const Metavision::HalException &err) {
        // THEN it throws an exception
        EXPECT_EQ(err.code().value(), Metavision::HalErrorCode::InvalidArgument);
    }
}

TEST_F(BiasFile_GTest, save_to_file) {
    // GIVEN a Biases instance with given biases
    std::map<std::string, int> biases_map = {{"bias_diff", 299}, {"bias_diff_off", 228}, {"bias_diff_on", 370},
                                             {"bias_fo", 1507},  {"bias_hpf", 1499},     {"bias_pr", 1250},
                                             {"bias_refr", 1500}};
    auto dummy_biases                      = std::make_unique<Dummy_LL_Biases>(Metavision::DeviceConfig(), biases_map);

    // WHEN saving the biases to a file
    std::string filename = tmpdir_handler_->get_full_path("output.bias");
    ASSERT_NO_THROW(dummy_biases->save_to_file(filename));

    // THEN when reading back the file we get the same biases of the original Biases instance
    ASSERT_NO_THROW(i_ll_biases_->load_from_file(filename));
    std::map<std::string, int> biases_set = i_ll_biases_->get_all_biases();
    compare_biases(biases_map, biases_set);
}

TEST_F(BiasFile_GTest, save_to_file_wrong_extension) {
    // GIVEN an output file with wrong extension
    std::string filename = tmpdir_handler_->get_full_path("output.txt");

    try {
        // WHEN saving the biases to a file
        i_ll_biases_->save_to_file(filename);
        FAIL() << "Expected exception Metavision::HalErrorCode::InvalidArgument";
    } catch (const Metavision::HalException &err) {
        // THEN it throws an exception
        EXPECT_EQ(err.code().value(), Metavision::HalErrorCode::InvalidArgument);
    }
}

TEST_F(BiasFile_GTest, save_to_file_when_passing_invalid_filename) {
    // GIVEN an output file that is invalid
    std::string invalid_filename = tmpdir_handler_->get_full_path("output.bias");
    // Invalid because it's an existing directory
    ASSERT_TRUE(std::filesystem::create_directory(invalid_filename));

    try {
        // WHEN saving the biases to the provided file
        i_ll_biases_->save_to_file(invalid_filename);
        FAIL() << "Expected exception Metavision::HalErrorCode::InvalidArgument";
    } catch (const Metavision::HalException &err) {
        // THEN it throws an exception
        EXPECT_EQ(err.code().value(), Metavision::HalErrorCode::InvalidArgument);
    }
}
