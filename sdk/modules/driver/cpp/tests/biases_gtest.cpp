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

#include <memory>
#include <fstream>
#include <boost/filesystem.hpp>

#include "metavision/sdk/driver/biases.h"
#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/utils/gtest/warning_removal_helper.h"
#include "metavision/sdk/driver/camera_exception.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/facilities/i_ll_biases.h"

namespace {

class Mock_LL_Biases : public Metavision::I_LL_Biases {
public:
    Mock_LL_Biases(const Metavision::DeviceConfig &device_config, const std::map<std::string, int> &biases_map) :
        Metavision::I_LL_Biases(device_config), biases_map_(biases_map) {}

    ~Mock_LL_Biases() override {}

    bool set_impl(const std::string &bias_name, int bias_value) override {
        biases_map_[bias_name] = bias_value;
        return true;
    }

    int get_impl(const std::string &bias_name) override {
        return biases_map_[bias_name];
    }

    std::map<std::string, int> get_all_biases() override {
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

} // anonymous namespace

class Biases_GTest : public Metavision::GTestWithTmpDir {
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
        i_ll_biases_ = std::make_unique<Mock_LL_Biases>(Metavision::DeviceConfig(), biases_map);
        biases_.reset(new Metavision::Biases(i_ll_biases_.get()));
    }

    std::unique_ptr<Metavision::I_LL_Biases> i_ll_biases_;
    std::unique_ptr<Metavision::Biases> biases_;
};

TEST_F(Biases_GTest, set_from_file_compatible_with_legacy_format) {
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
    ASSERT_NO_THROW(biases_->set_from_file(filename));

    // THEN the biases set in the HAL facility have the values written in the file
    std::map<std::string, int> biases_set      = i_ll_biases_->get_all_biases();
    std::map<std::string, int> expected_biases = {{"bias_diff", 300}, {"bias_diff_off", 222}, {"bias_diff_on", 385},
                                                  {"bias_fo", 1480},  {"bias_hpf", 1450},     {"bias_pr", 1250},
                                                  {"bias_refr", 1500}};

    compare_biases(expected_biases, biases_set);
}

TEST_F(Biases_GTest, set_from_file_with_current_format) {
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
    ASSERT_NO_THROW(biases_->set_from_file(filename));

    // THEN the biases set in the HAL facility have the values written in the file
    std::map<std::string, int> biases_set      = i_ll_biases_->get_all_biases();
    std::map<std::string, int> expected_biases = {{"bias_diff", 299}, {"bias_diff_off", 228}, {"bias_diff_on", 370},
                                                  {"bias_fo", 1507},  {"bias_hpf", 1499},     {"bias_pr", 1250},
                                                  {"bias_refr", 1500}};
    compare_biases(expected_biases, biases_set);
}

TEST_F(Biases_GTest, set_from_file_exa) {
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
    ASSERT_NO_THROW(biases_->set_from_file(filename));

    // THEN the biases set in the HAL facility have the values written in the file
    std::map<std::string, int> biases_set      = i_ll_biases_->get_all_biases();
    std::map<std::string, int> expected_biases = {{"bias_pr", 151},      {"bias_fo", 23},   {"bias_hpf", 48},
                                                  {"bias_diff_on", 112}, {"bias_diff", 69}, {"bias_diff_off", 52},
                                                  {"bias_refr", 45}};
    compare_biases(expected_biases, biases_set);
}

TEST_F(Biases_GTest, set_from_file_wrong_extension) {
    // GIVEN a bias file with wrong extension
    std::string filename = tmpdir_handler_->get_full_path("input.txt");
    write_file(filename, "299  % bias_diff");

    try {
        // WHEN we set the biases from the given file
        biases_->set_from_file(filename);
        FAIL() << "Expected exception Metavision::CameraErrorCode::WrongExtension";
    } catch (const Metavision::CameraException &err) {
        // THEN it throws an exception
        EXPECT_EQ(err.code().value(), Metavision::CameraErrorCode::WrongExtension);
    }
}

TEST_F(Biases_GTest, set_from_file_nonexistent_file) {
    // GIVEN a bias file that doesn't exist
    std::string filename = tmpdir_handler_->get_full_path("nonexistent.bias");

    try {
        // WHEN we set the biases from this nonexistent file
        biases_->set_from_file(filename);
        FAIL() << "Expected exception Metavision::CameraErrorCode::CouldNotOpenFile";
    } catch (const Metavision::CameraException &err) {
        // THEN it throws an exception
        EXPECT_EQ(err.code().value(), Metavision::CameraErrorCode::CouldNotOpenFile);
    }
}

TEST_F(Biases_GTest, set_from_file_wrong_format) {
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
        biases_->set_from_file(filename);
        FAIL() << "Expected exception Metavision::CameraErrorCode::BiasesError";
    } catch (const Metavision::CameraException &err) {
        // THEN it throws an exception
        EXPECT_EQ(err.code().value(), Metavision::CameraErrorCode::BiasesError);
    }
}

TEST_F(Biases_GTest, set_from_file_with_incompatible_biases) {
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
        biases_->set_from_file(filename);
        FAIL() << "Expected exception Metavision::CameraErrorCode::BiasesError";
    } catch (const Metavision::CameraException &err) {
        // THEN it throws an exception
        EXPECT_EQ(err.code().value(), Metavision::CameraErrorCode::BiasesError);
    }
}

TEST_F(Biases_GTest, set_from_file_with_two_different_values_for_same_bias) {
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
        biases_->set_from_file(filename);
        FAIL() << "Expected exception Metavision::CameraErrorCode::BiasesError";
    } catch (const Metavision::CameraException &err) {
        // THEN it throws an exception
        EXPECT_EQ(err.code().value(), Metavision::CameraErrorCode::BiasesError);
    }
}

TEST_F(Biases_GTest, save_to_file) {
    // GIVEN a Biases instance with given biases
    std::map<std::string, int> biases_map = {{"bias_diff", 299}, {"bias_diff_off", 228}, {"bias_diff_on", 370},
                                             {"bias_fo", 1507},  {"bias_hpf", 1499},     {"bias_pr", 1250},
                                             {"bias_refr", 1500}};
    auto mock_biases                      = std::make_unique<Mock_LL_Biases>(Metavision::DeviceConfig(), biases_map);
    Metavision::Biases biases(mock_biases.get());

    // WHEN saving the biases to a file
    std::string filename = tmpdir_handler_->get_full_path("output.bias");
    ASSERT_NO_THROW(biases.save_to_file(filename));

    // THEN when reading back the file we get the same biases of the original Biases instance
    ASSERT_NO_THROW(biases_->set_from_file(filename));
    std::map<std::string, int> biases_set = i_ll_biases_->get_all_biases();
    compare_biases(biases_map, biases_set);
}

TEST_F(Biases_GTest, save_to_file_wrong_extension) {
    // GIVEN an output file with wrong extension
    std::string filename = tmpdir_handler_->get_full_path("output.txt");

    try {
        // WHEN saving the biases to a file
        biases_->save_to_file(filename);
        FAIL() << "Expected exception Metavision::CameraErrorCode::WrongExtension";
    } catch (const Metavision::CameraException &err) {
        // THEN it throws an exception
        EXPECT_EQ(err.code().value(), Metavision::CameraErrorCode::WrongExtension);
    }
}

TEST_F(Biases_GTest, save_to_file_when_passing_invalid_filename) {
    // GIVEN an output file that is invalid
    std::string invalid_filename = tmpdir_handler_->get_full_path("output.bias");
    // Invalid because it's an existing directory
    ASSERT_TRUE(boost::filesystem::create_directory(invalid_filename));

    try {
        // WHEN saving the biases to the provided file
        biases_->save_to_file(invalid_filename);
        FAIL() << "Expected exception Metavision::CameraErrorCode::CouldNotOpenFile";
    } catch (const Metavision::CameraException &err) {
        // THEN it throws an exception
        EXPECT_EQ(err.code().value(), Metavision::CameraErrorCode::CouldNotOpenFile);
    }
}

TEST_F(Biases_GTest, get_facility) {
    // GIVEN a Biases instance
    // Biases instance already built in the ctor of class Biases_GTest

    // WHEN getting the HAL facility
    auto facility = biases_->get_facility();

    // THEN the facility we got is the same used when building the Biases instance
    ASSERT_EQ(i_ll_biases_.get(), facility);
}
