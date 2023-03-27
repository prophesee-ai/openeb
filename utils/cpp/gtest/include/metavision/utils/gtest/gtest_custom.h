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

#ifndef METAVISION_UTILS_GTEST_CUSTOM_H
#define METAVISION_UTILS_GTEST_CUSTOM_H

#include <gtest/gtest.h>
#include <string>

namespace Metavision {

struct GtestsCameraParameters {
    static auto constexpr ANY = "any";

    GtestsCameraParameters(const std::string &integrator_name = ANY, const std::string &gen = ANY,
                           const std::string &camera_board = ANY) :
        integrator_(integrator_name), generation_(gen), board_(camera_board) {}

    GtestsCameraParameters &integrator(const std::string &integrator_name) {
        integrator_ = integrator_name;
        return *this;
    }

    GtestsCameraParameters &generation(const std::string &gen) {
        generation_ = gen;
        return *this;
    }

    GtestsCameraParameters &board(const std::string &camera_board) {
        board_ = camera_board;
        return *this;
    }

    std::string integrator_{ANY};
    std::string generation_{ANY};
    std::string board_{ANY};
};

struct GtestsParameters {
    static GtestsParameters &instance() {
        static GtestsParameters p;
        return p;
    }
    bool run_test_with_camera    = false;
    bool run_test_without_camera = false;
    std::string serial;
    bool is_remote_camera = false;
    GtestsCameraParameters camera_param{GtestsCameraParameters::ANY, GtestsCameraParameters::ANY,
                                        GtestsCameraParameters::ANY};

    std::string dataset_dir{""};
};

inline void parse(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            std::cout << "Custom flags :" << std::endl;
            std::cout << "\033[0;32m  --with-camera\033[m" << std::endl;
            std::cout << "    Run tests that require a device to be plugged in." << std::endl;
            std::cout << "\033[0;32m  --serial SERIAL\033[m" << std::endl;
            std::cout << "    if --with-camera open the camera with the SERIAL specified." << std::endl;
            std::cout << "\033[0;32m  --integrator INTEGRATOR\033[m" << std::endl;
            std::cout << "    if --with-camera, it specify the camera's integrator name (default value : Prophesee)."
                      << std::endl;
            std::cout << "\033[0;32m  --generation GENERATION\033[m" << std::endl;
            std::cout << "    if --with-camera, it specify the sensor's generation (ex : 3.0, 3.1, 4.0, 4.1)."
                      << std::endl;
            std::cout << "\033[0;32m  --board BOARD\033[m" << std::endl;
            std::cout << "    if --with-camera, it specify the camera's board (ex : fx3, cx3, evk2)." << std::endl;
            std::cout << "\033[0;32m  --remote\033[m" << std::endl;
            std::cout << "    if --with-camera, it specify the camera is a remote one, otherwise a local one."
                      << std::endl;
            std::cout << "\033[0;32m  --without-camera\033[m" << std::endl;
            std::cout << "    Run tests that require a device NOT to be plugged in." << std::endl;
            std::cout << "\033[0;32m  --dataset-dir DATASET_DIR\033[m" << std::endl;
            std::cout << "    Run tests that require access to dataset directory DATASET_DIR" << std::endl;
            std::cout << std::endl;

        } else if (arg == "--with-camera") {
            GtestsParameters::instance().run_test_with_camera = true;
        } else if (arg == "--without-camera") {
            GtestsParameters::instance().run_test_without_camera = true;
        } else if (arg == "--serial") {
            if (i + 1 < argc) {
                GtestsParameters::instance().serial = argv[i + 1];
                ++i;
            }
        } else if (arg == "--integrator") {
            if (i + 1 < argc) {
                GtestsParameters::instance().camera_param.integrator_ = argv[i + 1];
                ++i;
            }
        } else if (arg == "--generation") {
            if (i + 1 < argc) {
                GtestsParameters::instance().camera_param.generation_ = argv[i + 1];
                ++i;
            }
        } else if (arg == "--board") {
            if (i + 1 < argc) {
                GtestsParameters::instance().camera_param.board_ = argv[i + 1];
                ++i;
            }
        } else if (arg == "--remote") {
            GtestsParameters::instance().is_remote_camera = true;
        } else if (arg == "--dataset-dir") {
            if (i + 1 < argc) {
                GtestsParameters::instance().dataset_dir = argv[i + 1];
                ++i;
            }
        }
    }
    if (GtestsParameters::instance().run_test_with_camera && GtestsParameters::instance().run_test_without_camera) {
        std::cerr << "Two options that can not be used simultaneously were given :" << std::endl;
        std::cerr << "  --with-camera is incompatible with --without-camera" << std::endl;
        exit(1);
    }
    if (GtestsParameters::instance().run_test_with_camera &&
        GtestsParameters::instance().camera_param.integrator_ == GtestsCameraParameters::ANY) {
        GtestsParameters::instance().camera_param.integrator_ = "Prophesee";
    }
}

} // namespace Metavision

inline Metavision::GtestsCameraParameters camera_param() {
    return Metavision::GtestsCameraParameters();
}

inline std::vector<Metavision::GtestsCameraParameters> camera_params(const Metavision::GtestsCameraParameters &param) {
    return {param};
}

template<typename... Tn>
inline std::vector<Metavision::GtestsCameraParameters> camera_params(const Metavision::GtestsCameraParameters &param,
                                                                     const Tn &...params) {
    std::vector<Metavision::GtestsCameraParameters> vparams = camera_params(params...);
    vparams.push_back(param);
    return vparams;
}

namespace testing {
#define CUSTOM_TEST_IMPL_CLASS_NAME(test_case_name, test_name) test_case_name##_##test_name##_CustomTest

#define DECLARE_CUSTOM_TEST_IMPL_CLASS(test_case_name, test_name)                            \
    struct CUSTOM_TEST_IMPL_CLASS_NAME(test_case_name, test_name) : public ::testing::Test { \
        void TestBody();                                                                     \
    }

#define GET_MACRO(_1, _2, _3, NAME, ...) NAME
#define EXPAND_ARG(x) x

#define TEST_WITH_SPECIFIC_CAMERA(test_case_name, test_name, camera_param_vector)                                   \
    DECLARE_CUSTOM_TEST_IMPL_CLASS(test_case_name, test_name);                                                      \
    TEST(test_case_name, test_name) {                                                                               \
        bool to_run(false);                                                                                         \
        std::vector<std::string> needed_options;                                                                    \
        if (Metavision::GtestsParameters::instance().run_test_with_camera) {                                        \
            auto &input_camera_param = Metavision::GtestsParameters::instance().camera_param;                       \
            for (auto &camera_param : camera_param_vector) {                                                        \
                if ((camera_param.integrator_ == Metavision::GtestsCameraParameters::ANY ||                         \
                     camera_param.integrator_ == input_camera_param.integrator_) &&                                 \
                    (camera_param.generation_ == Metavision::GtestsCameraParameters::ANY ||                         \
                     camera_param.generation_ == input_camera_param.generation_) &&                                 \
                    (camera_param.board_ == Metavision::GtestsCameraParameters::ANY ||                              \
                     camera_param.board_ == input_camera_param.board_)) {                                           \
                    to_run = true;                                                                                  \
                    break;                                                                                          \
                } else {                                                                                            \
                    std::string needed_option("");                                                                  \
                    if (camera_param.integrator_ != Metavision::GtestsCameraParameters::ANY) {                      \
                        needed_option += " --integrator " + camera_param.integrator_;                               \
                    }                                                                                               \
                    if (camera_param.generation_ != Metavision::GtestsCameraParameters::ANY) {                      \
                        needed_option += " --generation " + camera_param.generation_;                               \
                    }                                                                                               \
                    if (camera_param.board_ != Metavision::GtestsCameraParameters::ANY) {                           \
                        needed_option += " --board " + camera_param.board_;                                         \
                    }                                                                                               \
                    needed_options.push_back(needed_option);                                                        \
                }                                                                                                   \
            }                                                                                                       \
        }                                                                                                           \
        if (to_run) {                                                                                               \
            CUSTOM_TEST_IMPL_CLASS_NAME(test_case_name, test_name)().TestBody();                                    \
        } else {                                                                                                    \
            std::string reason_not_run = "Skipping test requiring connected camera, to enable re-run with option "; \
            if (needed_options.size() > 0) {                                                                        \
                auto it = needed_options.begin(), it_end = needed_options.end();                                    \
                reason_not_run += "'--with-camera" + *it + "'";                                                     \
                ++it;                                                                                               \
                for (; it != it_end; ++it) {                                                                        \
                    reason_not_run += " or '--with-camera" + *it + "'";                                             \
                }                                                                                                   \
            } else {                                                                                                \
                reason_not_run += "'--with-camera'";                                                                \
            }                                                                                                       \
            std::cout << "\033[0;34m" << reason_not_run << "\033[m" << std::endl;                                   \
        }                                                                                                           \
        SUCCEED();                                                                                                  \
    }                                                                                                               \
    void CUSTOM_TEST_IMPL_CLASS_NAME(test_case_name, test_name)::TestBody()

#define TEST_WITH_ANY_CAMERA(test_case_name, test_name) \
    TEST_WITH_SPECIFIC_CAMERA(test_case_name, test_name, {camera_param()})

#define TEST_WITH_CAMERA(...) \
    EXPAND_ARG(GET_MACRO(__VA_ARGS__, TEST_WITH_SPECIFIC_CAMERA, TEST_WITH_ANY_CAMERA)(__VA_ARGS__))

#define TEST_WITHOUT_CAMERA(test_fixture, test_name)                               \
    DECLARE_CUSTOM_TEST_IMPL_CLASS(test_fixture, test_name);                       \
    TEST(test_fixture, test_name) {                                                \
        if (Metavision::GtestsParameters::instance().run_test_without_camera) {    \
            CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name)().TestBody();     \
        } else {                                                                   \
            std::cout << "\033[0;34mSkipping test requiring no connected camera, " \
                         "to enable re-run with --without-camera.\033[m"           \
                      << std::endl;                                                \
        }                                                                          \
        SUCCEED();                                                                 \
    }                                                                              \
    void CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name)::TestBody()

#define TEST_WITH_DATASET(test_fixture, test_name)                             \
    DECLARE_CUSTOM_TEST_IMPL_CLASS(test_fixture, test_name);                   \
    TEST(test_fixture, test_name) {                                            \
        if (!Metavision::GtestsParameters::instance().dataset_dir.empty()) {   \
            CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name)().TestBody(); \
        } else {                                                               \
            std::cout << "\033[0;34mSkipping test requiring dataset, "         \
                         "to enable re-run with --dataset-dir DATASET.\033[m"  \
                      << std::endl;                                            \
        }                                                                      \
        SUCCEED();                                                             \
    }                                                                          \
    void CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name)::TestBody()

#define DECLARE_CUSTOM_TEST_IMPL_CLASS_F(test_fixture, test_name, test_parent)         \
    struct CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name) : public test_parent { \
        void TestBody();                                                               \
        friend class GTEST_TEST_CLASS_NAME_(test_fixture, test_name);                  \
    }

#define TEST_F_WITH_SPECIFIC_CAMERA(test_fixture, test_name, camera_param_vector)                                   \
    DECLARE_CUSTOM_TEST_IMPL_CLASS_F(test_fixture, test_name, test_fixture);                                        \
    GTEST_TEST_(test_fixture, test_name, ::testing::Test, ::testing::internal::GetTypeId<test_fixture>()) {         \
        bool to_run(false);                                                                                         \
        std::vector<std::string> needed_options;                                                                    \
        if (Metavision::GtestsParameters::instance().run_test_with_camera) {                                        \
            auto &input_camera_param = Metavision::GtestsParameters::instance().camera_param;                       \
            for (auto &camera_param : camera_param_vector) {                                                        \
                if ((camera_param.integrator_ == Metavision::GtestsCameraParameters::ANY ||                         \
                     camera_param.integrator_ == input_camera_param.integrator_) &&                                 \
                    (camera_param.generation_ == Metavision::GtestsCameraParameters::ANY ||                         \
                     camera_param.generation_ == input_camera_param.generation_) &&                                 \
                    (camera_param.board_ == Metavision::GtestsCameraParameters::ANY ||                              \
                     camera_param.board_ == input_camera_param.board_)) {                                           \
                    to_run = true;                                                                                  \
                    break;                                                                                          \
                } else {                                                                                            \
                    std::string needed_option("");                                                                  \
                    if (camera_param.integrator_ != Metavision::GtestsCameraParameters::ANY) {                      \
                        needed_option += " --integrator " + camera_param.integrator_;                               \
                    }                                                                                               \
                    if (camera_param.generation_ != Metavision::GtestsCameraParameters::ANY) {                      \
                        needed_option += " --generation " + camera_param.generation_;                               \
                    }                                                                                               \
                    if (camera_param.board_ != Metavision::GtestsCameraParameters::ANY) {                           \
                        needed_option += " --board " + camera_param.board_;                                         \
                    }                                                                                               \
                    needed_options.push_back(needed_option);                                                        \
                }                                                                                                   \
            }                                                                                                       \
        }                                                                                                           \
        if (to_run) {                                                                                               \
            CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name) test;                                              \
            test.SetUp();                                                                                           \
            test.TestBody();                                                                                        \
            test.TearDown();                                                                                        \
        } else {                                                                                                    \
            std::string reason_not_run = "Skipping test requiring connected camera, to enable re-run with option "; \
            if (needed_options.size() > 0) {                                                                        \
                auto it = needed_options.begin(), it_end = needed_options.end();                                    \
                reason_not_run += "'--with-camera" + *it + "'";                                                     \
                ++it;                                                                                               \
                for (; it != it_end; ++it) {                                                                        \
                    reason_not_run += " or '--with-camera" + *it + "'";                                             \
                }                                                                                                   \
            } else {                                                                                                \
                reason_not_run += "'--with-camera'";                                                                \
            }                                                                                                       \
            std::cout << "\033[0;34m" << reason_not_run << "\033[m" << std::endl;                                   \
        }                                                                                                           \
        SUCCEED();                                                                                                  \
    }                                                                                                               \
    void CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name)::TestBody()

#define TEST_F_WITH_ANY_CAMERA(test_case_name, test_name) \
    TEST_F_WITH_SPECIFIC_CAMERA(test_case_name, test_name, {camera_param()})
#define TEST_F_WITH_CAMERA(...) \
    EXPAND_ARG(GET_MACRO(__VA_ARGS__, TEST_F_WITH_SPECIFIC_CAMERA, TEST_F_WITH_ANY_CAMERA)(__VA_ARGS__))

#define TEST_F_WITHOUT_CAMERA(test_fixture, test_name)                                                      \
    DECLARE_CUSTOM_TEST_IMPL_CLASS_F(test_fixture, test_name, test_fixture);                                \
    GTEST_TEST_(test_fixture, test_name, ::testing::Test, ::testing::internal::GetTypeId<test_fixture>()) { \
        if (Metavision::GtestsParameters::instance().run_test_without_camera) {                             \
            CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name) test;                                      \
            test.SetUp();                                                                                   \
            test.TestBody();                                                                                \
            test.TearDown();                                                                                \
        } else {                                                                                            \
            std::cout << "\033[0;34mSkipping test requiring no connected camera, "                          \
                         "to enable re-run with --without-camera.\033[m"                                    \
                      << std::endl;                                                                         \
        }                                                                                                   \
        SUCCEED();                                                                                          \
    }                                                                                                       \
    void CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name)::TestBody()

#define TEST_F_WITH_DATASET(test_fixture, test_name)                                                        \
    DECLARE_CUSTOM_TEST_IMPL_CLASS_F(test_fixture, test_name, test_fixture);                                \
    GTEST_TEST_(test_fixture, test_name, ::testing::Test, ::testing::internal::GetTypeId<test_fixture>()) { \
        if (!Metavision::GtestsParameters::instance().dataset_dir.empty()) {                                \
            CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name) test;                                      \
            test.SetUp();                                                                                   \
            test.TestBody();                                                                                \
            test.TearDown();                                                                                \
        } else {                                                                                            \
            std::cout << "\033[0;34mSkipping test requiring dataset, "                                      \
                         "to enable re-run with --dataset-dir DATASET.\033[m"                               \
                      << std::endl;                                                                         \
        }                                                                                                   \
        SUCCEED();                                                                                          \
    }                                                                                                       \
    void CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name)::TestBody()

#define DECLARE_CUSTOM_TYPED_TEST_IMPL_CLASS(test_case_name, test_name)                                 \
    /* forward declare for friend class declaration */                                                  \
    template<typename T>                                                                                \
    class GTEST_TEST_CLASS_NAME_(test_case_name, test_name);                                            \
    /* actual test class definition */                                                                  \
    template<typename TypeParamT>                                                                       \
    struct CUSTOM_TEST_IMPL_CLASS_NAME(test_case_name, test_name) : public test_case_name<TypeParamT> { \
        void TestBody();                                                                                \
        typedef test_case_name<TypeParamT> TestFixture;                                                 \
        typedef TypeParamT TypeParam;                                                                   \
        friend class GTEST_TEST_CLASS_NAME_(test_case_name, test_name)<TypeParamT>;                     \
    }

#define TYPED_TEST_WITH_SPECIFIC_CAMERA(test_fixture, test_name, camera_param_vector)                               \
    DECLARE_CUSTOM_TYPED_TEST_IMPL_CLASS(test_fixture, test_name);                                                  \
    TYPED_TEST(test_fixture, test_name) {                                                                           \
        bool to_run(false);                                                                                         \
        std::vector<std::string> needed_options;                                                                    \
        if (Metavision::GtestsParameters::instance().run_test_with_camera) {                                        \
            auto &input_camera_param = Metavision::GtestsParameters::instance().camera_param;                       \
            for (auto &camera_param : camera_param_vector) {                                                        \
                if ((camera_param.integrator_ == Metavision::GtestsCameraParameters::ANY ||                         \
                     camera_param.integrator_ == input_camera_param.integrator_) &&                                 \
                    (camera_param.generation_ == Metavision::GtestsCameraParameters::ANY ||                         \
                     camera_param.generation_ == input_camera_param.generation_) &&                                 \
                    (camera_param.board_ == Metavision::GtestsCameraParameters::ANY ||                              \
                     camera_param.board_ == input_camera_param.board_)) {                                           \
                    to_run = true;                                                                                  \
                    break;                                                                                          \
                } else {                                                                                            \
                    std::string needed_option("");                                                                  \
                    if (camera_param.integrator_ != Metavision::GtestsCameraParameters::ANY) {                      \
                        needed_option += " --integrator " + camera_param.integrator_;                               \
                    }                                                                                               \
                    if (camera_param.generation_ != Metavision::GtestsCameraParameters::ANY) {                      \
                        needed_option += " --generation " + camera_param.generation_;                               \
                    }                                                                                               \
                    if (camera_param.board_ != Metavision::GtestsCameraParameters::ANY) {                           \
                        needed_option += " --board " + camera_param.board_;                                         \
                    }                                                                                               \
                    needed_options.push_back(needed_option);                                                        \
                }                                                                                                   \
            }                                                                                                       \
        }                                                                                                           \
        if (to_run) {                                                                                               \
            CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name)<TypeParam> test;                                   \
            test.SetUp();                                                                                           \
            test.TestBody();                                                                                        \
            test.TearDown();                                                                                        \
        } else {                                                                                                    \
            std::string reason_not_run = "Skipping test requiring connected camera, to enable re-run with option "; \
            if (needed_options.size() > 0) {                                                                        \
                auto it = needed_options.begin(), it_end = needed_options.end();                                    \
                reason_not_run += "'--with-camera" + *it + "'";                                                     \
                ++it;                                                                                               \
                for (; it != it_end; ++it) {                                                                        \
                    reason_not_run += " or '--with-camera" + *it + "'";                                             \
                }                                                                                                   \
            } else {                                                                                                \
                reason_not_run += "'--with-camera'";                                                                \
            }                                                                                                       \
            std::cout << "\033[0;34m" << reason_not_run << "\033[m" << std::endl;                                   \
        }                                                                                                           \
        SUCCEED();                                                                                                  \
    }                                                                                                               \
    template<typename TypeParamT>                                                                                   \
    void CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name)<TypeParamT>::TestBody()

#define TYPED_TEST_WITH_ANY_CAMERA(test_case_name, test_name) \
    TYPED_TEST_WITH_SPECIFIC_CAMERA(test_case_name, test_name, {camera_param()})
#define TYPED_TEST_WITH_CAMERA(...) \
    EXPAND_ARG(GET_MACRO(__VA_ARGS__, TYPED_TEST_WITH_SPECIFIC_CAMERA, TYPED_TEST_WITH_ANY_CAMERA)(__VA_ARGS__))

#define TYPED_TEST_WITHOUT_CAMERA(test_fixture, test_name)                         \
    DECLARE_CUSTOM_TYPED_TEST_IMPL_CLASS(test_fixture, test_name);                 \
    TYPED_TEST(test_fixture, test_name) {                                          \
        if (Metavision::GtestsParameters::instance().run_test_without_camera) {    \
            CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name)<TypeParam> test;  \
            test.SetUp();                                                          \
            test.TestBody();                                                       \
            test.TearDown();                                                       \
        } else {                                                                   \
            std::cout << "\033[0;34mSkipping test requiring no connected camera, " \
                         "to enable re-run with --without-camera.\033[m"           \
                      << std::endl;                                                \
        }                                                                          \
        SUCCEED();                                                                 \
    }                                                                              \
    template<typename TypeParamT>                                                  \
    void CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name)<TypeParamT>::TestBody()

#define TYPED_TEST_WITH_DATASET(test_fixture, test_name)                          \
    DECLARE_CUSTOM_TYPED_TEST_IMPL_CLASS(test_fixture, test_name);                \
    TYPED_TEST(test_fixture, test_name) {                                         \
        if (!Metavision::GtestsParameters::instance().dataset_dir.empty()) {      \
            CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name)<TypeParam> test; \
            test.SetUp();                                                         \
            test.TestBody();                                                      \
            test.TearDown();                                                      \
        } else {                                                                  \
            std::cout << "\033[0;34mSkipping test requiring dataset, "            \
                         "to enable re-run with --dataset-dir DATASET.\033[m"     \
                      << std::endl;                                               \
        }                                                                         \
        SUCCEED();                                                                \
    }                                                                             \
    template<typename TypeParamT>                                                 \
    void CUSTOM_TEST_IMPL_CLASS_NAME(test_fixture, test_name)<TypeParamT>::TestBody()

} // namespace testing

#endif // METAVISION_UTILS_GTEST_CUSTOM_H
