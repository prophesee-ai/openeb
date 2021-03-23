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

#ifndef METAVISION_UTILS_GTEST_WARNING_REMOVAL_HELPER_H
#define METAVISION_UTILS_GTEST_WARNING_REMOVAL_HELPER_H

// https://stackoverflow.com/questions/3378560/how-to-disable-gcc-warnings-for-a-few-lines-of-code
#ifdef _MSC_VER
#define DIAG_DO_PRAGMA(x) __pragma(x)
#define DIAG_PRAGMA(compiler, x) DIAG_DO_PRAGMA(warning(x))
#else
#define DIAG_DO_PRAGMA(x) _Pragma(#x)
#define DIAG_PRAGMA(compiler, x) DIAG_DO_PRAGMA(compiler diagnostic x)
#endif
#if defined(__clang__)
#define DISABLE_WARNING(gcc_unused, clang_option, msvc_unused) \
    DIAG_PRAGMA(clang, push) DIAG_PRAGMA(clang, ignored "-W" clang_option)
#define ENABLE_WARNING(gcc_unused, clang_option, msvc_unused) DIAG_PRAGMA(clang, pop)
#elif defined(_MSC_VER)
#define DISABLE_WARNING(gcc_unused, clang_unused, msvc_errorcode) \
    DIAG_PRAGMA(msvc, push) DIAG_DO_PRAGMA(warning(disable : msvc_errorcode))
#define ENABLE_WARNING(gcc_unused, clang_unused, msvc_errorcode) DIAG_PRAGMA(msvc, pop)
#elif defined(__GNUC__)
#if ((__GNUC__ * 100) + __GNUC_MINOR__) >= 406
#define DISABLE_WARNING(gcc_option, clang_unused, msvc_unused) \
    DIAG_PRAGMA(GCC, push) DIAG_PRAGMA(GCC, ignored "-W" gcc_option)
#define ENABLE_WARNING(gcc_option, clang_unused, msvc_unused) DIAG_PRAGMA(GCC, pop)
#else
#define DISABLE_WARNING(gcc_option, clang_unused, msvc_unused) DIAG_PRAGMA(GCC, ignored "-W" gcc_option)
#define ENABLE_WARNING(gcc_option, clang_option, msvc_unused) DIAG_PRAGMA(GCC, warning "-W" gcc_option)
#endif
#endif

// clang-format off
#define TEMPORARLY_DISABLE_WARNING(gcc_option, clang_option, msvc_errorcode, call) \
    DISABLE_WARNING(gcc_option, clang_option, msvc_errorcode)                      \
    call                                                                           \
    ENABLE_WARNING(gcc_option, clang_option, msvc_errorcode)
// clang-format on

#endif /* METAVISION_UTILS_GTEST_WARNING_REMOVAL_HELPER_H */