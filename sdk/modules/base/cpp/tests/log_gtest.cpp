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

#include <cstdlib>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <gtest/gtest.h>
#include <type_traits>

#include "metavision/sdk/base/utils/log.h"
#include "metavision/utils/gtest/gtest_with_tmp_dir.h"

using namespace Metavision;

namespace {
#if defined DEBUG || !defined NDEBUG
constexpr bool CompiledInDebug = true;
#else
constexpr bool CompiledInDebug = false;
#endif
} // namespace

TEST(Log_GTest, MV_LOG_DEBUG_level) {
    setLogLevel(LogLevel::Debug);
    EXPECT_EQ(LogLevel::Debug, MV_LOG_DEBUG().Level);
    EXPECT_EQ(LogLevel::Debug, MV_LOG_DEBUG("blub").Level);
    EXPECT_EQ(LogLevel::Debug, MV_LOG_DEBUG(std::string("blub")).Level);
}

TEST(TLog_GTest, MV_LOG_DEBUG_file_prefix) {
    setLogLevel(LogLevel::Debug);
    auto buff = std::cerr.rdbuf();
    std::ostringstream oss;
    // redirect std::cerr
    std::cerr.rdbuf(oss.rdbuf());

    std::string prefixFmt("<FILE>");
    std::string fileStr("log_gtest.cpp");
    MV_LOG_DEBUG(prefixFmt) << Metavision::Log::no_endline;

    std::string output = oss.str();
    if (!CompiledInDebug)
        EXPECT_EQ(std::string(), output);
    else
        EXPECT_EQ(fileStr, output);

    // restore std::cerr
    std::cerr.rdbuf(buff);
}

TEST(TLog_GTest, MV_LOG_DEBUG_line_prefix) {
    setLogLevel(LogLevel::Debug);
    auto buff = std::cerr.rdbuf();
    std::ostringstream oss;
    // redirect std::cerr
    std::cerr.rdbuf(oss.rdbuf());

    std::string prefixFmt("<LINE>");
    std::string fileStr(std::to_string(__LINE__ + 1));
    MV_LOG_DEBUG(prefixFmt) << Metavision::Log::no_endline;

    std::string output = oss.str();
    if (!CompiledInDebug)
        EXPECT_EQ(std::string(), output);
    else
        EXPECT_EQ(fileStr, output);

    // restore std::cerr
    std::cerr.rdbuf(buff);
}

TEST(TLog_GTest, MV_LOG_DEBUG_function_prefix) {
    setLogLevel(LogLevel::Debug);
    auto buff = std::cerr.rdbuf();
    std::ostringstream oss;
    // redirect std::cerr
    std::cerr.rdbuf(oss.rdbuf());

    std::string prefixFmt("<FUNCTION>");
    std::string fileStr(__PRETTY_FUNCTION__);
    MV_LOG_DEBUG(prefixFmt) << Metavision::Log::no_endline;

    std::string output = oss.str();
    if (!CompiledInDebug)
        EXPECT_EQ(std::string(), output);
    else
        EXPECT_EQ(fileStr, output);

    // restore std::cerr
    std::cerr.rdbuf(buff);
}

TEST(Log_GTest, MV_LOG_TRACE_level) {
    EXPECT_EQ(LogLevel::Trace, MV_LOG_TRACE().Level);
    EXPECT_EQ(LogLevel::Trace, MV_LOG_TRACE("blub").Level);
    EXPECT_EQ(LogLevel::Trace, MV_LOG_TRACE(std::string("blub")).Level);
}

TEST(TLog_GTest, MV_LOG_TRACE_file_prefix) {
    setLogLevel(LogLevel::Trace);
    auto buff = std::cerr.rdbuf();
    std::ostringstream oss;
    // redirect std::cerr
    std::cerr.rdbuf(oss.rdbuf());

    std::string prefixFmt("<FILE>");
    std::string fileStr("log_gtest.cpp");
    MV_LOG_TRACE(prefixFmt) << Metavision::Log::no_endline;

    std::string output = oss.str();
    EXPECT_EQ(fileStr, output);

    // restore std::cerr
    std::cerr.rdbuf(buff);
}

TEST(TLog_GTest, MV_LOG_TRACE_line_prefix) {
    setLogLevel(LogLevel::Debug);
    auto buff = std::cerr.rdbuf();
    std::ostringstream oss;
    // redirect std::cerr
    std::cerr.rdbuf(oss.rdbuf());

    std::string prefixFmt("<LINE>");
    std::string fileStr(std::to_string(__LINE__ + 1));
    MV_LOG_TRACE(prefixFmt) << Metavision::Log::no_endline;

    std::string output = oss.str();
    EXPECT_EQ(fileStr, output);

    // restore std::cerr
    std::cerr.rdbuf(buff);
}

TEST(TLog_GTest, MV_LOG_TRACE_function_prefix) {
    setLogLevel(LogLevel::Debug);
    auto buff = std::cerr.rdbuf();
    std::ostringstream oss;
    // redirect std::cerr
    std::cerr.rdbuf(oss.rdbuf());

    std::string prefixFmt("<FUNCTION>");
    std::string fileStr(__PRETTY_FUNCTION__);
    MV_LOG_TRACE(prefixFmt) << Metavision::Log::no_endline;

    std::string output = oss.str();
    EXPECT_EQ(fileStr, output);

    // restore std::cerr
    std::cerr.rdbuf(buff);
}

TEST(Log_GTest, MV_LOG_INFO_level) {
    EXPECT_EQ(LogLevel::Info, MV_LOG_INFO().Level);
    EXPECT_EQ(LogLevel::Info, MV_LOG_INFO("blub").Level);
    EXPECT_EQ(LogLevel::Info, MV_LOG_INFO(std::string("blub")).Level);
}

TEST(TLog_GTest, MV_LOG_INFO_file_prefix) {
    setLogLevel(LogLevel::Info);
    auto buff = std::cerr.rdbuf();
    std::ostringstream oss;
    // redirect std::cerr
    std::cerr.rdbuf(oss.rdbuf());

    std::string prefixFmt("<FILE>");
    std::string fileStr("log_gtest.cpp");
    MV_LOG_INFO(prefixFmt) << Metavision::Log::no_endline;

    std::string output = oss.str();
    EXPECT_EQ(fileStr, output);

    // restore std::cerr
    std::cerr.rdbuf(buff);
}

TEST(TLog_GTest, MV_LOG_INFO_line_prefix) {
    setLogLevel(LogLevel::Debug);
    auto buff = std::cerr.rdbuf();
    std::ostringstream oss;
    // redirect std::cerr
    std::cerr.rdbuf(oss.rdbuf());

    std::string prefixFmt("<LINE>");
    std::string fileStr(std::to_string(__LINE__ + 1));
    MV_LOG_INFO(prefixFmt) << Metavision::Log::no_endline;

    std::string output = oss.str();
    EXPECT_EQ(fileStr, output);

    // restore std::cerr
    std::cerr.rdbuf(buff);
}

TEST(TLog_GTest, MV_LOG_INFO_function_prefix) {
    setLogLevel(LogLevel::Debug);
    auto buff = std::cerr.rdbuf();
    std::ostringstream oss;
    // redirect std::cerr
    std::cerr.rdbuf(oss.rdbuf());

    std::string prefixFmt("<FUNCTION>");
    std::string fileStr(__PRETTY_FUNCTION__);
    MV_LOG_INFO(prefixFmt) << Metavision::Log::no_endline;

    std::string output = oss.str();
    EXPECT_EQ(fileStr, output);

    // restore std::cerr
    std::cerr.rdbuf(buff);
}

TEST(Log_GTest, MV_LOG_WARNING_level) {
    EXPECT_EQ(LogLevel::Warning, MV_LOG_WARNING().Level);
    EXPECT_EQ(LogLevel::Warning, MV_LOG_WARNING("blub").Level);
    EXPECT_EQ(LogLevel::Warning, MV_LOG_WARNING(std::string("blub")).Level);
}

TEST(TLog_GTest, MV_LOG_WARNING_file_prefix) {
    setLogLevel(LogLevel::Warning);
    auto buff = std::cerr.rdbuf();
    std::ostringstream oss;
    // redirect std::cerr
    std::cerr.rdbuf(oss.rdbuf());

    std::string prefixFmt("<FILE>");
    std::string fileStr("log_gtest.cpp");
    MV_LOG_WARNING(prefixFmt) << Metavision::Log::no_endline;

    std::string output = oss.str();
    EXPECT_EQ(fileStr, output);

    // restore std::cerr
    std::cerr.rdbuf(buff);
}

TEST(TLog_GTest, MV_LOG_WARNING_line_prefix) {
    setLogLevel(LogLevel::Debug);
    auto buff = std::cerr.rdbuf();
    std::ostringstream oss;
    // redirect std::cerr
    std::cerr.rdbuf(oss.rdbuf());

    std::string prefixFmt("<LINE>");
    std::string fileStr(std::to_string(__LINE__ + 1));
    MV_LOG_WARNING(prefixFmt) << Metavision::Log::no_endline;

    std::string output = oss.str();
    EXPECT_EQ(fileStr, output);

    // restore std::cerr
    std::cerr.rdbuf(buff);
}

TEST(TLog_GTest, MV_LOG_WARNING_function_prefix) {
    setLogLevel(LogLevel::Debug);
    auto buff = std::cerr.rdbuf();
    std::ostringstream oss;
    // redirect std::cerr
    std::cerr.rdbuf(oss.rdbuf());

    std::string prefixFmt("<FUNCTION>");
    std::string fileStr(__PRETTY_FUNCTION__);
    MV_LOG_WARNING(prefixFmt) << Metavision::Log::no_endline;

    std::string output = oss.str();
    EXPECT_EQ(fileStr, output);

    // restore std::cerr
    std::cerr.rdbuf(buff);
}

TEST(Log_GTest, MV_LOG_ERROR_level) {
    EXPECT_EQ(LogLevel::Error, MV_LOG_ERROR().Level);
    EXPECT_EQ(LogLevel::Error, MV_LOG_ERROR("blub").Level);
    EXPECT_EQ(LogLevel::Error, MV_LOG_ERROR(std::string("blub")).Level);
}

TEST(TLog_GTest, MV_LOG_ERROR_file_prefix) {
    setLogLevel(LogLevel::Error);
    auto buff = std::cerr.rdbuf();
    std::ostringstream oss;
    // redirect std::cerr
    std::cerr.rdbuf(oss.rdbuf());

    std::string prefixFmt("<FILE>");
    std::string fileStr("log_gtest.cpp");
    MV_LOG_ERROR(prefixFmt) << Metavision::Log::no_endline;

    std::string output = oss.str();
    EXPECT_EQ(fileStr, output);

    // restore std::cerr
    std::cerr.rdbuf(buff);
}

TEST(TLog_GTest, MV_LOG_ERROR_line_prefix) {
    setLogLevel(LogLevel::Debug);
    auto buff = std::cerr.rdbuf();
    std::ostringstream oss;
    // redirect std::cerr
    std::cerr.rdbuf(oss.rdbuf());

    std::string prefixFmt("<LINE>");
    std::string fileStr(std::to_string(__LINE__ + 1));
    MV_LOG_ERROR(prefixFmt) << Metavision::Log::no_endline;

    std::string output = oss.str();
    EXPECT_EQ(fileStr, output);

    // restore std::cerr
    std::cerr.rdbuf(buff);
}

TEST(TLog_GTest, MV_LOG_ERROR_function_prefix) {
    setLogLevel(LogLevel::Debug);
    auto buff = std::cerr.rdbuf();
    std::ostringstream oss;
    // redirect std::cerr
    std::cerr.rdbuf(oss.rdbuf());

    std::string prefixFmt("<FUNCTION>");
    std::string fileStr(__PRETTY_FUNCTION__);
    MV_LOG_ERROR(prefixFmt) << Metavision::Log::no_endline;

    std::string output = oss.str();
    EXPECT_EQ(fileStr, output);

    // restore std::cerr
    std::cerr.rdbuf(buff);
}

struct LogWithTmpDir_GTest : public GTestWithTmpDir {
    void SetUp() override {
        filename_ = tmpdir_handler_->get_full_path("log.txt");
        ofs_      = std::ofstream(filename_);
        setLogLevel(LogLevel::Debug);
        resetLogLevelFromEnv();
        resetLogStreamFromEnv();
    }
    std::string filename_;
    std::ofstream ofs_;
};

TEST_F(LogWithTmpDir_GTest, streams) {
    setLogStream(ofs_);
    MV_LOG_INFO("") << Metavision::Log::no_space << "test"
                    << "bla"
                    << "\n"
                    << "yo";
    ofs_.close();

    std::ifstream ifs(filename_);
    std::vector<char> buf(std::istreambuf_iterator<char>(ifs), {});
    std::string content(buf.begin(), buf.end());
    EXPECT_EQ("testbla\nyo\n", content);
}

TEST_F(LogWithTmpDir_GTest, streams_with_env_reset) {
#ifdef _WIN32
    std::string s("MV_LOG_FILE=");
    s += filename_;
    _putenv(s.c_str());
#else
    setenv("MV_LOG_FILE", filename_.c_str(), 1);
#endif
    auto &stream = getLogStream();

#ifdef _WIN32
    _putenv("MV_LOG_FILE=");
#else
    unsetenv("MV_LOG_FILE");
#endif
    resetLogStreamFromEnv();

    auto &stream2 = getLogStream();

    EXPECT_NE(&stream, &stream2);
}

TEST_F(LogWithTmpDir_GTest, streams_with_env) {
#ifdef _WIN32
    std::string s("MV_LOG_FILE=");
    s += filename_;
    _putenv(s.c_str());
#else
    setenv("MV_LOG_FILE", filename_.c_str(), 1);
#endif
    MV_LOG_INFO("") << Metavision::Log::no_space << "test"
                    << "bla"
                    << "\n"
                    << "yo";

#ifdef _WIN32
    _putenv("MV_LOG_FILE=");
#else
    unsetenv("MV_LOG_FILE");
#endif

    // force closing (and flushing..) the file
    resetLogStreamFromEnv();
    auto &stream = getLogStream();

    std::ifstream ifs(filename_);
    std::vector<char> buf(std::istreambuf_iterator<char>(ifs), {});
    std::string content(buf.begin(), buf.end());
    EXPECT_EQ("testbla\nyo\n", content);
}

template<LogLevel level>
using LogLevelConstant = std::integral_constant<LogLevel, level>;

// clang-format off
using TestingCases = testing::Types<
    LogLevelConstant<LogLevel::Debug>,
    LogLevelConstant<LogLevel::Trace>,
    LogLevelConstant<LogLevel::Info>,
    LogLevelConstant<LogLevel::Warning>,
    LogLevelConstant<LogLevel::Error>
>;
// clang-format on

namespace {

template<LogLevel Level>
LoggingOperation<Level> streamLog(const std::string &file, int line, const std::string &function, std::ostream &stream,
                                  const std::string &prefixFmt) {
    return LoggingOperation<Level>(stream, prefixFmt, file, line, function);
}

template<LogLevel Level>
LoggingOperation<Level> streamLog(const std::string &file, int line, const std::string &function, std::ostream &stream,
                                  const char *const prefixFmt) {
    return LoggingOperation<Level>(stream, prefixFmt, file, line, function);
}

#define streamLogCall(T, ...) streamLog<T>(__FILE__, __LINE__, __PRETTY_FUNCTION__, ##__VA_ARGS__)
} // namespace

template<typename T>
struct TLog_GTest : public testing::Test {
    void SetUp() override {
        setLogLevel(LogLevel::Debug);
        resetLogLevelFromEnv();
        resetLogStreamFromEnv();
    }
};

TYPED_TEST_CASE(TLog_GTest, TestingCases);

TYPED_TEST(TLog_GTest, spaces) {
    static constexpr LogLevel Level = TypeParam::value;
    std::string str;
    std::ostringstream oss(str);
    streamLogCall(Level, oss, "") << Metavision::Log::space << "test"
                                  << "bla"
                                  << "yo";
    std::string output = oss.str();
    if (Level == LogLevel::Debug && !CompiledInDebug)
        EXPECT_EQ(std::string(), output);
    else
        EXPECT_EQ("test bla yo \n", output);
}

TYPED_TEST(TLog_GTest, no_spaces) {
    static constexpr LogLevel Level = TypeParam::value;
    std::string str;
    std::ostringstream oss(str);
    streamLogCall(Level, oss, "") << Metavision::Log::no_space << "test"
                                  << "bla"
                                  << "yo";
    std::string output = oss.str();
    if (Level == LogLevel::Debug && !CompiledInDebug)
        EXPECT_EQ(std::string(), output);
    else
        EXPECT_EQ("testblayo\n", output);
}

TYPED_TEST(TLog_GTest, endline) {
    static constexpr LogLevel Level = TypeParam::value;
    std::string str;
    std::ostringstream oss(str);
    streamLogCall(Level, oss, "") << Metavision::Log::endline << "test"
                                  << "bla"
                                  << "yo";
    std::string output = oss.str();
    if (Level == LogLevel::Debug && !CompiledInDebug)
        EXPECT_EQ(std::string(), output);
    else
        EXPECT_EQ("test bla yo \n", output);
}

TYPED_TEST(TLog_GTest, no_endline) {
    static constexpr LogLevel Level = TypeParam::value;
    std::string str;
    std::ostringstream oss(str);
    streamLogCall(Level, oss, "") << Metavision::Log::no_endline << "test"
                                  << "bla"
                                  << "yo";
    std::string output = oss.str();
    if (Level == LogLevel::Debug && !CompiledInDebug)
        EXPECT_EQ(std::string(), output);
    else
        EXPECT_EQ("test bla yo ", output);
}

TYPED_TEST(TLog_GTest, manip_no_space) {
    static constexpr LogLevel Level = TypeParam::value;
    std::string str;
    std::ostringstream oss(str);
    streamLogCall(Level, oss, "") << Metavision::Log::no_space << std::left << std::setw(5) << "a" << std::right
                                  << std::setw(7) << std::setfill('#') << std::fixed << std::setprecision(2)
                                  << 67.12345;
    std::string output = oss.str();
    if (Level == LogLevel::Debug && !CompiledInDebug)
        EXPECT_EQ(std::string(), output);
    else
        EXPECT_EQ("a    ##67.12\n", output);
}

TYPED_TEST(TLog_GTest, manip_space) {
    static constexpr LogLevel Level = TypeParam::value;
    std::string str;
    std::ostringstream oss(str);
    streamLogCall(Level, oss, "") << std::left << std::setw(5) << "a" << std::right << std::setw(7) << std::setfill('#')
                                  << std::fixed << std::setprecision(2) << 67.12345;
    std::string output = oss.str();
    if (Level == LogLevel::Debug && !CompiledInDebug)
        EXPECT_EQ(std::string(), output);
    else
        EXPECT_EQ("a     ##67.12 \n", output);
}

namespace {
std::map<LogLevel, std::string> expected_output;
} // namespace

TYPED_TEST(TLog_GTest, level_prefix) {
    expected_output = {{LogLevel::Debug, "[DEBUG] "},
                       {LogLevel::Trace, "[TRACE] "},
                       {LogLevel::Info, "[INFO] "},
                       {LogLevel::Warning, "[WARNING] "},
                       {LogLevel::Error, "[ERROR] "}};

    static constexpr LogLevel Level = TypeParam::value;
    std::string str;
    std::ostringstream oss(str);
    streamLogCall(Level, oss, "[<LEVEL>] ") << Metavision::Log::no_endline << Metavision::Log::no_space;
    std::string output = oss.str();
    if (Level == LogLevel::Debug && !CompiledInDebug)
        EXPECT_EQ(std::string(), output);
    else
        EXPECT_EQ(expected_output[Level], output);
}

TYPED_TEST(TLog_GTest, empty_prefix) {
    static constexpr LogLevel Level = TypeParam::value;
    std::string str;
    std::ostringstream oss(str);
    streamLogCall(Level, oss, "") << Metavision::Log::no_endline << Metavision::Log::no_space;
    std::string output = oss.str();
    EXPECT_EQ(std::string(), output);
}

TYPED_TEST(TLog_GTest, file_prefix) {
    static constexpr LogLevel Level = TypeParam::value;
    std::string prefixFmt("<FILE>");
    std::string fileStr("log_gtest.cpp");
    std::string str;
    std::ostringstream oss(str);
    streamLogCall(Level, oss, prefixFmt) << Metavision::Log::no_endline << Metavision::Log::no_space;
    std::string output = oss.str();
    if (Level == LogLevel::Debug && !CompiledInDebug)
        EXPECT_EQ(std::string(), output);
    else
        EXPECT_EQ(fileStr, output);
}

TYPED_TEST(TLog_GTest, custom_prefix) {
    expected_output = {{LogLevel::Debug, "TEST[<123> <!ADebug]"},
                       {LogLevel::Trace, "TEST[<123> <!ATrace]"},
                       {LogLevel::Info, "TEST[<123> <!AInfo]"},
                       {LogLevel::Warning, "TEST[<123> <!AWarning]"},
                       {LogLevel::Error, "TEST[<123> <!AError]"}};

    static constexpr LogLevel Level = TypeParam::value;
    std::string prefixFmt("TEST[<123> <!A<Level>]");
    std::string str;
    std::ostringstream oss(str);
    streamLogCall(Level, oss, prefixFmt) << Metavision::Log::no_endline << Metavision::Log::no_space;
    std::string output = oss.str();
    if (Level == LogLevel::Debug && !CompiledInDebug)
        EXPECT_EQ(std::string(), output);
    else
        EXPECT_EQ(expected_output[Level], output);
}

namespace {
std::string getFormattedDateTime(const char *const fmt) {
    char buf[1024];
    std::time_t t = std::time(nullptr);
    struct tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    std::strftime(buf, 1024, fmt, &tm_buf);
    return std::string(buf);
}
} // namespace

TYPED_TEST(TLog_GTest, time_prefix) {
    static constexpr LogLevel Level = TypeParam::value;
    std::string beforeStr, afterStr;
    do {
        std::string prefixFmt("<123<DATETIME:%A>%blub!");
        std::string str;
        std::ostringstream oss(str);
        beforeStr = getFormattedDateTime("%A");

        streamLogCall(Level, oss, prefixFmt) << Metavision::Log::no_endline << Metavision::Log::no_space;
        std::string output = oss.str();

        afterStr = getFormattedDateTime("%A");
        if (beforeStr == afterStr) {
            std::string expected_output;
            expected_output += "<123";
            expected_output += beforeStr;
            expected_output += "%blub!";
            if (Level == LogLevel::Debug && !CompiledInDebug)
                EXPECT_EQ(std::string(), output);
            else
                EXPECT_EQ(expected_output, output);
        }
    } while (beforeStr != afterStr);
}

TYPED_TEST(TLog_GTest, time_prefix_big) {
    static constexpr LogLevel Level = TypeParam::value;
    std::string beforeStr, afterStr;
    int num_repl = 10;
    do {
        std::string prefixFmt("<123<DATETIME:");
        for (int i = 0; i < num_repl; ++i) {
            prefixFmt += "%A";
        }
        prefixFmt += ">%blub!";
        std::string str;
        std::ostringstream oss(str);
        beforeStr = getFormattedDateTime("%A");

        streamLogCall(Level, oss, prefixFmt) << Metavision::Log::no_endline << Metavision::Log::no_space;
        std::string output = oss.str();

        afterStr = getFormattedDateTime("%A");
        if (beforeStr == afterStr) {
            std::string expected_output;
            expected_output += "<123";
            for (int i = 0; i < num_repl; ++i) {
                expected_output += beforeStr;
            }
            expected_output += "%blub!";
            if (Level == LogLevel::Debug && !CompiledInDebug)
                EXPECT_EQ(std::string(), output);
            else
                EXPECT_EQ(expected_output, output);
        }
    } while (beforeStr != afterStr);
}

TYPED_TEST(TLog_GTest, time_prefix_too_big) {
    static constexpr LogLevel Level = TypeParam::value;
    std::string beforeStr, afterStr;
    int num_repl = 1000;
    std::string prefixFmt("<123<DATETIME:");
    for (int i = 0; i < num_repl; ++i) {
        prefixFmt += "%A";
    }
    prefixFmt += ">%blub!";
    std::string str;
    std::ostringstream oss(str);

    streamLogCall(Level, oss, prefixFmt) << Metavision::Log::no_endline << Metavision::Log::no_space;
    std::string output = oss.str();

    if (Level == LogLevel::Debug && !CompiledInDebug)
        EXPECT_EQ(std::string(), output);
    else
        EXPECT_EQ(prefixFmt, output);
}

TYPED_TEST(TLog_GTest, levels) {
    static constexpr LogLevel Level = TypeParam::value;
    for (int gLevel = static_cast<int>(LogLevel::Debug); gLevel <= static_cast<int>(LogLevel::Error); ++gLevel) {
        setLogLevel(static_cast<LogLevel>(gLevel));
        std::string str;
        std::ostringstream oss(str);
        streamLogCall(Level, oss, "") << Metavision::Log::no_space << "test"
                                      << "bla"
                                      << "yo";
        std::string output = oss.str();
        if ((Level == LogLevel::Debug && !CompiledInDebug) || (Level < static_cast<LogLevel>(gLevel)))
            EXPECT_EQ(std::string(), output);
        else {
            EXPECT_EQ("testblayo\n", output);
        }
    }
}

TYPED_TEST(TLog_GTest, levels_with_env_reset) {
#ifdef _WIN32
    std::string s("MV_LOG_LEVEL=TRACE");
    _putenv(s.c_str());
#else
    setenv("MV_LOG_LEVEL", "TRACE", 1);
#endif
    auto level = getLogLevel();

#ifdef _WIN32
    _putenv("MV_LOG_LEVEL=");
#else
    unsetenv("MV_LOG_LEVEL");
#endif
    resetLogLevelFromEnv();

    auto level2 = getLogLevel();

    EXPECT_NE(level, level2);
}

namespace {
std::map<LogLevel, std::string> env_values{{LogLevel::Debug, "DEBUG"},
                                           {LogLevel::Trace, "TRACE"},
                                           {LogLevel::Info, "INFO"},
                                           {LogLevel::Warning, "WARNING"},
                                           {LogLevel::Error, "ERROR"}};
}

TYPED_TEST(TLog_GTest, levels_with_env) {
    static constexpr LogLevel Level = TypeParam::value;
    for (auto &p : env_values) {
        resetLogLevelFromEnv();
#ifdef _WIN32
        std::string s("MV_LOG_LEVEL=");
        s += p.second;
        _putenv(s.c_str());
#else
        setenv("MV_LOG_LEVEL", p.second.c_str(), 1);
#endif
        std::string str;
        std::ostringstream oss(str);
        streamLogCall(Level, oss, "") << Metavision::Log::no_space << "test"
                                      << "bla"
                                      << "yo";
        std::string output = oss.str();
        if ((Level == LogLevel::Debug && !CompiledInDebug) || (Level < p.first))
            EXPECT_EQ(std::string(), output);
        else
            EXPECT_EQ("testblayo\n", output);
    }
#ifdef _WIN32
    _putenv("MV_LOG_LEVEL=");
#else
    unsetenv("MV_LOG_LEVEL");
#endif
}