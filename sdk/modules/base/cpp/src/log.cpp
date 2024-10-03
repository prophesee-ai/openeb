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

#include <iostream>
#include <fstream>
#include <memory>
#include <mutex>
#ifdef __ANDROID__
#include <android/log.h>
#include <dlfcn.h>
#endif

#include "metavision/sdk/base/utils/log.h"
#ifdef __ANDROID__
#include "metavision/sdk/base/utils/detail/android_log.h"

// https://stackoverflow.com/questions/28413530/api-to-get-android-system-properties-is-removed-in-arm64-platforms
#if (__ANDROID_API__ >= 21)
// Android 'L' makes __system_property_get a non-global symbol.
// Here we provide a stub which loads the symbol from libc via dlsym.
typedef int (*PFN_SYSTEM_PROP_GET)(const char *, char *);
int __system_property_get(const char *name, char *value) {
    static PFN_SYSTEM_PROP_GET __real_system_property_get = NULL;
    if (!__real_system_property_get) {
        // libc.so should already be open, get a handle to it.
        void *handle = dlopen("libc.so", RTLD_NOLOAD);
        if (!handle) {
            __android_log_print(ANDROID_LOG_ERROR, "foobar", "Cannot dlopen libc.so: %s.\n", dlerror());
        } else {
            __real_system_property_get = (PFN_SYSTEM_PROP_GET)dlsym(handle, "__system_property_get");
        }
        if (!__real_system_property_get) {
            __android_log_print(ANDROID_LOG_ERROR, "foobar", "Cannot resolve __system_property_get(): %s.\n",
                                dlerror());
        }
    }
    if (!__real_system_property_get)
        return (0);
    return (*__real_system_property_get)(name, value);
}
#endif // __ANDROID_API__ >= 21

#endif

namespace Metavision {

namespace {
bool gLogLevelEnvRead = false;
std::string gLogLevelEnv;
const char *gLogStreamEnv = nullptr;
bool gLogStreamEnvRead    = false;
std::unique_ptr<std::ofstream> gFileStream;
LogOptions gLogOptions;
#ifdef __ANDROID__
detail::android_streambuf streambuf("MV_LOG");
std::ostream android_stream(reinterpret_cast<std::streambuf *>(&streambuf));
#endif

std::mutex concurrent_ostreambuf_mutex;
} // namespace

namespace detail {
concurrent_ostreambuf::concurrent_ostreambuf(std::streambuf *buf) : buf_(buf), has_output_(false) {}

std::streamsize concurrent_ostreambuf::xsputn(const char_type *s, std::streamsize n) {
    bytes_.insert(bytes_.end(), s, s + n);
    has_output_ = true;
    return n;
}

int concurrent_ostreambuf::overflow(int ch) {
    if (ch != EOF) {
        has_output_ = true;
        bytes_.push_back(ch);
    }
    return ch;
}

int concurrent_ostreambuf::sync() {
    {
        std::lock_guard<std::mutex> lock(concurrent_ostreambuf_mutex);
        buf_->sputn(bytes_.data(), bytes_.size());
    }
    bytes_.clear();
    return 1;
}

void concurrent_ostreambuf::reset_output_sentinel() {
    has_output_ = false;
}

bool concurrent_ostreambuf::get_output_sentinel() const {
    return has_output_;
}

LogLevelNameMap::const_iterator getLongestLogLevelName(const LogLevelNameMap &levelnames) {
    return std::max_element(levelnames.cbegin(), levelnames.cend(),
                            [](const auto &lhs, const auto &rhs) { return lhs.second.size() < rhs.second.size(); });
}

std::string getPaddedLevelLabel(const LogLevel &level, const LogLevelNameMap &labels, char padding_char) {
    std::string level_label  = labels.at(level);
    auto longest_level_label = getLongestLogLevelName(labels)->second;
    size_t padding_size      = longest_level_label.size() - level_label.size();
    level_label.insert(0, padding_size, ' ');
    return level_label;
}

std::string getLevelName(const LogLevel &level, const LogLevelNameMap &labels, bool level_prefix_padding) {
    if (level_prefix_padding) {
        return getPaddedLevelLabel(level, labels);
    }
    return labels.at(level);
}

#ifdef __ANDROID__
android_streambuf::android_streambuf(const std::string tag) : tag_(tag) {}

std::streamsize android_streambuf::xsputn(const char *s, std::streamsize n) {
    ostr_.write(s, n);
    if ((!ostr_.str().empty()) && (ostr_.str().back() == '\n')) {
        auto size = ostr_.str().size();
        __android_log_print(ANDROID_LOG_DEBUG, tag_.c_str(), "%s", ostr_.str().c_str());
        ostr_.str("");
        return size;
    }
    return 0;
};

int android_streambuf::overflow(int ch) {
    if (ch != EOF) {
        xsputn(reinterpret_cast<const char *>(&ch), 1);
    }
    return ch;
}

int android_streambuf::sync() {
    return 0;
}
#endif // __ANDROID__
} // namespace detail

LogLevel getLogLevel() {
    return gLogOptions.getLevel();
}

void setLogLevel(const LogLevel &level) {
    gLogOptions.setLevel(level);
}

void setLogOptions(LogOptions opts) {
    gLogOptions = opts;
}

void resetLogOptions() {
    setLogOptions(LogOptions());
}

LogOptions getLogOptions() {
    return gLogOptions;
}

void resetLogLevelFromEnv() {
    gLogLevelEnvRead = false;
    gLogLevelEnv     = "";
}

std::ostream &getLogStream() {
    return gLogOptions.getStream();
}

void setLogStream(std::ostream &stream) {
    gLogOptions.setStream(stream);
}

void resetLogStreamFromEnv() {
    gLogStreamEnvRead = false;
    gFileStream.reset(nullptr);
}

LogOptions::LogOptions(LogLevel level, std::ostream &stream, bool level_prefix_padding) :
    level_(level), stream_(&stream), level_prefix_padding_(level_prefix_padding) {
#ifdef __ANDROID__
    if (stream_ == &std::cerr) {
        stream_ = &android_stream;
    }
#endif
}

LogOptions &LogOptions::setLevelPrefixPadding(bool is_padded) {
    level_prefix_padding_ = is_padded;
    return *this;
}
bool LogOptions::isLevelPrefixPadding() const {
    return level_prefix_padding_;
}
LogOptions &LogOptions::setStream(std::ostream &stream) {
    stream_ = &stream;
    return *this;
}
std::ostream &LogOptions::getStream() const {
    if (!gLogStreamEnvRead) {
        gLogStreamEnv     = getenv("MV_LOG_FILE");
        gLogStreamEnvRead = true;
        if (gLogStreamEnv) {
            gFileStream.reset(new std::ofstream(gLogStreamEnv));
        }
    }
    if (gLogStreamEnv) {
        if (gFileStream && gFileStream->is_open()) {
            return *gFileStream;
        }
    }
    return *stream_;
}

LogOptions &LogOptions::setLevel(const LogLevel &level) {
    level_ = level;
    return *this;
}

LogLevel LogOptions::getLevel() const {
    if (!gLogLevelEnvRead) {
#ifdef __ANDROID__
        auto get_str_prop = [](const std::string &prop, std::string &value) {
            char prop_str[1024];
            if (__system_property_get(prop.c_str(), prop_str)) {
                value = std::string(prop_str);
                return true;
            }
            return false;
        };
        if (!get_str_prop("debug.metavision.log.level", gLogLevelEnv)) {
            if (!get_str_prop("persist.metavision.log.level", gLogLevelEnv)) {
                gLogLevelEnv = "";
            }
        }
        if (gLogLevelEnv == "") {
#endif // __ANDROID__
            const char *log_level_env = getenv("MV_LOG_LEVEL");
            gLogLevelEnv              = std::string((log_level_env != nullptr) ? log_level_env : "");
#ifdef __ANDROID__
        }
#endif // __ANDROID__
        gLogLevelEnvRead = true;
    }
    if (!gLogLevelEnv.empty()) {
        if (gLogLevelEnv == "ERROR") {
            return LogLevel::Error;
        } else if (gLogLevelEnv == "WARNING") {
            return LogLevel::Warning;
        } else if (gLogLevelEnv == "INFO") {
            return LogLevel::Info;
        } else if (gLogLevelEnv == "TRACE") {
            return LogLevel::Trace;
        } else if (gLogLevelEnv == "DEBUG") {
            return LogLevel::Debug;
        }
        // In case of non of the above returns have been called, it means that gLogLevelEnv is incorrect
        // Resetting it will prevent to entering this IF condition in next calls of getLogLevel()
        gLogLevelEnv = "";
    }
    return level_;
}

#if !defined DEBUG && defined NDEBUG
constexpr LogLevel LoggingOperation<LogLevel::Debug>::Level;

namespace detail {
template<>
LoggingOperation<LogLevel::Debug> log<LogLevel::Debug>(const std::filesystem::path &file, int line,
                                                       const std::string &function, const std::string &prefixFmt) {
    // when in release, this should return an "object" on which all operations are in fact no-op
    return LoggingOperation<LogLevel::Debug>(LogOptions(), std::string(), std::filesystem::path(), 0, std::string());
}
} // namespace detail

#endif

} // namespace Metavision
