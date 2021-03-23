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

#include "metavision/sdk/base/utils/log.h"

namespace Metavision {

namespace {
bool gLogLevelEnvRead    = false;
const char *gLogLevelEnv = nullptr;
LogLevel gLevel(LogLevel::Info);
const char *gLogStreamEnv = nullptr;
bool gLogStreamEnvRead    = false;
std::ostream *gStream(&std::cerr);
std::unique_ptr<std::ofstream> gFileStream;

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
} // namespace detail

LogLevel getLogLevel() {
    if (!gLogLevelEnvRead) {
        gLogLevelEnv     = getenv("MV_LOG_LEVEL");
        gLogLevelEnvRead = true;
    }
    if (gLogLevelEnv) {
        const std::string s(gLogLevelEnv);
        if (s == "ERROR") {
            return LogLevel::Error;
        } else if (s == "WARNING") {
            return LogLevel::Warning;
        } else if (s == "INFO") {
            return LogLevel::Info;
        } else if (s == "TRACE") {
            return LogLevel::Trace;
        } else if (s == "DEBUG") {
            return LogLevel::Debug;
        }
    }
    return gLevel;
}

void setLogLevel(const LogLevel &level) {
    gLevel = level;
}

void resetLogLevelFromEnv() {
    gLogLevelEnvRead = false;
    gLogLevelEnv     = nullptr;
}

std::ostream &getLogStream() {
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
    return *gStream;
}

void setLogStream(std::ostream &stream) {
    gStream = &stream;
}

void resetLogStreamFromEnv() {
    gLogStreamEnvRead = false;
    gFileStream.reset(nullptr);
}

#if !defined DEBUG && defined NDEBUG
constexpr LogLevel LoggingOperation<LogLevel::Debug>::Level;

namespace detail {
template<>
LoggingOperation<LogLevel::Debug> log<LogLevel::Debug>(const std::string &file, int line, const std::string &function,
                                                       const std::string &prefixFmt) {
    // when in release, this should return an "object" on which all operations are in fact no-op
    return LoggingOperation<LogLevel::Debug>(std::cerr, std::string(), std::string(), 0, std::string());
}
} // namespace detail

#endif

} // namespace Metavision
