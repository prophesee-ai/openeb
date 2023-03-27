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

#ifndef METAVISION_SDK_BASE_DETAIL_LOG_IMPL_H
#define METAVISION_SDK_BASE_DETAIL_LOG_IMPL_H

#include <algorithm>
#include <ctime>
#include <map>
#include <cstring>
#include <string>
#include <streambuf>

namespace Metavision {
#if !defined DEBUG && defined NDEBUG
template<>
class LoggingOperation<LogLevel::Debug> {
public:
    static constexpr LogLevel Level = LogLevel::Debug;

    LoggingOperation(const LogOptions & /**/ = LogOptions(), const std::string & /**/ = std::string(),
                     const std::string & /**/ = std::string(), int = 0, const std::string & /**/ = std::string()) {}

    const std::string &file() {
        static std::string s;
        return s;
    }

    std::string function() const {
        return std::string();
    }

    std::string file() const {
        return std::string();
    }

    int line() const {
        return 0;
    }

    std::string prefix() const {
        return std::string();
    }

    void enableSpaceBetweenTokens() {}

    void disableSpaceBetweenTokens() {}

    void enableEndOfLineAtDestruction() {}

    void disableEndOfLineAtDestruction() {}

    void log(bool) {}

    template<typename T>
    void log(const std::vector<T> &) {}

    template<typename T>
    void log(const T &) {}

    void apply(std::ostream &(*)(std::ostream &)) {}

    void handleSpace() {}
};
#endif

namespace detail {
template<LogLevel Level>
LoggingOperation<Level> log(const std::string &file, int line, const std::string &function) {
    return LoggingOperation<Level>(getLogOptions(), "", file, line, function);
}

template<LogLevel Level>
LoggingOperation<Level> log(const std::string &file, int line, const std::string &function,
                            const std::string &prefixFmt) {
    return LoggingOperation<Level>(getLogOptions(), prefixFmt, file, line, function);
}

template<LogLevel Level>
LoggingOperation<Level> log(const std::string &file, int line, const std::string &function,
                            const char *const prefixFmt) {
    return LoggingOperation<Level>(getLogOptions(), std::string(prefixFmt), file, line, function);
}
} // namespace detail
} // namespace Metavision

#ifdef _WIN32
#ifndef __PRETTY_FUNCTION__
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif
#endif

#define MV_LOG_WRAP(fn, level)                                                              \
    ([file = __FILE__, line = __LINE__, function = __PRETTY_FUNCTION__](auto &&...params) { \
        return fn<level>(file, line, function, params...);                                  \
    })

#define MV_LOG_WRAP_LEVEL(level) MV_LOG_WRAP(Metavision::detail::log, (level))

#define MV_LOG_DEBUG MV_LOG_WRAP_LEVEL(Metavision::LogLevel::Debug)
#define MV_LOG_TRACE MV_LOG_WRAP_LEVEL(Metavision::LogLevel::Trace)
#define MV_LOG_INFO MV_LOG_WRAP_LEVEL(Metavision::LogLevel::Info)
#define MV_LOG_WARNING MV_LOG_WRAP_LEVEL(Metavision::LogLevel::Warning)
#define MV_LOG_ERROR MV_LOG_WRAP_LEVEL(Metavision::LogLevel::Error)

namespace Metavision {
namespace detail {
static char datetime_buffer[1024];

using LogLevelNameMap = std::map<LogLevel, std::string>;

static LogLevelNameMap LabelsUpperCase{{LogLevel::Debug, "DEBUG"},
                                       {LogLevel::Trace, "TRACE"},
                                       {LogLevel::Info, "INFO"},
                                       {LogLevel::Warning, "WARNING"},
                                       {LogLevel::Error, "ERROR"}};

static LogLevelNameMap Labels{{LogLevel::Debug, "Debug"},
                              {LogLevel::Trace, "Trace"},
                              {LogLevel::Info, "Info"},
                              {LogLevel::Warning, "Warning"},
                              {LogLevel::Error, "Error"}};

LogLevelNameMap::const_iterator getLongestLogLevelName(const LogLevelNameMap &levelnames);
std::string getPaddedLevelLabel(const LogLevel &level, const LogLevelNameMap &labels, char padding_char = ' ');
std::string getLevelName(const LogLevel &level, const LogLevelNameMap &labels, bool level_prefix_padding);

template<LogLevel Level>
std::string getLogPrefixFormatString(bool level_prefix_padding, const std::string &prefixFmt, const std::string &file,
                                     int line, const std::string &function) {
    size_t pos;
    std::string s = prefixFmt;
    std::string token;
    token = "<Level>";
    if ((pos = s.find(token)) != std::string::npos) {
        s.replace(pos, token.size(), getLevelName(Level, Labels, level_prefix_padding));
    }
    token = "<LEVEL>";
    if ((pos = s.find(token)) != std::string::npos) {
        s.replace(pos, token.size(), getLevelName(Level, LabelsUpperCase, level_prefix_padding));
    }
    token = "<FILE>";
    if ((pos = s.find(token)) != std::string::npos) {
        std::string basename;
#ifdef _WIN32
        const char *const p = strrchr(file.c_str(), '\\');
#else
        const char *const p = strrchr(file.c_str(), '/');
#endif
        if (p)
            basename = std::string(p + 1);
        else
            basename = file;
        s.replace(pos, token.size(), basename);
    }
    token = "<LINE>";
    if ((pos = s.find(token)) != std::string::npos) {
        s.replace(pos, token.size(), std::to_string(line));
    }
    token = "<FUNCTION>";
    if ((pos = s.find(token)) != std::string::npos) {
        s.replace(pos, token.size(), function);
    }
    token = "<DATETIME:";
    if ((pos = s.find(token)) != std::string::npos) {
        size_t bpos = pos + token.size();
        token       = ">";
        size_t epos;
        if ((epos = s.find(token, bpos)) != std::string::npos) {
            std::string strftimefmt(s.substr(bpos, epos - bpos));
            std::time_t t = std::time(nullptr);
            struct tm tm_buf;
#ifdef _WIN32
            localtime_s(&tm_buf, &t);
#else
            localtime_r(&t, &tm_buf);
#endif
            if (std::strftime(datetime_buffer, 1024, strftimefmt.c_str(), &tm_buf)) {
                s.replace(pos, epos - pos + 1, datetime_buffer);
            } else {
                std::cerr << "Error when substituting token in log message, date time format yields a string that is "
                             "wider than 1024 characters, token replacement ignored."
                          << std::endl;
            }
        }
    }
    return s;
}

class concurrent_ostreambuf : public std::streambuf {
public:
    concurrent_ostreambuf(std::streambuf *buf);

    bool get_output_sentinel() const;
    void reset_output_sentinel();

protected:
    std::streamsize xsputn(const char *s, std::streamsize n) override;
    int overflow(int ch) override;
    int sync() override;

private:
    std::vector<char> bytes_;
    std::streambuf *buf_;
    bool has_output_;
};
} // namespace detail

template<LogLevel Level>
constexpr LogLevel LoggingOperation<Level>::Level;

template<LogLevel Level>
LoggingOperation<Level>::LoggingOperation(const LogOptions &opts, const std::string &prefixFmt, const std::string &file,
                                          int line, const std::string &function) :
    streambuf_(new detail::concurrent_ostreambuf(opts.getStream().rdbuf())),
    stream_(new std::ostream(streambuf_.get())),
    addSpaceBetweenTokens_(true),
    addEndLine_(true),
    should_output_(Level >= opts.getLevel()),
    prefix_(detail::getLogPrefixFormatString<Level>(opts.isLevelPrefixPadding(), prefixFmt, file, line, function)),
    file_(file),
    function_(function),
    line_(line) {
    if (should_output_)
        (*stream_) << prefix_;
}

template<LogLevel Level>
LoggingOperation<Level>::LoggingOperation(LoggingOperation &&) = default;

template<LogLevel Level>
LoggingOperation<Level> &LoggingOperation<Level>::operator=(LoggingOperation &&) = default;

template<LogLevel Level>
LoggingOperation<Level>::~LoggingOperation() {
    if (stream_ && should_output_) {
        if (addEndLine_)
            (*stream_) << "\n";
        stream_->flush();
    }
}

template<LogLevel Level>
std::string LoggingOperation<Level>::function() const {
    return function_;
}

template<LogLevel Level>
std::string LoggingOperation<Level>::file() const {
    return file_;
}

template<LogLevel Level>
int LoggingOperation<Level>::line() const {
    return line_;
}

template<LogLevel Level>
std::string LoggingOperation<Level>::prefix() const {
    return prefix_;
}

template<LogLevel Level>
void LoggingOperation<Level>::enableSpaceBetweenTokens() {
    addSpaceBetweenTokens_ = true;
}

template<LogLevel Level>
void LoggingOperation<Level>::disableSpaceBetweenTokens() {
    addSpaceBetweenTokens_ = false;
}

template<LogLevel Level>
void LoggingOperation<Level>::enableEndOfLineAtDestruction() {
    addEndLine_ = true;
}

template<LogLevel Level>
void LoggingOperation<Level>::disableEndOfLineAtDestruction() {
    addEndLine_ = false;
}

template<LogLevel Level>
template<typename T>
void LoggingOperation<Level>::log(const T &t) {
    if (stream_ && should_output_) {
        streambuf_->reset_output_sentinel();
        (*stream_) << t;
        if (streambuf_->get_output_sentinel()) {
            handleSpace();
        }
    }
}

template<LogLevel Level>
void LoggingOperation<Level>::log(bool b) {
    if (stream_ && should_output_) {
        (*stream_) << (b ? "true" : "false");

        handleSpace();
    }
}

template<LogLevel Level>
template<typename T>
void LoggingOperation<Level>::log(const std::vector<T> &v) {
    if (stream_ && should_output_) {
        (*stream_) << "[ ";
        using SizeType = typename std::vector<T>::size_type;
        for (SizeType i = 0; i < v.size() - 1; ++i) {
            (*stream_) << v[i] << ", ";
        }
        (*stream_) << v.back();
        (*stream_) << " ]";

        handleSpace();
    }
}

template<LogLevel Level>
void LoggingOperation<Level>::apply(std::ostream &(*manip)(std::ostream &)) {
    if (stream_ && should_output_) {
        (*stream_) << manip;
    }
}

template<LogLevel Level>
void LoggingOperation<Level>::handleSpace() {
    if (addSpaceBetweenTokens_)
        (*stream_) << " ";
}

template<LogLevel Level, typename T>
LoggingOperation<Level> &operator<<(LoggingOperation<Level> &op, const T &t) {
    op.log(t);
    return op;
}

template<LogLevel Level, typename T>
LoggingOperation<Level> &&operator<<(LoggingOperation<Level> &&op, const T &t) {
    op.log(t);
    return std::move(op);
}

template<LogLevel Level>
LoggingOperation<Level> &operator<<(LoggingOperation<Level> &op, std::ostream &(*manip)(std::ostream &)) {
    op.apply(manip);
    return op;
}

template<LogLevel Level>
LoggingOperation<Level> &&operator<<(LoggingOperation<Level> &&op, std::ostream &(*manip)(std::ostream &)) {
    op.apply(manip);
    return std::move(op);
}

template<LogLevel Level>
LoggingOperation<Level> &operator<<(LoggingOperation<Level> &op,
                                    LoggingOperation<Level> &(*f)(LoggingOperation<Level> &)) {
    (*f)(op);
    return op;
}

template<LogLevel Level>
LoggingOperation<Level> &&operator<<(LoggingOperation<Level> &&op,
                                     LoggingOperation<Level> && (*f)(LoggingOperation<Level> &&)) {
    (*f)(std::move(op));
    return std::move(op);
}

namespace Log {

template<LogLevel Level>
LoggingOperation<Level> &space(LoggingOperation<Level> &op) {
    op.enableSpaceBetweenTokens();
    return op;
}

template<LogLevel Level>
LoggingOperation<Level> &no_space(LoggingOperation<Level> &op) {
    op.disableSpaceBetweenTokens();
    return op;
}

template<LogLevel Level>
LoggingOperation<Level> &endline(LoggingOperation<Level> &op) {
    op.enableEndOfLineAtDestruction();
    return op;
}

template<LogLevel Level>
LoggingOperation<Level> &no_endline(LoggingOperation<Level> &op) {
    op.disableEndOfLineAtDestruction();
    return op;
}

template<LogLevel Level>
LoggingOperation<Level> &function(LoggingOperation<Level> &op) {
    op << op.function();
    return op;
}

template<LogLevel Level>
LoggingOperation<Level> &file(LoggingOperation<Level> &op) {
    op << op.file();
    return op;
}

template<LogLevel Level>
LoggingOperation<Level> &line(LoggingOperation<Level> &op) {
    op << op.line();
    return op;
}

template<LogLevel Level>
LoggingOperation<Level> &prefix(LoggingOperation<Level> &op) {
    op << op.prefix();
    return op;
}

template<LogLevel Level>
LoggingOperation<Level> &&space(LoggingOperation<Level> &&op) {
    op.enableSpaceBetweenTokens();
    return std::move(op);
}

template<LogLevel Level>
LoggingOperation<Level> &&no_space(LoggingOperation<Level> &&op) {
    op.disableSpaceBetweenTokens();
    return std::move(op);
}

template<LogLevel Level>
LoggingOperation<Level> &&endline(LoggingOperation<Level> &&op) {
    op.enableEndOfLineAtDestruction();
    return std::move(op);
}

template<LogLevel Level>
LoggingOperation<Level> &&no_endline(LoggingOperation<Level> &&op) {
    op.disableEndOfLineAtDestruction();
    return std::move(op);
}

template<LogLevel Level>
LoggingOperation<Level> &&function(LoggingOperation<Level> &&op) {
    op << op.function();
    return std::move(op);
}

template<LogLevel Level>
LoggingOperation<Level> &&file(LoggingOperation<Level> &&op) {
    op << op.file();
    return std::move(op);
}

template<LogLevel Level>
LoggingOperation<Level> &&line(LoggingOperation<Level> &&op) {
    op << op.line();
    return std::move(op);
}

template<LogLevel Level>
LoggingOperation<Level> &&prefix(LoggingOperation<Level> &&op) {
    op << op.prefix();
    return std::move(op);
}

} // namespace Log

} // namespace Metavision

#endif // METAVISION_SDK_BASE_DETAIL_LOG_IMPL_H
