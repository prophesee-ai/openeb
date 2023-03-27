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

#ifndef METAVISION_SDK_BASE_LOG_H
#define METAVISION_SDK_BASE_LOG_H

#include <iostream>
#include <vector>
#include <memory>

#ifdef DOXYGEN_FULL_VERSION
/// @brief Returns a logging operation of DEBUG level
///
/// Convenience function to return a logging operation of DEBUG level using the current logging stream,
/// an optional prefix format @p prefixFmt and automatically adding an end of line token at the end of the logging
/// operation
///
/// @param prefixFmt (Optional) a format string that will be be output as the first message token
/// @return A logging operation of DEBUG level
#define MV_LOG_DEBUG(prefixFmt...)

/// @brief Returns a logging operation of TRACE level
///
/// Convenience macro to return a logging operation of TRACE level using the current logging stream,
/// an optional prefix format @p prefixFmt and automatically adding an end of line token at the end of the logging
/// operation
///
/// @param prefixFmt (Optional) a format string that will be be output as the first message token
/// @return A logging operation of TRACE level
#define MV_LOG_TRACE(prefixFmt...)

/// @brief Returns a logging operation of INFO level
///
/// Convenience function to return a logging operation of INFO level using the current logging stream,
/// an optional prefix format @p prefixFmt and automatically adding an end of line token at the end of the logging
/// operation
///
/// @param prefixFmt (Optional) a format string that will be be output as the first message token
/// @return A logging operation of INFO level
#define MV_LOG_INFO(prefixFmt...)

/// @brief Returns a logging operation of WARNING level
///
/// Convenience function to return a logging operation of WARNING level using the current logging stream,
/// an optional prefix format @p prefixFmt and automatically adding an end of line token at the end of the logging
/// operation
///
/// @param prefixFmt (Optional) a format string that will be be output as the first message token
/// @return A logging operation of WARNING level
#define MV_LOG_WARNING(prefixFmt...)

/// @brief Returns a logging operation of ERROR level
///
/// Convenience function to return a logging operation of ERROR level using the current logging stream,
/// an optional prefix format @p prefixFmt and automatically adding an end of line token at the end of the logging
/// operation
///
/// @param prefixFmt (Optional) a format string that will be be output as the first message token
/// @return A logging operation of ERROR level
#define MV_LOG_ERROR(prefixFmt...)
#endif

namespace Metavision {

/// @brief Enumeration used to control the level of logged messages that are allowed to pass through
enum class LogLevel {
    /// This level is reserved for internal debugging purposes
    /// @note These messages are simply ignored when the code is compiled in Release even if @ref setLogLevel is called
    /// with the Debug level
    Debug,
    /// This level should be used for external debugging purposes
    Trace,
    /// This level should be used for general information for the user
    Info,
    /// This level should be used when a potential problem could occur that requires the user attention
    Warning,
    /// This level should be used when a problem occurs, possibly leading to a failure of the application
    Error
};

/// @brief Get global level of logging
/// @sa @ref LogOptions::getLevel and @ref getLogOptions
LogLevel getLogLevel();

/// @brief Sets the current level of logging
/// @param level The minimum level of messages allowed to pass through
/// @sa @ref LogOptions::setLevel and @ref getLogOptions
void setLogLevel(const LogLevel &level);

/// @brief Resets the current level of logging value read from the environment variable MV_LOG_LEVEL
/// @sa @ref LogOptions::setLevel and @ref getLogOptions
void resetLogLevelFromEnv();

/// @brief Gets the current stream in which all messages are logged
/// @return The current stream in which all messages are logged
/// @sa @ref LogOptions::setStream and @ref getLogOptions
std::ostream &getLogStream();

/// @brief Sets the current stream in which all messages are logged
/// @param stream The stream in which all messages will be written
/// @sa @ref LogOptions::setStream and ref @ref getLogOptions
void setLogStream(std::ostream &stream);

/// @brief Resets the current logging stream read from the environment variable MV_LOG_FILE
void resetLogStreamFromEnv();

/// @brief Struct that defines the settings used for the logging behaviors
class LogOptions {
private:
    /// @brief the current level of logging
    LogLevel level_ = LogLevel::Info;

    /// @brief The stream in which message tokens will be written
    std::ostream *stream_ = &std::cerr;

    /// @brief When set to true, the log level block `[<level>]` will be padded with leading space,
    /// so that all level label will be displayed with a fixed length.
    bool level_prefix_padding_ = false;

public:
    /// @brief Construct a LogOptions object.
    /// @param level The current level of logging
    /// @param stream The stream that will be used to issue the logs
    /// @param level_prefix_padding If enabled, the [level] prefix is padded with whitespace to a fixed length
    LogOptions(LogLevel level = LogLevel::Info, std::ostream &stream = std::cerr, bool level_prefix_padding = false);

    /// @brief Sets the current level of logging
    ///
    /// Any message that has a higher or equal level will be enabled to pass through, and
    /// any message that has a lower level will be ignored
    ///
    /// @param level The minimum level of messages allowed to pass through
    /// @return The current LogOptions object
    /// @note By default, the level is LogLevel::Info
    /// @note It is also possible to set the current level of logging by setting the environment variable
    /// MV_LOG_LEVEL with one of the following (string) value : DEBUG, TRACE, INFO, WARNING, ERROR. If the
    /// environment variable is set, it will have precedence over the value set by this function.
    /// @note The environment variable MV_LOG_LEVEL is only read once at initialization of the logging utility,
    /// if the value of the environment variable is changed after, its value won't be reflected unless you explicitly
    /// call
    /// @ref resetLogLevelFromEnv
    /// @note In Android the environment variable for logging must be set by using one of the following commands:
    /// - works until next reboot, has higher priority to persist property
    ///
    ///       adb shell setprop debug.metavision.log.level <LEVEL>
    /// - root permissions requested, works permanently
    ///
    ///       adb shell setprop persist.metavision.log.level <LEVEL>
    /// @note To reset a property in Android environment use the following command:
    ///
    ///     adb shell setprop \<property_name\> \"\"
    LogOptions &setLevel(const LogLevel &level);

    /// @brief Gets the current level of logging
    /// @return The current level of logging
    /// @sa @ref setLevel
    LogLevel getLevel() const;

    /// @brief Sets the current stream in which all messages are logged
    /// @param stream The stream in which all messages will be written
    /// @return The current LogOptions object
    /// @note By default, the stream is std::cerr.
    /// @note If you want to log in a file, you can pass your own file stream, but you have to manage its life time.
    /// @note It is also possible to set the current stream to point to a file by setting the environment variable
    /// MV_LOG_FILE with the path corresponding to the desired log file. If the environment variable is set, it will
    /// have precedence over the value set by this function.
    /// @note The environment variable MV_LOG_FILE is only read once at initialization of the logging utility,
    /// if the value of the environment variable is changed after, its value won't be reflected unless you explicitly
    /// call
    /// @ref resetLogStreamFromEnv
    LogOptions &setStream(std::ostream &stream);

    /// @brief Gets the current stream in which all messages are logged
    /// @return The current stream in which all messages are logged
    std::ostream &getStream() const;

    /// @brief Define if the [level] prefix should be padded with white spaces
    /// @return The current LogOptions object
    LogOptions &setLevelPrefixPadding(bool is_padded);

    /// @brief Is the "[level]" prefix padded with white spaces
    /// @return a boolean that defines if the option is enabled
    bool isLevelPrefixPadding() const;
};

/// @brief define global options to tweak logging behavior
/// @param opts The LogOptions to be globally set
/// @sa @ref getLogOptions to retrieve current Log options
void setLogOptions(LogOptions opts);

/// @brief retrieve global logging options
/// @return a copy of the global LogOptions object
/// @sa @ref setLogOptions to set Log options
LogOptions getLogOptions();

/// @brief Set global logging options bask to its original state.
void resetLogOptions();

// Forward declaration
namespace detail {
class concurrent_ostreambuf;
}

/// @brief Base class for any logging operation
///
/// This is a base class easing the use of the logging system
/// This class uses classical C++ techniques to easily provide modifiers to automatically add spaces between messages
/// tokens or an end of line character when the operation is finished
/// It is much more convenient to use one of the @ref MV_LOG_DEBUG, @ref MV_LOG_TRACE, @ref
/// MV_LOG_INFO, @ref MV_LOG_WARNING or @ref MV_LOG_ERROR functions to return an instance of
/// this class
///
/// @tparam level The level of this logging operation
/// @sa @ref MV_LOG_DEBUG, @ref MV_LOG_TRACE, @ref MV_LOG_INFO, @ref MV_LOG_WARNING, @ref MV_LOG_ERROR
template<LogLevel level>
class LoggingOperation {
public:
    /// @brief The level of this operation
    static constexpr LogLevel Level = level;

    /// @brief Constructor
    ///
    /// Construct a LoggingOperation that will write in the @p stream message tokens, prefixed by the formatted
    /// @p prefixFmt
    ///
    /// The @p prefixFmt can be any fixed string or a format string with the following known replacement tokens :
    ///  - \<LEVEL\> : the level of the logging operation in upper case
    ///  - \<Level\> : the level of the logging operation in camel case
    ///  - \<FILE\> : the basename of the file where the logging operation is created
    ///  - \<LINE\> : the line where the logging operation is created
    ///  - \<FUNCTION\> : the function where the logging operation is created
    ///  - \<DATETIME:strftime_fmt\> : the date and time formatted with a format as specified by std::strftime function
    ///                                (e.g. %d%H%m), note that the formatted string can not exceed 1024 characters.
    ///
    /// @param opts The object that defines the general configuration for the logging mechanisms
    /// @param prefixFmt The prefix format to display
    /// @param file The file from which the logging operation was created
    /// @param line The line where the logging operation was created
    /// @param function The function in which the logging operation was created
    /// @note The replacement in the prefix format only occurs once, i.e. each token is searched only once and not
    /// replaced multiple times.
    LoggingOperation(const LogOptions &opts = LogOptions(), const std::string &prefixFmt = std::string(),
                     const std::string &file = std::string(), int line = 0,
                     const std::string &function = std::string());

    /// @brief Copy constructor
    /// A logging operation cannot be copy constructed
    LoggingOperation(const LoggingOperation &) = delete;

    /// @brief Move constructor
    /// A logging operation can be move constructed
    /// @param op The logging operation to be move constructed from
    LoggingOperation(LoggingOperation &&op);

    /// @brief Copy assignment
    /// A logging operation cannot be copy assigned
    LoggingOperation &operator=(const LoggingOperation &) = delete;

    /// @brief Move assignment
    /// A logging operation can be move assigned
    /// @param op The logging operation to be move assigned from
    /// @return The modified logging operation
    LoggingOperation &operator=(LoggingOperation &&op);

    /// @brief Destructor
    ~LoggingOperation();

    /// @brief Enables automatically adding spaces between message tokens
    /// @note This feature is enabled by default
    void enableSpaceBetweenTokens();

    /// @brief Disables automatically adding spaces between message tokens
    void disableSpaceBetweenTokens();

    /// @brief Enables automatically adding an end of line token when this operation is destroyed
    /// @note This feature is enabled by default
    void enableEndOfLineAtDestruction();

    /// @brief Disables automatically adding an end of line token when this operation is destroyed
    void disableEndOfLineAtDestruction();

    /// @brief Returns the name of the file associated to this logging operation
    std::string file() const;

    /// @brief Returns the line associated to this logging operation
    int line() const;

    /// @brief Returns the function associated to this logging operation
    std::string function() const;

    /// @brief Returns the prefix associated to this logging operation
    std::string prefix() const;

    /// @brief Logs the corresponding value
    /// @tparam T Type of the value to be logged
    /// @param t Value to be logged
    template<typename T>
    void log(const T &t);

    /// @brief Logs the corresponding value
    /// @overload
    void log(bool val);

    /// @brief Logs the corresponding value
    /// @overload
    template<typename T>
    void log(const std::vector<T> &v);

    /// @brief Applies a stream manipulator
    /// @param manip Stream manipulator
    void apply(std::ostream &(*manip)(std::ostream &));

private:
    void handleSpace();

    std::unique_ptr<detail::concurrent_ostreambuf> streambuf_;
    std::unique_ptr<std::ostream> stream_;
    bool addSpaceBetweenTokens_;
    bool addEndLine_;
    bool should_output_;
    std::string prefix_, file_, function_;
    int line_;
};

/// @brief Logs a value
/// @tparam Level The level of the logging operation
/// @tparam T Type of the message to be logged
/// @param op Logging operation to be logged to
/// @param t Value to be logged
/// @return The modified logging operation
template<LogLevel Level, typename T>
LoggingOperation<Level> &operator<<(LoggingOperation<Level> &op, const T &t);
/// @copydoc operator<<
template<LogLevel Level, typename T>
LoggingOperation<Level> &&operator<<(LoggingOperation<Level> &&op, const T &t);

/// @brief Applies a function to the logging operation
/// @tparam Level The level of the logging operation
/// @param op Logging operation to be modified
/// @param f Function modifying a logging operation
/// @return The modified logging operation
template<LogLevel Level>
LoggingOperation<Level> &operator<<(LoggingOperation<Level> &op,
                                    LoggingOperation<Level> &(*f)(LoggingOperation<Level> &));
/// @copydoc operator<<
template<LogLevel Level>
LoggingOperation<Level> &&operator<<(LoggingOperation<Level> &&op,
                                     LoggingOperation<Level> && (*f)(LoggingOperation<Level> &&));

/// @brief Applies a stream manipulator to logging operation
/// @tparam Level The level of the logging operation
/// @param op Logging operation to be modified
/// @param manip Stream manipulator
/// @return The modified logging operation
template<LogLevel Level>
LoggingOperation<Level> &operator<<(LoggingOperation<Level> &op, std::ostream &(*manip)(std::ostream &));
/// @copydoc operator<<
template<LogLevel Level>
LoggingOperation<Level> &&operator<<(LoggingOperation<Level> &&op, std::ostream &(*manip)(std::ostream &));

namespace Log {
/// @brief Stream manipulator enabling the automatic addition of spaces between message tokens
/// @tparam Level The level of the logging operation
/// @param op The logging operation modified by this modifier
/// @return The modified logging operation
template<LogLevel Level>
LoggingOperation<Level> &space(LoggingOperation<Level> &op);
/// @copydoc space
template<LogLevel Level>
LoggingOperation<Level> &&space(LoggingOperation<Level> &&op);

/// @brief Stream manipulator disabling the automatic addition of spaces between message tokens
/// @tparam Level The level of the logging operation
/// @param op The logging operation modified by this modifier
/// @return The modified logging operation
template<LogLevel Level>
LoggingOperation<Level> &no_space(LoggingOperation<Level> &op);
/// @copydoc no_space
template<LogLevel Level>
LoggingOperation<Level> &&no_space(LoggingOperation<Level> &&op);

/// @brief Stream manipulator enabling the automatic addition of an end of line token at the end of the operation
/// @tparam Level The level of the logging operation
/// @param op The logging operation modified by this modifier
/// @return The modified logging operation
template<LogLevel Level>
LoggingOperation<Level> &endline(LoggingOperation<Level> &op);
/// @copydoc endline
template<LogLevel Level>
LoggingOperation<Level> &&endline(LoggingOperation<Level> &&op);

/// @brief Stream manipulator disabling the automatic addition of an end of line token at the end of the operation
/// @tparam Level The level of the logging operation
/// @param op The logging operation modified by this modifier
/// @return The modified logging operation
template<LogLevel Level>
LoggingOperation<Level> &no_endline(LoggingOperation<Level> &op);
/// @copydoc no_endline
template<LogLevel Level>
LoggingOperation<Level> &&no_endline(LoggingOperation<Level> &&op);

/// @brief Stream manipulator that outputs the function name associated to the logging operation
/// @tparam Level The level of the logging operation
/// @param op The logging operation modified by this modifier
/// @return The modified logging operation
template<LogLevel Level>
LoggingOperation<Level> &function(LoggingOperation<Level> &op);
/// @copydoc function
template<LogLevel Level>
LoggingOperation<Level> &&function(LoggingOperation<Level> &&op);

/// @brief Stream manipulator that outputs the file name associated to the logging operation
/// @tparam Level The level of the logging operation
/// @param op The logging operation modified by this modifier
/// @return The modified logging operation
template<LogLevel Level>
LoggingOperation<Level> &file(LoggingOperation<Level> &op);
/// @copydoc file
template<LogLevel Level>
LoggingOperation<Level> &&file(LoggingOperation<Level> &&op);

/// @brief Stream manipulator that outputs the line associated to the logging operation
/// @tparam Level The level of the logging operation
/// @param op The logging operation modified by this modifier
/// @return The modified logging operation
template<LogLevel Level>
LoggingOperation<Level> &line(LoggingOperation<Level> &op);
/// @copydoc line
template<LogLevel Level>
LoggingOperation<Level> &&line(LoggingOperation<Level> &&op);

/// @brief Stream manipulator that outputs the prefix associated to the logging operation
/// @tparam Level The level of the logging operation
/// @param op The logging operation modified by this modifier
/// @return The modified logging operation
template<LogLevel Level>
LoggingOperation<Level> &prefix(LoggingOperation<Level> &op);
/// @copydoc prefix
template<LogLevel Level>
LoggingOperation<Level> &&prefix(LoggingOperation<Level> &&op);

} // namespace Log

} // namespace Metavision

#include "detail/log_impl.h"

#endif // METAVISION_SDK_BASE_LOG_H
