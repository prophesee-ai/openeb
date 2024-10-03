/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_UTILS_PROFILING_CHROME_TRACING_PROFILER_H
#define METAVISION_UTILS_PROFILING_CHROME_TRACING_PROFILER_H

#include <cstdint>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <sstream>

#include <boost/variant.hpp>

#include "metavision/utils/profiling/utils/chrome_tracing_events.h"

#define ENABLE_MAIN_PROFILER // Enables the instantiation of a profiler global instance.

namespace Profiling {

/// @brief Class that generates profiling logs in a json format that is compatible with the Chrome Tracing Viewer
///
/// There is a global instance of the class, called Profiling::main_profiler_, so that it can be used everywhere to
/// profile code, while having all profiling logs in a single file.
///
/// Don't forget to save the results before exiting the program: Profiling::main_profiler_.save();
///
/// To open the logs in chrome:
///     - Just tap "chrome://tracing" in the address bar,
///     - click on the "Load" button, select your file and voilà!
class ChromeTracingProfiler {
public:
    using chrono = std::chrono::high_resolution_clock;

    /// @brief Constructor
    /// @param name Name of the profiler. This name will be used to generate the name of the
    /// file into which logs will be saved.
    /// @param save_on_destruction Flag that indicates if the logs have to be automatically saved
    /// when the profiler is destructed. Ideally this should be the case all the time but some problems
    /// appear when we use a global instance and python bindings. So we let the capacity to save the file
    /// before the profiler destruction.
    ChromeTracingProfiler(const std::string &name, bool save_on_destruction = true);

    /// @brief Destructor
    ~ChromeTracingProfiler();

    /// @brief Starts a duration slice
    ///
    /// Duration events provide a way to mark a duration of work on a given thread
    /// @note The start and end events are decorrelated from the scopes in which they are created. They can also
    /// belong to different scopes
    /// @warning The start event must come before its corresponding end event
    /// @param name Name of the slice (Key to link to the corresponding end event)
    /// @param args Additional information that will be stored in the timeslice as a dictionary
    void start_duration_event(const std::string &name,
                              const std::vector<EventArgument> &args = std::vector<EventArgument>());

    /// @brief Ends a duration slice
    ///
    /// Duration events provide a way to mark a duration of work on a given thread
    /// @note The start and end events are decorrelated from the scopes in which they are created. They can also
    /// belong to different scopes
    /// @warning The end event must come after its corresponding start event
    /// @param name Name of the slice (Key to link to the corresponding start event)
    /// @param color Color of the slice in the tracing view. Constants::Color::Undefined means that we don't log the
    /// color information and let the Chrome Tracing Viewer pick a random color
    /// @param args Additional information that will be stored in the timeslice as a dictionary
    void end_duration_event(const std::string &name, const Constants::Color color = Constants::Color::Undefined,
                            const std::vector<EventArgument> &args = std::vector<EventArgument>());

    /// @brief Adds a complete duration slice
    ///
    /// Duration events provide a way to mark a duration of work on a given thread.
    /// Unlike @ref start_duration_event and @ref end_duration_event, both start and end events are specified at the
    /// same time
    /// @param name Name of the slice
    /// @param ts Time of the start of the timeslice
    /// @param duration_us Timeslice duration in microseconds
    /// @param color Color of the slice in the tracing view. Constants::Color::Undefined means that we don't log the
    /// color information and let the Chrome Tracing Viewer pick a random color
    /// @param args Additional information that will be stored in the timeslice as a dictionary
    void add_complete_event(const std::string &name, const chrono::time_point &ts, std::int64_t duration_us,
                            const Constants::Color color           = Constants::Color::Undefined,
                            const std::vector<EventArgument> &args = std::vector<EventArgument>());

    /// @brief Adds an instant event
    ///
    /// Instant events correspond to something that happens but has no duration associated with it
    /// @param name Name of the instant event
    /// @param thread_scope Flag indicating how tall to draw the instant event in the visualization. If true, draw
    /// the height of a single thread. Otherwise, draw through all threads. By default, the instant event is
    /// thread-scoped
    /// @param args Additional information that will be stored in the instant event as a dictionary
    void add_instant_event(const std::string &name, bool thread_scope = true,
                           const std::vector<EventArgument> &args = std::vector<EventArgument>());

    /// @brief Adds a counter event
    ///
    /// Counter events can track a value or multiple values as they change over time
    /// @param name Name of the counter event
    /// @param ts Time of the start of the timeslice
    /// @param args Multiple series of data to display
    /// @param id The combination of the event name and id is used as the full counter name
    /// @param color Color of the slice in the tracing view. Constants::Color::Undefined means that we don't log the
    /// color information and let the Chrome Tracing Viewer pick a random color
    void add_counter_event(const std::string &name, const std::vector<EventArgument> &args, std::uint16_t id = 0,
                           const Constants::Color color = Constants::Color::Undefined);

    /// @brief Clears all previous logs
    void reset();

    /// @brief Saves the profiling logs into a file
    void save();

private:
    using ProfileEvent = boost::variant<EventBase, CompleteEvent, InstantEvent, CounterEvent>;

    /// @brief Clears all previous logs
    void reset_impl();

    void build_base_event(const std::string &name, const chrono::time_point &ts, const Constants::Color color,
                          const std::vector<EventArgument> &args, EventBase &evt);

    void store_event(ProfileEvent &&event);

    std::filesystem::path output_path_;  ///< Path to the file where profiling logs will be saved
    bool save_on_destruction_; ///< Flag to indicate if the profiling logs have to be saved when the destructor is
                               ///< called
    chrono::time_point start_; ///< Starting timestamp
    std::mutex lock_;          ///< Mutex to prevent concurrency
    std::vector<ProfileEvent> profiles_; ///< All recorded profiles
};

template<typename T>
inline EventArgument create_profile_argument(const std::string &name, const T &value) {
    EventArgument arg;
    arg.name_  = name;
    arg.value_ = std::to_string(value);

    return arg;
}

template<>
inline EventArgument create_profile_argument<std::string>(const std::string &name, const std::string &value) {
    EventArgument arg;
    arg.name_  = name;
    arg.value_ = value;

    return arg;
}

#ifdef ENABLE_MAIN_PROFILER
/// @brief Profiler global instance used to profile code anywhere
/// and to have the profiling logs saved to a single file
extern ChromeTracingProfiler main_profiler_;
#endif

} // namespace Profiling

#endif // METAVISION_UTILS_PROFILING_CHROME_TRACING_PROFILER_H
