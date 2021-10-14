/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_UTILS_PROFILING_CHROME_TRACING_SCOPE_COMPLETED_EVENT_H
#define METAVISION_UTILS_PROFILING_CHROME_TRACING_SCOPE_COMPLETED_EVENT_H

#include "metavision/utils/profiling/chrome_tracing_profiler.h"

#define CHROME_PROFILE_SCOPE(name) Profiling::ScopeCompleteEvent profile##__LINE__(Profiling::main_profiler_, name)
#define CHROME_PROFILE_FUNCTION() CHROME_PROFILE_SCOPE(__PRETTY_FUNCTION__)

namespace Profiling {

/// @brief Profile event that measures the computation time of a given scope
///
/// The idea is to tie a timer to the lifetime of an object created on the stack.
/// When the object gets created, we start the timer, when the object gets deleted, we stop the timer
/// (Resource acquisition is initialization (RAII))
///
/// When it gets destroyed the class will feed its timing information to an instance of @ref ChromeTracingProfiler
///
/// Don't forget to save the results before exiting the program: Profiling::main_profiler_.save();
class ScopeCompleteEvent {
public:
    using chrono = ChromeTracingProfiler::chrono;

    /// @brief Constructor
    /// @param profiler Class adding the current scope event to profiling logs compatible with the Chrome Tracing Viewer
    /// @param name Name of the timeslice to display in the Chrome Tracing Viewer
    /// @param color Color of the slice in the tracing view. Constants::Color::Undefined means that we don't log the
    /// color information and let the Chrome Tracing Viewer pick a random color
    /// @param args Additional information stored in the timeslice as a dictionary for the Chrome Tracing Viewer
    /// @param add_1_us Avoid artifacts in the Chrome Tracing Viewer when a scope event lasting less than 1us is
    /// immediately followed by another event. Otherwise Chrome isn't able to determine which one started first
    ScopeCompleteEvent(ChromeTracingProfiler &profiler, const std::string &name,
                       const Constants::Color color           = Constants::Color::Undefined,
                       const std::vector<EventArgument> &args = std::vector<EventArgument>(), bool add_1_us = false);

    /// @brief Destructor
    /// @note The class stops the timer and feeds the timing information to the profiler
    ~ScopeCompleteEvent();

private:
    ChromeTracingProfiler &profiler_;
    std::string name_;
    Constants::Color color_;
    std::vector<EventArgument> args_;

    chrono::time_point start_;
};

} // namespace Profiling

#endif // METAVISION_UTILS_PROFILING_CHROME_TRACING_SCOPE_COMPLETED_EVENT_H
