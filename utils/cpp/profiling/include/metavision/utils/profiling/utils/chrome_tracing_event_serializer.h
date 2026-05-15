/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_UTILS_PROFILING_CHROME_TRACING_EVENT_SERIALIZER_H
#define METAVISION_UTILS_PROFILING_CHROME_TRACING_EVENT_SERIALIZER_H

#include <boost/variant.hpp>
#include <fstream>

namespace Profiling {

struct EventBase;
struct CompleteEvent;
struct InstantEvent;
struct CounterEvent;

/// @brief Class that logs profile events into a text file in the format that the Chrome Tracing tool expects
class ChromeTracingEventSerializer : boost::static_visitor<> {
public:
    /// @brief Constructor
    /// @param ofstream Output text file in which to log profile events
    ChromeTracingEventSerializer(std::ofstream &ofstream);

    void operator()(const EventBase &evt);

    void operator()(const CompleteEvent &evt);

    void operator()(const InstantEvent &evt);

    void operator()(const CounterEvent &evt);

private:
    std::ofstream &ofstream_;
};

} // namespace Profiling

#endif // METAVISION_UTILS_PROFILING_CHROME_TRACING_EVENT_SERIALIZER_H
