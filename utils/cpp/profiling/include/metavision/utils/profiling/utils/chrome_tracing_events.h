/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_UTILS_PROFILING_CHROME_TRACING_EVENTS_H
#define METAVISION_UTILS_PROFILING_CHROME_TRACING_EVENTS_H

#include "metavision/utils/profiling/utils/chrome_tracing_constants.h"

namespace Profiling {

/// @brief Additional information that can be stored in the args_ dictionary of a profile event
struct EventArgument {
    std::string name_;
    std::string value_;
};

/// @brief Base profile event
struct EventBase {
    std::string name_;
    char type_;
    Constants::Color color_;
    std::int64_t elapsed_since_start_us_;
    uint32_t thread_id_;
    std::vector<EventArgument> args_;
};

/// @brief Complete duration slice event
struct CompleteEvent : EventBase {
    std::int64_t duration_us_;
};

/// @brief Instant event, i.e. something that happens but has no duration associated with it
struct InstantEvent : EventBase {
    bool thread_scope_;
};

/// @brief Counter event that tracks a value or multiple values as they change over time
struct CounterEvent : EventBase {
    std::uint16_t id_;
};

} // namespace Profiling

#endif // METAVISION_UTILS_PROFILING_CHROME_TRACING_EVENTS_H
