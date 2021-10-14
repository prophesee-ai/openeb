/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include "metavision/utils/profiling/chrome_tracing_scope_completed_event.h"

namespace Profiling {

ScopeCompleteEvent::ScopeCompleteEvent(ChromeTracingProfiler &profiler, const std::string &name,
                                       const Constants::Color color, const std::vector<EventArgument> &args,
                                       bool add_1_us) :
    profiler_(profiler) {
    name_  = name;
    color_ = color;
    args_  = args;
    start_ = chrono::now();

    if (add_1_us)
        start_ += std::chrono::microseconds(1);
}

ScopeCompleteEvent::~ScopeCompleteEvent() {
    const auto now         = chrono::now();
    const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(now - start_).count();
    profiler_.add_complete_event(name_, start_, duration_us, color_, args_);
}

} // namespace Profiling