/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include "metavision/utils/profiling/utils/chrome_tracing_event_serializer.h"
#include "metavision/utils/profiling/utils/chrome_tracing_events.h"

namespace Profiling {

ChromeTracingEventSerializer::ChromeTracingEventSerializer(std::ofstream &ofstream) : ofstream_(ofstream) {}

void ChromeTracingEventSerializer::operator()(const EventBase &evt) {
    // clang-format off
        ofstream_ << "\"name\":\""   << evt.name_ << "\""
                  << ",\"ph\":\""  << evt.type_ << "\""
                  << ",\"ts\":"    << evt.elapsed_since_start_us_ 
                  << ",\"pid\":"   << 0;
    // clang-format on

    if (evt.thread_id_ != 0)
        ofstream_ << ",\"tid\":" << evt.thread_id_;

    if (evt.color_ != Constants::Color::Undefined)
        ofstream_ << ",\"cname\":\"" << Constants::kColorArray[evt.color_] << "\"";

    if (!evt.args_.empty()) {
        ofstream_ << ",\"args\":{";
        const auto &n_args = evt.args_.size();
        for (size_t i = 0; i < n_args; ++i) {
            const auto &arg = evt.args_[i];
            ofstream_ << "\"" << arg.name_ << "\":\"" << arg.value_ << "\"";

            if (i != n_args - 1)
                ofstream_ << ",";
        }
        ofstream_ << "}";
    }
}

void ChromeTracingEventSerializer::operator()(const CompleteEvent &evt) {
    (*this)(static_cast<const EventBase &>(evt));

    const std::int64_t duration = (evt.duration_us_ != 0) ? evt.duration_us_ : 1;
    ofstream_ << ",\"dur\":" << duration;
}

void ChromeTracingEventSerializer::operator()(const InstantEvent &evt) {
    (*this)(static_cast<const EventBase &>(evt));

    ofstream_ << ",\"s\":\"";
    ofstream_ << (evt.thread_scope_ ? Constants::kThreadScopeType : Constants::kGlobalScopeType);
    ofstream_ << "\"";
}

void ChromeTracingEventSerializer::operator()(const CounterEvent &evt) {
    (*this)(static_cast<const EventBase &>(evt));

    ofstream_ << ",\"id\":" << evt.id_;
}

} // namespace Profiling