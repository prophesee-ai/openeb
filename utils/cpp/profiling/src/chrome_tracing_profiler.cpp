/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <iostream>
#include <cstddef>
#include <utility>
#include <thread>

#include "metavision/utils/profiling/chrome_tracing_profiler.h"
#include "metavision/utils/profiling/utils/chrome_tracing_event_serializer.h"

namespace Profiling {

#ifdef ENABLE_MAIN_PROFILER
Profiling::ChromeTracingProfiler main_profiler_("main_profiler",
                                                false); // Save method will have to be manually called
#endif

void ChromeTracingProfiler::start_duration_event(const std::string &name, const std::vector<EventArgument> &args) {
    EventBase evt;
    build_base_event(name, chrono::now(), Constants::Color::Undefined, args, evt);

    evt.type_ = Constants::kDurationEventBeginType;

    store_event(std::move(evt));
}

void ChromeTracingProfiler::end_duration_event(const std::string &name, const Constants::Color color,
                                               const std::vector<EventArgument> &args) {
    EventBase evt;
    build_base_event(name, chrono::now(), color, args, evt);

    evt.type_ = Constants::kDurationEventEndType;

    store_event(std::move(evt));
}

void ChromeTracingProfiler::add_complete_event(const std::string &name, const chrono::time_point &ts,
                                               std::int64_t duration_us, const Constants::Color color,
                                               const std::vector<EventArgument> &args) {
    CompleteEvent evt;
    build_base_event(name, ts, color, args, evt);

    evt.type_        = Constants::kCompleteEventType;
    evt.duration_us_ = duration_us;

    store_event(std::move(evt));
}

void ChromeTracingProfiler::add_instant_event(const std::string &name, bool thread_scope,
                                              const std::vector<EventArgument> &args) {
    InstantEvent evt;
    build_base_event(name, chrono::now(), Constants::Color::Undefined, args, evt);

    evt.type_         = Constants::kInstantEventType;
    evt.thread_scope_ = thread_scope;

    store_event(std::move(evt));
}

void ChromeTracingProfiler::add_counter_event(const std::string &name, const std::vector<EventArgument> &args,
                                              std::uint16_t id, const Constants::Color color) {
    CounterEvent evt;
    build_base_event(name, chrono::now(), color, args, evt);

    evt.type_ = Constants::kCounterEventType;
    evt.id_   = id;

    store_event(std::move(evt));
}

ChromeTracingProfiler::ChromeTracingProfiler(const std::string &name, bool save_on_destruction) {
    output_path_         = std::filesystem::temp_directory_path() / name;
    save_on_destruction_ = save_on_destruction;
    reset_impl();
}

ChromeTracingProfiler::~ChromeTracingProfiler() {
    if (save_on_destruction_) {
        save();
    }
}

inline void ChromeTracingProfiler::store_event(ProfileEvent &&event) {
    std::lock_guard<std::mutex> guard(lock_);
    profiles_.emplace_back(std::move(event));
}

void ChromeTracingProfiler::reset() {
    std::lock_guard<std::mutex> guard(lock_);
    reset_impl();
}

void ChromeTracingProfiler::save() {
    std::ofstream file_stream(output_path_);
    if (file_stream.is_open()) {
        file_stream << "{\"traceEvents\":[";

        ChromeTracingEventSerializer serializer(file_stream);

        {
            std::lock_guard<std::mutex> guard(lock_);
            const size_t size = profiles_.size();
            for (size_t i = 0; i < size; ++i) {
                const auto &evt = profiles_[i];

                file_stream << "{";
                boost::apply_visitor(serializer, evt);
                file_stream << "}";

                if (i != size - 1) {
                    file_stream << ",";
                }
            }
        }

        file_stream << "]}";
        std::cout << "ChromeTracingProfiler results saved to '" << output_path_ << "'" << std::endl;
    }
}

void ChromeTracingProfiler::reset_impl() {
    std::lock_guard<std::mutex> guard(lock_);
    profiles_.reserve(100000);
    profiles_.clear();
    start_ = chrono::now();
}

void ChromeTracingProfiler::build_base_event(const std::string &name, const chrono::time_point &ts,
                                             const Constants::Color color, const std::vector<EventArgument> &args,
                                             EventBase &evt) {
    evt.name_                   = name;
    evt.color_                  = color;
    evt.elapsed_since_start_us_ = std::chrono::duration_cast<std::chrono::microseconds>(ts - start_).count();
    evt.thread_id_              = std::hash<std::thread::id>{}(std::this_thread::get_id());
    evt.args_                   = args;
}

} // namespace Profiling