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

#ifndef METAVISION_SDK_CORE_DETAIL_TIMING_PROFILER_DETAIL_H
#define METAVISION_SDK_CORE_DETAIL_TIMING_PROFILER_DETAIL_H

// define HAS_BOOST if boost is available on your system
#ifdef HAS_BOOST
#include <boost/format.hpp>
#include <boost/timer/timer.hpp>
#endif
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <metavision/sdk/base/utils/sdk_log.h>

namespace Metavision {
namespace detail {

/********************************************************************************
 * ConcurrencyPolicy : protect or not from concurrent access                    *
 ********************************************************************************/
class ConcurrencyPolicyThreadSafe {
public:
    using Mutex = std::mutex;
    using Lock  = std::unique_lock<Mutex>;
    Lock protect_scope() {
        return Lock(mutex_);
    }

private:
    Mutex mutex_;
};

class ConcurrencyPolicyThreadUnsafe {
public:
    struct Lock {
        ~Lock() {}
    };
    Lock protect_scope() {
        return Lock();
    }
};

/********************************************************************************
 * TimingPolicy : classes to define how to time operations (which timer to use) *
 ********************************************************************************/
struct CpuTimes {
    CpuTimes() : wall{}, user{}, system{}, has_user{false}, has_system{false} {}
    CpuTimes(const std::chrono::nanoseconds &wall) : wall{wall}, user{}, system{}, has_user(false), has_system(false) {}
    CpuTimes(const std::chrono::nanoseconds &wall, const std::chrono::nanoseconds &user,
             const std::chrono::nanoseconds &system) :
        wall{wall}, user{user}, system{system}, has_user{true}, has_system{true} {}

    CpuTimes &operator+=(const CpuTimes &times) {
        wall += times.wall;
        user += times.user;
        system += times.system;
        return *this;
    }

    std::chrono::nanoseconds wall, user, system;
    bool has_user, has_system;
};

inline CpuTimes operator+(const CpuTimes &t1, const CpuTimes &t2) {
    CpuTimes t(t1);
    t += t2;
    return t;
}

#ifdef HAS_BOOST
struct TimingPolicyUsingBoostCpuTimer {
    struct Timer {
        CpuTimes elapsed() const {
            const auto &times = t_.elapsed();
            return {std::chrono::nanoseconds(times.wall), std::chrono::nanoseconds(times.user),
                    std::chrono::nanoseconds(times.system)};
        }
        boost::timer::cpu_timer t_;
    };
};
#endif

struct TimingPolicyUsingSTL {
    struct Timer {
        Timer() : t_{std::chrono::high_resolution_clock::now()} {}
        CpuTimes elapsed() const {
            return {
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - t_)};
        }
        std::chrono::high_resolution_clock::time_point t_;
    };
};

/********************************************************************************
 * PrintingPolicy : classes to define how to print the timings                  *
 ********************************************************************************/
#ifdef HAS_BOOST
template<typename OperationStoragePolicy>
class PrinterUsingBoost {
public:
    PrinterUsingBoost(bool show_full_timings = false) :
        show_full_timings_{show_full_timings},
        size_time_{10},
        size_op_{20},
        size_duration_{show_full_timings ? 3 * size_time_ : size_time_},
        size_calls_{20},
        size_elements_{20},
        size_time_per_call_{show_full_timings ? 3 * size_time_ : size_time_},
        size_total_{size_op_ + size_duration_ + size_calls_ + size_elements_ + size_time_per_call_},
        header_fmter_{boost::str(boost::format("%%|=%1%|%%|=%2%|%%|=%3%|%%|=%4%|%%|=%5%|") % size_op_ % size_duration_ %
                                 size_calls_ % size_time_per_call_ % size_elements_)},
        subheader_fmter_{boost::str(boost::format("%%|=%1%|%2%%%|=%3%|%%|%4%%%|=%5%|") % size_op_ %
                                    (show_full_timings ? boost::str(boost::format("%%|=%1%|%%|=%2%|%%|=%3%|") %
                                                                    size_time_ % size_time_ % size_time_) :
                                                         boost::str(boost::format{"%%|=%1%|"} % size_time_)) %
                                    size_calls_ %
                                    (show_full_timings ? boost::str(boost::format("%%|=%1%|%%|=%2%|%%|=%3%|") %
                                                                    size_time_ % size_time_ % size_time_) :
                                                         boost::str(boost::format{"%%|=%1%|"} % size_time_)) %
                                    size_elements_)},
        row_fmter_{boost::str(boost::format("%%|-%1%|%2%%%|%3%|%%|%4%%%|%5%|") % size_op_ %
                              (show_full_timings ? boost::str(boost::format("%%|%1%.2f| %%|%2%.2f| %%|%3%.2f|") %
                                                              (size_time_ - 1) % (size_time_ - 1) % (size_time_ - 1)) :
                                                   boost::str(boost::format("%%|%1%.2f|") % size_time_)) %
                              size_calls_ %
                              (show_full_timings ? boost::str(boost::format("%%|%1%.1e| %%|%2%.1e| %%|%3%.1e|") %
                                                              (size_time_ - 1) % (size_time_ - 1) % (size_time_ - 1)) :
                                                   boost::str(boost::format("%%|%1%.1e|") % size_time_)) %
                              size_elements_)} {}

    void operator()(const OperationStoragePolicy &storage_policy) {
        const auto &keys = storage_policy.get_ordered_keys();
        if (keys.empty())
            return;

        auto log = MV_SDK_LOG_INFO() << Log::no_endline << Log::no_space
                                     << header_fmter_ % "Operation" % "Total duration" % "Number of calls" %
                                            "Time per call" % "Total elements"
                                     << "\n";
        if (show_full_timings_)
            log << Metavision::Log::prefix
                << subheader_fmter_ % "" % "(wall:ms)" % "(user:ms)" % "(sys:ms)" % "" % "(wall:us)" % "(user:us)" %
                       "(sys:us)" % ""
                << "\n";
        else
            log << Metavision::Log::prefix << subheader_fmter_ % "" % "ms" % "" % "us" % ""
                << "\n";

        log << Metavision::Log::prefix << std::setfill('-') << std::setw(size_total_) << "-"
            << "\n"
            << std::setfill(' ');

        for (const auto &key : keys) {
            const auto &time              = storage_policy.get_time(key);
            size_t count                  = storage_policy.get_count(key);
            size_t num_processed_elements = storage_policy.get_num_processed_elements(key);

            auto wall_dur = std::chrono::duration_cast<std::chrono::microseconds>(time.wall).count();
            if (show_full_timings_) {
                auto user_dur = std::chrono::duration_cast<std::chrono::microseconds>(time.user).count();
                auto sys_dur  = std::chrono::duration_cast<std::chrono::microseconds>(time.system).count();
                log << Metavision::Log::prefix
                    << row_fmter_ % key.substr(0, size_op_) % (wall_dur / 1000.0) % (user_dur / 1000.0) %
                           (sys_dur / 1000.0) % count % (double(wall_dur) / count) % (double(user_dur) / count) %
                           (double(sys_dur) / count) % num_processed_elements
                    << "\n";
            } else {
                log << Metavision::Log::prefix
                    << row_fmter_ % key.substr(0, size_op_) % (wall_dur / 1000.0) % count % (double(wall_dur) / count) %
                           num_processed_elements "\n";
            }
        }
        log << std::flush();
    }

private:
    bool show_full_timings_;
    int size_time_;
    int size_op_;
    int size_duration_;
    int size_calls_;
    int size_elements_;
    int size_time_per_call_;
    int size_total_;
    boost::format header_fmter_;
    boost::format subheader_fmter_;
    boost::format row_fmter_;
};
#endif

template<class OperationStoragePolicy>
class PrinterUsingSTL {
public:
    PrinterUsingSTL() :
        size_time_{10},
        size_op_{20},
        size_duration_{size_time_ * 2},
        size_calls_{20},
        size_elements_{20},
        size_time_per_call_{size_time_ * 2},
        size_total_{size_op_ + size_duration_ + size_calls_ + size_elements_ + size_time_per_call_} {}

    void operator()(const OperationStoragePolicy &storage_policy) {
        const auto &keys = storage_policy.get_ordered_keys();
        if (keys.empty())
            return;

        auto log = MV_SDK_LOG_INFO() << Metavision::Log::no_endline << Metavision::Log::no_space << std::setw(size_op_)
                                     << "Operation" << std::setw(size_duration_) << "Total duration"
                                     << std::setw(size_calls_) << "Number of calls" << std::setw(size_time_per_call_)
                                     << "Time per call" << std::setw(size_elements_) << "Total elements"
                                     << "\n";
        log << Metavision::Log::prefix << std::setw(size_op_) << "" << std::setw(size_duration_) << "(ms)"
            << std::setw(size_calls_) << "" << std::setw(size_time_per_call_) << "(us)" << std::setw(size_elements_)
            << ""
            << "\n";
        log << Metavision::Log::prefix << std::setw(size_op_) << std::setfill('-') << std::setw(size_total_) << ""
            << "\n"
            << std::setfill(' ');

        for (const auto &key : keys) {
            const auto &time              = storage_policy.get_time(key);
            size_t count                  = storage_policy.get_count(key);
            size_t num_processed_elements = storage_policy.get_num_processed_elements(key);

            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(time.wall).count();
            log << Metavision::Log::prefix << std::setw(size_op_) << key.substr(0, size_op_)
                << std::setw(size_duration_) << std::fixed << std::setprecision(2) << (dur / 1000)
                << std::setw(size_calls_) << count << std::setw(size_time_per_call_) << std::scientific
                << std::setprecision(1) << double(dur) / count << std::setw(size_elements_) << std::fixed
                << num_processed_elements << "\n";
        }
        log << std::flush;
    }

private:
    int size_time_;
    int size_op_;
    int size_duration_;
    int size_calls_;
    int size_elements_;
    int size_time_per_call_;
    int size_total_;
};

/********************************************************************************
 * OperationStoragePolicy : classes to define how to store the operation in     *
 * the map                                                                      *
 ********************************************************************************/
class OperationStoragePolicyInsertionOrder {
public:
    void insert(const std::string &op) {
        if (times_.find(op) == times_.end()) {
            times_[op]  = detail::CpuTimes{};
            counts_[op] = 0;
            order_.push_back(op);
            num_processed_elements_[op] = 0;
        }
    }

    void insert(const std::string &op, size_t num_processed_elements, const detail::CpuTimes &time) {
        if (times_.find(op) != times_.end()) {
            times_[op] += time;
            counts_[op] += 1;
            num_processed_elements_[op] += num_processed_elements;
        } else {
            times_[op]  = time;
            counts_[op] = 1;
            order_.push_back(op);
            num_processed_elements_[op] = num_processed_elements;
        }
    }

    const std::vector<std::string> &get_ordered_keys() const {
        return order_;
    }

    const detail::CpuTimes &get_time(const std::string &op, bool *found = nullptr) const {
        static detail::CpuTimes dummy;
        auto it = times_.find(op);
        if (it == times_.end()) {
            if (found)
                *found = false;
            return dummy;
        } else {
            if (found)
                *found = true;
            return it->second;
        }
    }

    size_t get_num_processed_elements(const std::string &op, bool *found = nullptr) const {
        static size_t dummy = 0;
        auto it             = num_processed_elements_.find(op);
        if (it == num_processed_elements_.end()) {
            if (found)
                *found = false;
            return dummy;
        } else {
            if (found)
                *found = true;
            return it->second;
        }
    }

    size_t get_count(const std::string &op, bool *found = nullptr) const {
        static size_t dummy = 0;
        auto it             = counts_.find(op);
        if (it == counts_.end()) {
            if (found)
                *found = false;
            return dummy;
        } else {
            if (found)
                *found = true;
            return it->second;
        }
    }

    std::vector<std::string> order_;
    std::unordered_map<std::string, detail::CpuTimes> times_;
    std::unordered_map<std::string, size_t> counts_, num_processed_elements_;
};

class OperationStoragePolicyLexicalOrder {
public:
    void insert(const std::string &op) {
        auto &t = data_[op];
        std::get<0>(t) += detail::CpuTimes{};
        std::get<1>(t) += 0;
        std::get<2>(t) += 0;
    }

    void insert(const std::string &op, size_t num_processed_elements, const detail::CpuTimes &time) {
        auto &t = data_[op];
        std::get<0>(t) += time;
        std::get<1>(t)++;
        std::get<2>(t) += num_processed_elements;
    }

    std::vector<std::string> get_ordered_keys() const {
        std::vector<std::string> keys;
        for (auto &p : data_) {
            keys.push_back(p.first);
        }
        return keys;
    }

    const detail::CpuTimes &get_time(const std::string &op, bool *found = nullptr) const {
        static detail::CpuTimes dummy;
        auto it = data_.find(op);
        if (it == data_.end()) {
            if (found)
                *found = false;
            return dummy;
        } else {
            if (found)
                *found = true;
            return std::get<0>(it->second);
        }
    }

    size_t get_num_processed_elements(const std::string &op, bool *found = nullptr) const {
        static size_t dummy = 0;
        auto it             = data_.find(op);
        if (it == data_.end()) {
            if (found)
                *found = false;
            return dummy;
        } else {
            if (found)
                *found = true;
            return std::get<2>(it->second);
        }
    }

    size_t get_count(const std::string &op, bool *found = nullptr) const {
        static size_t dummy = 0;
        auto it             = data_.find(op);
        if (it == data_.end()) {
            if (found)
                *found = false;
            return dummy;
        } else {
            if (found)
                *found = true;
            return std::get<1>(it->second);
        }
    }

private:
    std::map<std::string, std::tuple<detail::CpuTimes, size_t, size_t>> data_;
};

} // namespace detail
} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_TIMING_PROFILER_DETAIL_H
