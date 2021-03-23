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

#ifndef METAVISION_SDK_CORE_TIMING_PROFILER_H
#define METAVISION_SDK_CORE_TIMING_PROFILER_H

#include <type_traits>
#include <functional>

#include "detail/timing_profiler_detail.h"

namespace Metavision {

/// @brief Class used for profiling algorithms
template<bool do_profiling = true, typename ConcurrencyPolicy = detail::ConcurrencyPolicyThreadSafe,
         typename OperationStoragePolicy = detail::OperationStoragePolicyInsertionOrder,
         typename TimingPolicy           = detail::TimingPolicyUsingSTL>
class TimingProfiler {
public:
    /// @brief Class that represents a timed operation to be accounted for by a profiler.
    /// Can be constructed with a Utils::TimingProfiler instance or not.
    /// If no instance is provided, it will use a single static instance of Utils::TimingProfiler that will
    /// be printed at the end of the program.
    class TimedOperation {
    public:
        /// @brief Constructor for TimedOperation.
        /// @param op Name of the operation
        /// @param num_processed_elements Number of elements processed by the operation
        /// @param profiler Instance of the profiler to use
        TimedOperation(const std::string &op, size_t num_processed_elements = 0,
                       TimingProfiler *profiler = TimingProfiler::instance()) :
            op_{op}, num_processed_elements_{num_processed_elements}, profiler_{profiler} {
            profiler->add_label(op);
        }

        /// @brief Constructor for TimedOperation.
        /// @param op Name of the operation
        /// @param profiler Instance of the profiler to use
        TimedOperation(const std::string &op, TimingProfiler *profiler) : TimedOperation(op, 0, profiler) {}

        /// @brief Destructor
        ~TimedOperation() {
            if (profiler_)
                profiler_->add_timed_operation(op_, num_processed_elements_, t_.elapsed());
        }

        /// @brief Sets the number of elements processed by the operation.
        /// @param num_processed_elements Number of elements processed by the operation
        void setNumProcessedElements(size_t num_processed_elements) {
            num_processed_elements_ = num_processed_elements;
        }

        friend class TimingProfiler;

    private:
        std::string op_;
        size_t num_processed_elements_;
        TimingProfiler *profiler_;
        typename TimingPolicy::Timer t_;
    };

    /// @brief Constructor
    /// @param storage_policy Instance of the storage policy (instantiates a new one by default)
    TimingProfiler(const OperationStoragePolicy &storage_policy = OperationStoragePolicy()) :
        storage_policy_{storage_policy}, printing_cb_{detail::PrinterUsingSTL<OperationStoragePolicy>()} {}

    /// @brief Destructor
    ~TimingProfiler() {
        auto protection = concurrency_policy_.protect_scope();
        printing_cb_(storage_policy_);
    }

    /// @brief Adds label to the profiler session
    /// @param op Name of the label to add
    void add_label(const std::string &op) {
        auto protection = concurrency_policy_.protect_scope();
        storage_policy_.insert(op);
    }

    /// @brief Adds operation to the profiler session
    /// @param op Name of the operation to add
    /// @param num_processed_elements Number of elements processed by the operation
    /// @param times Object counting the time taken by the operation
    void add_timed_operation(const std::string &op, size_t num_processed_elements, const detail::CpuTimes &times) {
        auto protection = concurrency_policy_.protect_scope();
        storage_policy_.insert(op, num_processed_elements, times);
    }

    /// @brief Gets the singleton instance of the profiler
    /// @return The profiler's singleton instance
    static TimingProfiler *instance() {
        static TimingProfiler profiler;
        return &profiler;
    }

    /// @brief Gets a copy of the storage policy
    /// @return The storage policy used
    const OperationStoragePolicy get_storage_policy() const {
        auto protection = concurrency_policy_.protect_scope();
        return storage_policy_;
    }

    /// @brief Sets callback called by the profiler's Destructor
    ///
    /// The storage policy is passed to the callback so it may access the data saved by the profiler
    /// until right before it finished running.
    void set_printing_callback(const std::function<void(const OperationStoragePolicy &)> &cb) {
        printing_cb_ = cb;
    }

private:
    OperationStoragePolicy storage_policy_;
    mutable ConcurrencyPolicy concurrency_policy_;
    std::function<void(const OperationStoragePolicy &)> printing_cb_;
};

/// @brief Helper to get a certain template Specialization of @ref Metavision::TimingProfiler base on a runtime
/// parameter, eg.:
///
/// Utils::TimingProfilerPair<> p;
/// if (profile) profile_with(p.get_profiler<true>());
/// else profile_with(p.get_profiler<false>());
template<typename ConcurrencyPolicy      = detail::ConcurrencyPolicyThreadSafe,
         typename OperationStoragePolicy = detail::OperationStoragePolicyInsertionOrder,
         typename TimingPolicy           = detail::TimingPolicyUsingSTL>
class TimingProfilerPair {
public:
    template<bool T, typename std::enable_if<T>::type * = nullptr>
    TimingProfiler<true, ConcurrencyPolicy, OperationStoragePolicy, TimingPolicy> *get_profiler() {
        return &op_;
    }

    template<bool T, typename std::enable_if<!T>::type * = nullptr>
    TimingProfiler<false, ConcurrencyPolicy, OperationStoragePolicy, TimingPolicy> *get_profiler() {
        return &noop_;
    }

private:
    TimingProfiler<true, ConcurrencyPolicy, OperationStoragePolicy, TimingPolicy> op_;
    TimingProfiler<false, ConcurrencyPolicy, OperationStoragePolicy, TimingPolicy> noop_;
};

/// @brief Template Specialization of @ref Metavision::TimingProfiler meant to be used in release builds.
/// Any use of this class should be nicely optimized as a no-op
/// Utils::TimingProfiler<false, ...>::TimedOperation::TimedOperation makes Doxygen fail
/// so we disable documentation generation for this class.
/// @cond false
template<typename ConcurrencyPolicy, typename OperationStoragePolicy, typename TimingPolicy>
class TimingProfiler<false, ConcurrencyPolicy, OperationStoragePolicy, TimingPolicy> {
public:
    struct TimedOperation {
        TimedOperation(const std::string &, size_t, TimingProfiler *) {}
        TimedOperation(const std::string &, size_t) {}
        TimedOperation(const std::string &, TimingProfiler *) {}
        TimedOperation(const std::string &) {}
        TimedOperation(const char *const, size_t, TimingProfiler *) {}
        TimedOperation(const char *const, size_t) {}
        TimedOperation(const char *const, TimingProfiler *) {}
        TimedOperation(const char *const) {}

        void setNumProcessedElements(size_t) {}
    };

    TimingProfiler() {}
    TimingProfiler(const OperationStoragePolicy &) {}

    template<typename T>
    void set_printing_callback(const T &) {}

    static TimingProfiler *instance() {
        return nullptr;
    }
};
/// @endcond

} // namespace Metavision

#endif // METAVISION_SDK_CORE_TIMING_PROFILER_H
