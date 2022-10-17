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

#ifndef METAVISION_SDK_CORE_TIME_SURFACE_PRODUCER_ALGORITHM_H
#define METAVISION_SDK_CORE_TIME_SURFACE_PRODUCER_ALGORITHM_H

#include <functional>
#include <type_traits>

#include "metavision/sdk/core/algorithms/async_algorithm.h"
#include "metavision/sdk/core/utils/mostrecent_timestamp_buffer.h"

namespace Metavision {

/// @brief Class that produces a @ref MostRecentTimestampBuffer (a.k.a. time surface) from events
///
/// This algorithm is asynchronous in the sense that it can be configured to produce a time surface every N events,
/// every N microseconds or a mixed condition of both (see @ref AsyncAlgorithm).
///
/// Like in other asynchronous algorithms, in order to retrieve the produced time surface, the user needs to set a
/// callback that will be called when the above condition is fulfilled. However, as opposed to other algorithms,
/// the user doesn't have here the capacity to take ownership of the produced time surface (using a swap mechanism
/// for example). Indeed, swapping the time surface would make the producer lose the whole history. If the user needs
/// to use the time surface out of the output callback, then a copy must be done.
///
/// @tparam CHANNELS Number of channels to use for producing the time surface. Only two values are possible for now: 1
/// or 2. When a 1-channel time surface is used, events with different polarities are stored all together while they are
/// stored separately when using a 2-channels time surface.
template<int CHANNELS = 1>
class TimeSurfaceProducerAlgorithm : public AsyncAlgorithm<TimeSurfaceProducerAlgorithm<CHANNELS>> {
public:
    static_assert(CHANNELS == 1 || CHANNELS == 2, "The timesurface producer is only compatible with 1 or 2 channels");

    using OutputCb = std::function<void(timestamp, const MostRecentTimestampBuffer &)>;

    /// @brief Constructs a new time surface producer
    /// @param width Sensor's width
    /// @param height Sensor's height
    TimeSurfaceProducerAlgorithm(int width, int height);

    /// @brief Sets a callback to retrieve the produced time surface
    ///
    /// A constant reference of the internal time surface is passed to the callback, allowing to process
    /// (i.e. read only) it inside the callback. If the time surface needs to be accessed from outside the callback,
    /// then a copy must be done.
    ///
    /// @param cb The callback called when the time surface is ready
    void set_output_callback(const OutputCb &cb);

private:
    friend class AsyncAlgorithm<TimeSurfaceProducerAlgorithm<CHANNELS>>;

    /// @brief Updates the time surface with the input events
    /// @tparam InputIt Type of the iterators pointing to the events
    /// @param it_begin Iterator pointing to the beginning of the events buffer
    /// @param it_end Iterator pointing to the end of the events buffer
    template<typename InputIt>
    inline void process_online(InputIt it_begin, InputIt it_end);

    /// @brief Calls the output callback when the time surface is ready (the output condition is satisfied)
    void process_async(const timestamp processing_ts, const size_t n_processed_events);

    MostRecentTimestampBuffer time_surface_; ///< Time surface updated internally
    OutputCb output_cb_;                     ///< Callback called when the time surface is ready
};
} // namespace Metavision

#include "metavision/sdk/core/algorithms/detail/time_surface_producer_algorithm_impl.h"

#endif // METAVISION_SDK_CORE_TIME_SURFACE_PRODUCER_ALGORITHM_H
