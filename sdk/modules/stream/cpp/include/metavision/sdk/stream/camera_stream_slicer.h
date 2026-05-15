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

#ifndef METAVISION_SDK_STREAM_CAMERA_STREAM_SLICER_H
#define METAVISION_SDK_STREAM_CAMERA_STREAM_SLICER_H

#include <filesystem>
#include <optional>

#include "metavision/sdk/base/utils/object_pool.h"
#include "metavision/sdk/core/algorithms/event_buffer_reslicer_algorithm.h"
#include "metavision/sdk/core/utils/concurrent_queue.h"
#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/slice_iterator.h"

namespace Metavision {
using EventBuffer   = std::vector<EventCD>;
using TriggerBuffer = std::vector<EventExtTrigger>;

/// @brief Structure representing a slice of events and triggers
struct Slice {
    using ConditionStatus = EventBufferReslicerAlgorithm::ConditionStatus;

    /// @brief Comparison operator
    /// @param other Slice to compare with
    /// @return True if the two slices are equal, false otherwise
    bool operator==(const Slice &other) const;

    ConditionStatus status;                        ///< Status indicating how the slice was completed
    timestamp t;                                   ///< Timestamp of the slice
    std::size_t n_events;                          ///< Number of CD events in the slice
    std::shared_ptr<const EventBuffer> events;     ///< Events in the slice
    std::shared_ptr<const TriggerBuffer> triggers; ///< Triggers in the slice
};

/// @brief Class that slices a stream of events and triggers according to a given condition
///
/// Internally, a concurrent queue is used to store the slices produced by the background threads of the @ref Camera
/// class (i.e. in the callbacks). The size of this concurrent queue can be limited to prevent the background threads
/// from producing too many slices (i.e. especially in offline mode) and consuming too much memory.
class CameraStreamSlicer {
public:
    using SliceCondition = EventBufferReslicerAlgorithm::Condition;
    using SliceIterator  = SliceIteratorT<Slice>;

    /// @brief Default constructor
    CameraStreamSlicer() = default;

    /// @brief Constructor
    /// @param camera Camera instance to slice, the ownership of the camera is transferred to the slicer
    /// @param slice_condition Slicing parameters
    /// @param max_queue_size Maximum number of slices that can be stored in the internal queue
    CameraStreamSlicer(Camera &&camera, const SliceCondition &slice_condition = SliceCondition::make_n_us(1000),
                       size_t max_queue_size = 5);

    /// @brief Move constructor
    /// @param slicer CameraStreamSlicer to move
    CameraStreamSlicer(CameraStreamSlicer &&slicer) noexcept;

    /// @brief Move assignment operator
    /// @param slicer CameraStreamSlicer to move
    CameraStreamSlicer &operator=(CameraStreamSlicer &&slicer) noexcept;

    CameraStreamSlicer(const CameraStreamSlicer &) = delete;
    CameraStreamSlicer &operator=(const CameraStreamSlicer &) = delete;

    /// @brief Destructor
    ~CameraStreamSlicer();

    /// @brief Starts the camera stream and returns an iterator to the first slice
    SliceIterator begin();

    /// @brief Returns an iterator to the end of the stream
    SliceIterator end();

    /// @brief Returns the underlying camera instance
    [[nodiscard]] const Camera &camera() const;

private:
    void init_slicing();

    std::shared_ptr<ConcurrentQueue<Slice>> queue_;
    SharedObjectPool<std::vector<EventCD>> event_buffer_pool_;
    SharedObjectPool<std::vector<EventExtTrigger>> trigger_buffer_pool_;
    std::shared_ptr<EventBuffer> curt_event_buffer_;
    std::shared_ptr<TriggerBuffer> curt_trigger_buffer_;
    EventBufferReslicerAlgorithm slicer_;
    Camera camera_;
};

} // namespace Metavision

#endif // METAVISION_SDK_STREAM_CAMERA_STREAM_SLICER_H
