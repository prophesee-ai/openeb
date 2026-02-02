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

#ifndef METAVISION_SDK_DRIVER_SYNCED_CAMERA_STREAM_SLICER_H
#define METAVISION_SDK_DRIVER_SYNCED_CAMERA_STREAM_SLICER_H

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

/// @brief A slice of synchronized events from a master and slave cameras.
struct SyncedSlice {
    using ConditionStatus = EventBufferReslicerAlgorithm::ConditionStatus;

    /// @brief Default constructor
    SyncedSlice() = default;

    /// @brief Comparison operator
    /// @param other Slice to compare with
    /// @return True if the two slices are equal, false otherwise
    bool operator==(const SyncedSlice &other) const;

    ConditionStatus status = ConditionStatus::NOT_MET;            ///< Status indicating how the slice was completed
    timestamp t            = 0;                                   ///< Timestamp of the slice
    std::size_t n_events   = 0;                                   ///< Number of events in the master slice
    std::shared_ptr<const EventBuffer> master_events;             ///< Events in the master slice
    std::shared_ptr<const TriggerBuffer> master_triggers;         ///< Triggers in the master slice
    std::vector<std::shared_ptr<const EventBuffer>> slave_events; ///< Events in the slave slices
};

/// @brief Class for slicing event streams from master and slave cameras based on a condition.
///
/// The slicing condition is applied to the master stream and the end of the slave streams are determined accordingly.
/// Internally, a concurrent queue is used to store the slices produced by the background threads of the master @ref
/// Camera class (i.e. in the callbacks). The size of this concurrent queue can be limited to prevent the background
/// threads from producing too many slices (i.e. especially in offline mode) and consuming too much memory.
class SyncedCameraStreamsSlicer {
public:
    using SliceCondition = EventBufferReslicerAlgorithm::Condition;
    using SliceIterator  = SliceIteratorT<SyncedSlice>;

    /// @brief Constructor
    /// @param master_camera Master camera instance
    /// @param slave_cameras Slave camera instances
    /// @param slice_condition Slicing parameters
    /// @param max_queue_size Maximum number of slices that can be stored in the internal queue
    /// @throw std::invalid_argument if no slave camera is provided
    SyncedCameraStreamsSlicer(Camera &&master_camera, std::vector<Camera> &&slave_cameras,
                              const SliceCondition &slice_condition = SliceCondition::make_n_us(1000),
                              size_t max_queue_size                 = 5);

    /// @brief Default constructor
    SyncedCameraStreamsSlicer();

    /// @brief Move constructor
    /// @param slicer SyncedCameraStreamsSlicer to move
    SyncedCameraStreamsSlicer(SyncedCameraStreamsSlicer &&slicer) noexcept;

    /// @brief Move assignment operator
    /// @param slicer SyncedCameraStreamsSlicer to move
    SyncedCameraStreamsSlicer &operator=(SyncedCameraStreamsSlicer &&slicer) noexcept;

    SyncedCameraStreamsSlicer(const SyncedCameraStreamsSlicer &) = delete;
    SyncedCameraStreamsSlicer &operator=(const SyncedCameraStreamsSlicer &) = delete;

    /// @brief Destructor
    ~SyncedCameraStreamsSlicer();

    /// @brief Starts the camera streams and returns an iterator to the first slice
    SliceIterator begin();

    /// @brief Returns an iterator to the end of the streams
    SliceIterator end();

    /// @brief Returns the underlying master camera instance
    [[nodiscard]] const Camera &master() const;

    /// @brief Returns the number of slave cameras
    [[nodiscard]] size_t slaves_count() const;

    /// @brief Returns the underlying slave camera instance
    [[nodiscard]] const Camera &slave(size_t i) const;

private:
    class Master;

    std::shared_ptr<ConcurrentQueue<SyncedSlice>> queue_;
    std::unique_ptr<Master> master_source_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_SYNCED_CAMERA_STREAM_SLICER_H
