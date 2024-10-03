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

#ifndef METAVISION_SDK_CORE_ROTATE_EVENTS_ALGORITHM_H
#define METAVISION_SDK_CORE_ROTATE_EVENTS_ALGORITHM_H

#include <memory>
#include <cmath>
#include <sstream>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/base/events/event2d.h"
#include "metavision/sdk/core/utils/detail/iterator_traits.h"

namespace Metavision {

/// @brief class that allows to rotate an event stream.
/// @note We assume the rotation to happen with respect to the center of the image
class RotateEventsAlgorithm {
public:
    /// @brief Builds a new RotateEventsAlgorithm object with the given width and height
    /// @param width_minus_one Maximum X coordinate of the events (width-1)
    /// @param height_minus_one Maximum Y coordinate of the events (height-1)
    /// @param rotation Value in radians used for the rotation
    inline explicit RotateEventsAlgorithm(std::int16_t width_minus_one, std::int16_t height_minus_one, float rotation);

    /// @brief Default destructor
    ~RotateEventsAlgorithm() = default;

    /// @brief Applies the rotate event filter to the given input buffer storing the result in the output buffer.
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @tparam OutputIt Read-Write output event iterator type. Works for iterators over containers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param inserter Output iterator or back inserter
    /// @return Iterator pointing to the past-the-end event added in the output
    template<class InputIt, class OutputIt>
    inline OutputIt process_events(InputIt it_begin, InputIt it_end, OutputIt inserter) {
        using output_type = typename Metavision::iterator_traits<OutputIt>::value_type;

        for (auto it = it_begin; it != it_end; ++it) {
            output_type ev(*it);
            auto new_x = std::round(precomputed_cos_ * (ev.x - rotation_center_x_) -
                                    precomputed_sin_ * (ev.y - rotation_center_y_) + rotation_center_x_);
            auto new_y = std::round(precomputed_sin_ * (ev.x - rotation_center_x_) +
                                    precomputed_cos_ * (ev.y - rotation_center_y_) + rotation_center_y_);

            if (new_x >= 0 && new_x <= width_minus_one_ && new_y >= 0 && new_y <= height_minus_one_) {
                ev.x      = new_x;
                ev.y      = new_y;
                *inserter = ev;
                ++inserter;
            }
        }
        return inserter;
    }

    /// @brief Returns the maximum X coordinate of the events
    /// @return Maximum X coordinate of the events
    inline std::int16_t width_minus_one() const;

    /// @brief Sets the maximum X coordinate of the events
    /// @param width_minus_one Maximum X coordinate of the events
    inline void set_width_minus_one(std::int16_t width_minus_one);

    /// @brief Returns the maximum Y coordinate of the events
    /// @return Maximum Y coordinate of the events
    inline std::int16_t height_minus_one() const;

    /// @brief Sets the maximum Y coordinate of the events
    /// @param height_minus_one Maximum Y coordinate of the events
    inline void set_height_minus_one(std::int16_t height_minus_one);

    /// @brief Sets the new rotation angle
    /// @param new_angle New angle in rad
    inline void set_rotation(const float new_angle);

private:
    std::int16_t width_minus_one_{0};
    std::int16_t height_minus_one_{0};
    std::int16_t rotation_center_x_{0};
    std::int16_t rotation_center_y_{0};
    float precomputed_cos_{0};
    float precomputed_sin_{0};
};

inline RotateEventsAlgorithm::RotateEventsAlgorithm(std::int16_t width_minus_one, std::int16_t height_minus_one,
                                                    float rotation) :
    width_minus_one_(width_minus_one),
    height_minus_one_(height_minus_one),
    rotation_center_x_((width_minus_one + 1) / 2),
    rotation_center_y_((height_minus_one + 1) / 2) {
    precomputed_cos_ = std::cos(rotation);
    precomputed_sin_ = std::sin(rotation);
}

inline std::int16_t RotateEventsAlgorithm::width_minus_one() const {
    return width_minus_one_;
}

inline void RotateEventsAlgorithm::set_width_minus_one(std::int16_t width_minus_one) {
    width_minus_one_ = width_minus_one;
}

inline std::int16_t RotateEventsAlgorithm::height_minus_one() const {
    return height_minus_one_;
}

inline void RotateEventsAlgorithm::set_height_minus_one(std::int16_t height_minus_one) {
    height_minus_one_ = height_minus_one;
}

inline void RotateEventsAlgorithm::set_rotation(const float new_angle) {
    precomputed_cos_ = cosf(new_angle);
    precomputed_sin_ = sinf(new_angle);
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_ROTATE_EVENTS_ALGORITHM_H
