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

#ifndef METAVISION_SDK_CORE_ROI_FILTER_ALGORITHM_H
#define METAVISION_SDK_CORE_ROI_FILTER_ALGORITHM_H

#include <memory>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/algorithms/detail/internal_algorithms.h"

namespace Metavision {

/// @brief Class that only propagates events which are contained in a certain window
/// of interest defined by the coordinates of the upper left corner and the
/// lower right corner
class RoiFilterAlgorithm {
public:
    /// @brief Builds a new RoiFilterAlgorithm object which propagates events in the given window
    /// @param x0 X coordinate of the upper left corner of the ROI window
    /// @param y0 Y coordinate of the upper left corner of the ROI window
    /// @param x1 X coordinate of the lower right corner of the ROI window
    /// @param y1 Y coordinate of the lower right corner of the ROI window
    /// @param output_relative_coordinates If false, events that passed the ROI filter are expressed in the whole
    ///                                              image coordinates.
    ///                                    If true, they are expressed in the ROI coordinates system
    inline RoiFilterAlgorithm(std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1,
                              bool output_relative_coordinates = false);

    // @briefDefault destructor
    ~RoiFilterAlgorithm() = default;

    /// @brief Applies the ROI Mask filter to the given input buffer storing the result in the output buffer.
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @tparam OutputIt Read-Write output event iterator type. Works for iterators over containers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param inserter Output iterator or back inserter
    /// @return Iterator pointing to the past-the-end event added in the output
    template<class InputIt, class OutputIt>
    inline OutputIt process_events(InputIt it_begin, InputIt it_end, OutputIt inserter);

    /// @brief Returns true if the algorithm returns events expressed in coordinates relative to the ROI
    /// @return true if the algorithm is resetting the filtered events
    inline bool is_resetting() const;

    /// @brief Returns the x coordinate of the upper left corner of the ROI window
    /// @return X coordinate of the upper left corner
    inline std::int32_t x0() const;

    /// @brief Returns the y coordinate of the upper left corner of the ROI window
    /// @return Y coordinate of the upper left corner
    inline std::int32_t y0() const;

    /// @brief Returns the x coordinate of the lower right corner of the ROI window
    /// @return X coordinate of the lower right corner
    inline std::int32_t x1() const;

    /// @brief Returns the y coordinate of the lower right corner of the ROI window
    /// @return Y coordinate of the lower right corner
    inline std::int32_t y1() const;

    /// @brief Sets the x coordinate of the upper left corner of the ROW window
    /// @param x0 X coordinate of the upper left corner
    inline void set_x0(std::int32_t x0);

    /// @brief Sets the y coordinate of the upper left corner of the ROW window
    /// @param y0 Y coordinate of the upper left corner
    inline void set_y0(std::int32_t y0);

    /// @brief Sets the x coordinate of the lower right corner of the ROW window
    /// @param x1 X coordinate of the lower right corner
    inline void set_x1(std::int32_t x1);

    /// @brief Sets the x coordinate of the lower right corner of the ROW window
    /// @param y1 Y coordinate of the lower right corner
    inline void set_y1(std::int32_t y1);

    /// @brief Operator applied when output_relative_coordinates is true, and the event is accepted
    /// @param ev Event to be updated
    template<typename T>
    inline bool operator()(const T &ev) const;

    /// @brief Operator applied when output_relative_coordinates == true, and the event is accepted
    /// @param ev Event to be updated
    template<typename T>
    inline void operator()(T &ev) const;

private:
    std::int32_t x0_{0};
    std::int32_t y0_{0};
    std::int32_t x1_{0};
    std::int32_t y1_{0};
    bool output_relative_coordinates_{false};
};

inline RoiFilterAlgorithm::RoiFilterAlgorithm(std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1,
                                              bool output_relative_coordinates) :
    x0_(x0), y0_(y0), x1_(x1), y1_(y1), output_relative_coordinates_(output_relative_coordinates) {}

template<class InputIt, class OutputIt>
inline OutputIt RoiFilterAlgorithm::process_events(InputIt it_begin, InputIt it_end, OutputIt inserter) {
    if (is_resetting()) {
        return Metavision::detail::transform_if(
            it_begin, it_end, inserter, [&](const auto &event) { return this->operator()(event); }, std::cref(*this));
    } else {
        return Metavision::detail::insert_if(it_begin, it_end, inserter,
                                             [&](const auto &event) { return this->operator()(event); });
    }
}

inline bool RoiFilterAlgorithm::is_resetting() const {
    return output_relative_coordinates_;
}

inline int32_t RoiFilterAlgorithm::x0() const {
    return x0_;
}

inline int32_t RoiFilterAlgorithm::y0() const {
    return y0_;
}

inline int32_t RoiFilterAlgorithm::x1() const {
    return x1_;
}

inline int32_t RoiFilterAlgorithm::y1() const {
    return y1_;
}

inline void RoiFilterAlgorithm::set_x0(int32_t x0) {
    x0_ = x0;
}

inline void RoiFilterAlgorithm::set_y0(int32_t y0) {
    y0_ = y0;
}

inline void RoiFilterAlgorithm::set_x1(int32_t x1) {
    x1_ = x1;
}

inline void RoiFilterAlgorithm::set_y1(int32_t y1) {
    y1_ = y1;
}

template<typename T>
inline bool RoiFilterAlgorithm::operator()(const T &ev) const {
    return (ev.x >= x0_ && ev.x <= x1_) && (ev.y >= y0_ && ev.y <= y1_);
}

template<typename T>
inline void RoiFilterAlgorithm::operator()(T &ev) const {
    ev.x -= x0_;
    ev.y -= y0_;
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_ROI_FILTER_ALGORITHM_H
