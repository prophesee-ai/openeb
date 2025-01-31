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

#ifndef METAVISION_SDK_CORE_ROI_MASK_ALGORITHM_H
#define METAVISION_SDK_CORE_ROI_MASK_ALGORITHM_H

#include <opencv2/core/core.hpp>
#include <memory>
#include <sstream>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/algorithms/detail/internal_algorithms.h"

namespace Metavision {

/// @brief Class that only propagates events which are contained in a certain region of interest.
///
/// The Region Of Interest (ROI) is defined by a mask (cv::Mat). An event is validated if
/// the mask at the event position stores a positive number.
///
/// Alternatively, the user can enable different rectangular regions defined by the upper left corner
/// and the bottom right corner that propagates any event inside them.
///
class RoiMaskAlgorithm {
public:
    /// @brief Builds a new RoiMaskAlgorithm object which propagates events in the given window
    /// @param pixel_mask Mask of pixels that should be retained (pixel <= 0 is filtered)
    inline explicit RoiMaskAlgorithm(const cv::Mat &pixel_mask);

    // Default destructor
    ~RoiMaskAlgorithm() = default;

    /// @brief Applies the ROI Mask filter to the given input buffer storing the result in the output buffer
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
        return Metavision::detail::insert_if(it_begin, it_end, inserter, *this);
    }

    /// @brief Enables a rectangular region defined by the upper left corner
    /// and the bottom right corner that propagates any event inside them.
    /// @param x0 X coordinate of the upper left corner
    /// @param y0 Y coordinate of the upper left corner
    /// @param x1 X coordinate of the lower right corner
    /// @param y1 Y coordinate of the lower right corner
    inline void enable_rectangle(int x0, int y0, int x1, int y1);

    /// @brief Returns the maximum number of pixels (height) of the mask
    /// @return Maximum height of the mask
    inline int max_height() const;

    /// @brief Returns the maximum number of pixels (width) of the mask
    /// @return Maximum width of the mask
    inline int max_width() const;

    /// @brief Returns the pixel mask of the filter
    /// @return cv::Mat containing the pixel mask of the filter
    inline const cv::Mat &pixel_mask() const;

    /// @brief Sets the pixel mask of the filter
    /// @param mask Pixel mask to be used while filtering
    inline void set_pixel_mask(const cv::Mat &mask);

    /// @brief Basic operator to check if an element is filtered
    /// @param ev Event to check
    template<typename T>
    inline bool operator()(const T &ev) const;

    /// @brief Basic operator to check if a position is filtered
    /// @param x X coordinate or the position to check
    /// @param y Y coordinate or the position to check
    inline bool operator()(int x, int y) const;

private:
    std::vector<cv::Rect> rectangles_{};
    cv::Mat pixel_mask_{};
};

inline RoiMaskAlgorithm::RoiMaskAlgorithm(const cv::Mat &pixel_mask) : pixel_mask_(pixel_mask) {}

inline void RoiMaskAlgorithm::enable_rectangle(int x0, int y0, int x1, int y1) {
    rectangles_.push_back(cv::Rect(cv::Point(x0, y0), cv::Size(x1 + 1 - x0, y1 + 1 - y0)));
}

inline int RoiMaskAlgorithm::max_height() const {
    return pixel_mask_.rows;
}

inline int RoiMaskAlgorithm::max_width() const {
    return pixel_mask_.cols;
}

inline const cv::Mat &RoiMaskAlgorithm::pixel_mask() const {
    return pixel_mask_;
}

inline void RoiMaskAlgorithm::set_pixel_mask(const cv::Mat &mask) {
    pixel_mask_ = mask;
}

template<typename T>
inline bool RoiMaskAlgorithm::operator()(const T &ev) const {
    return this->operator()(ev.x, ev.y);
}

inline bool RoiMaskAlgorithm::operator()(int x, int y) const {
    if (y > max_height() || x > max_width())
        return false;

    if (pixel_mask_.at<double>(y, x) > 0)
        return true;

    const auto found = std::find_if(std::cbegin(rectangles_), std::cend(rectangles_), [&](const cv::Rect &rectangle) {
        return rectangle.contains(cv::Point{x, y});
    });
    return (found != std::cend(rectangles_));
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_ROI_MASK_ALGORITHM_H
