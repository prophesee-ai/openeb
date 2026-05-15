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

#ifndef METAVISION_SDK_CORE_CONTRAST_MAP_GENERATION_ALGORITHM_H
#define METAVISION_SDK_CORE_CONTRAST_MAP_GENERATION_ALGORITHM_H

#include <opencv2/core/core.hpp>
#include "metavision/sdk/base/events/event_cd.h"

namespace Metavision {

/// @brief Class to generate a contrast map from a stream of events
class ContrastMapGenerationAlgorithm {
public:
    /// @brief Constructor
    /// @param width Width of the input event stream.
    /// @param height Height of the input event stream.
    /// @param contrast_on Contrast value for ON events.
    /// @param contrast_off Contrast value for OFF events. If non-positive, the contrast is set to the inverse of
    /// the @p contrast_on value.
    ContrastMapGenerationAlgorithm(unsigned int width, unsigned int height, float contrast_on = 1.2f,
                                   float contrast_off = -1);

    /// @brief Processes a range of events
    /// @tparam InputIt Iterator type.
    /// @param it_begin Iterator pointing to the first event to process.
    /// @param it_end Iterator pointing to the end of the range of events to process.
    template<typename InputIt>
    void process_events(InputIt it_begin, InputIt it_end);

    /// @brief Generates the contrast map and resets the internal state
    /// @param contrast_map Output contrast map, swapped with the one maintained internally.
    void generate(cv::Mat_<float> &contrast_map);

    /// @brief Generates the tonemapped contrast map and resets the internal state
    /// @param contrast_map_tonnemapped Output tonemapped contrast map.
    /// @param tonemapping_factor Tonemapping factor.
    /// @param tonemapping_bias Tonemapping bias.
    void generate(cv::Mat_<uchar> &contrast_map_tonnemapped, float tonemapping_factor, float tonemapping_bias);

    /// @brief Resets the internal state
    void reset();

private:
    void process_event(const EventCD &e);

    const unsigned int width_;
    const unsigned int height_;
    const std::array<float, 2> contrasts_;

    cv::Mat_<float> states_;
};

} // namespace Metavision

#include "metavision/sdk/core/algorithms/detail/contrast_map_generation_algorithm_impl.h"

#endif // METAVISION_SDK_CORE_CONTRAST_MAP_GENERATION_ALGORITHM_H
