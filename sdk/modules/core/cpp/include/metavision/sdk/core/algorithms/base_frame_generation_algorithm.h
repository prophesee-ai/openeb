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

#ifndef METAVISION_SDK_CORE_BASE_FRAME_GENERATION_ALGORITHM_H
#define METAVISION_SDK_CORE_BASE_FRAME_GENERATION_ALGORITHM_H

#include <array>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <opencv2/core/mat.hpp>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/core/utils/colors.h"

namespace Metavision {

/// @brief Base class whose purpose is to generate a frame from accumulated events when the user makes the request
///
/// Both @ref OnDemandFrameGenerationAlgorithm and @ref PeriodicFrameGenerationAlgorithm inherit from this class.
///
/// It features basic logic to allow child class to generate frames with various constraints such as using over
/// accumulation, regenerating a frame with different colors or accumulation time
///
/// To generate a frame from events:
/// - Each event has an x, y, p and t information (@ref EventCD)
/// - Each pixel of the generated frame can be updated with 3 colors: a background color, a positive contrast detection
/// event color, a negative contrast detection color
/// - An event with coordinate {x, y} updates the corresponding pixel in the frame with one of the above color depending
/// on its polarity p (0 - negative contrast detection - or 1 - positive contrast detection)
/// - A pixel in the generated frame will have a background color if no event occurred at this position
/// - One can accumulate a specific time interval of events in the frame. This time interval is called accumulation
/// time.
/// - An accumulation time of dt microseconds means that only the events that occurred in the last dt microseconds are
/// used to generate the frame.
///
/// The static method @ref generate_frame_from_events allows generating a frame from an event-buffer, an accumulation
/// time and a color palette
class BaseFrameGenerationAlgorithm {
public:
    /// @brief Returns the default palette used by the frame generation
    static constexpr Metavision::ColorPalette default_palette() {
        return Metavision::ColorPalette::Dark;
    }

    /// @brief Returns default Prophesee dark palette background color
    static const cv::Vec3b &bg_color_default();

    /// @brief Returns default Prophesee dark palette positive event color
    static const cv::Vec3b &on_color_default();

    /// @brief Returns default Prophesee dark palette negative event color
    static const cv::Vec3b &off_color_default();

    /// @brief Destructor
    virtual ~BaseFrameGenerationAlgorithm() = default;

    /// @brief Stand-alone (static) method to generate a frame from events
    ///
    /// All events in the interval ]t - dt, t] are used where t the timestamp of the last event in the buffer, and dt
    /// the input @p accumulation_time_us. If @p accumulation_time_us is kept to 0, all input events are used.
    ///
    /// @warning The input @p frame must be allocated beforehand
    /// @tparam EventIt Input event iterator type. Works for iterators over containers of @ref EventCD or equivalent
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param frame Pre-allocated frame that will be filled with CD events. It must have the same geometry as the input
    /// event source, and the color corresponding to the given @p palette (3 channels by default)
    /// @param accumulation_time_us Time range of events to update the frame with (in us)
    /// @param palette The Prophesee's color palette to use
    /// @note Even if there's no events, a frame filled with the background color will be generated
    /// @throw invalid_argument if @p frame does not have the expected type (CV_8U or CV_8UC3)
    template<typename EventIt>
    static void generate_frame_from_events(EventIt it_begin, EventIt it_end, cv::Mat &frame,
                                           const uint32_t accumulation_time_us     = 0,
                                           const Metavision::ColorPalette &palette = default_palette());

    /// @brief Sets the color used to generate the frame
    /// @param bg_color Color used as background, when no events were received for a pixel
    /// @param on_color Color used for on events
    /// @param off_color Color used for off events
    /// @param colored  If the generated frame should be grayscale (single channel) or in color (three channels)
    void set_colors(const cv::Scalar &bg_color, const cv::Scalar &on_color, const cv::Scalar &off_color, bool colored);

    /// @brief Sets the color used to generate the frame
    /// @param palette The Prophesee's color palette to use
    void set_color_palette(const Metavision::ColorPalette &palette);

    enum Parameters {
        GRAY   = (1 << 0),
        RGB    = (1 << 1),
        BGR    = (1 << 2),
        RGBA   = (1 << 3),
        BGRA   = (1 << 4),
        FLIP_Y = (1 << 10)
    };

    /// @brief Sets the parameters used to generate the frame
    /// @param bg_color  Color used as background, when no events were received for a pixel
    /// @param on_color  Color used for on events
    /// @param off_color Color used for off events
    /// @param flags     A combination of Parameters
    void set_parameters(const cv::Vec4b &bg_color, const cv::Vec4b &on_color, const cv::Vec4b &off_color, int flags);

    /// @brief Sets the parameters used to generate the frame
    /// @param palette The Prophesee's color palette to use
    /// @param flags     A combination of Parameters
    void set_parameters(const Metavision::ColorPalette &palette, int flags);

    /// @brief Gets the frame's dimension
    /// @param height Frame's height
    /// @param width Frame's width
    /// @param channels Frames's number of channels. 3 if the image is colored, 1 otherwise
    void get_dimension(uint32_t &height, uint32_t &width, uint32_t &channels) const;

protected:
    /// @brief Builds the algorithm
    /// @param sensor_width Sensor's width (in pixels)
    /// @param sensor_height Sensor's height (in pixels)
    /// @param palette The Prophesee's color palette to use
    BaseFrameGenerationAlgorithm(int sensor_width, int sensor_height, const Metavision::ColorPalette &palette);

    /// @brief Stand-alone (static) helper method to generate a frame from an input event buffer
    /// @warning The input @p frame must be allocated beforehand
    /// @note This method is used internally both by its public counterpart and the child classes
    /// @tparam EventIt Input iterator event type. Works for @ref EventCD or equivalent
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param frame Pre-allocated frame that will be filled with CD events. It must have the same geometry as the input
    /// event source, and the color corresponding to @p colored (3 channels by default)
    /// @param bg_color Background color
    /// @param off_on_colors Colors of negative and positive events
    /// @param flags A combination of Parameters
    /// @throw invalid_argument if @p frame does not have the expected type (CV_8U or CV_8UC3)
    template<typename EventIt>
    static void generate_frame_from_events(EventIt it_begin, EventIt it_end, cv::Mat &frame, const cv::Vec4b &bg_color,
                                           const std::array<cv::Vec4b, 2> &off_on_colors, int flags);

    // Frame properties
    const int width_, height_;               ///< Sensor's geometry
    int flags_;                              ///< Frame's generation parameters
    cv::Vec4b bg_color_;                     ///< The background color
    std::array<cv::Vec4b, 2> off_on_colors_; ///< The off and on color
};

} // namespace Metavision

#include "metavision/sdk/core/algorithms/detail/base_frame_generation_algorithm_impl.h"

#endif // METAVISION_SDK_CORE_BASE_FRAME_GENERATION_ALGORITHM_H