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

#ifndef METAVISION_SDK_CORE_EVENT_CUBE_PROCESSOR_H
#define METAVISION_SDK_CORE_EVENT_CUBE_PROCESSOR_H

#include "metavision/sdk/core/preprocessors/event_preprocessor.h"

namespace Metavision {

/// @brief Class used to compute an event cube from a stream of EventCD
/// @tparam InputIt The type of the input iterator for the range of events to process
template<typename InputIt>
class EventCubeProcessor : public EventPreprocessor<InputIt> {
public:
    EventCubeProcessor() = default;

    /// @brief Constructor
    /// @param delta_t Delta time used to accumulate events inside the frame
    /// @param event_input_width Width of the event stream
    /// @param event_input_height Height of the event stream
    /// @param num_utbins Number of micro temporal bins
    /// @param split_polarity Process positive and negative events into separate channels
    /// @param max_incr_per_pixel Maximum number of increments per pixel. This is used to normalize the contribution of
    /// each event
    /// @param clip_value_after_normalization Clipping value to apply after normalization (typically: 1.)
    /// @param width_scale Scale on the width previously applied to input events. This factor is considered to modulate
    /// the contribution of each event at its coordinates.
    /// @param height_scale Scale on the height previously applied to input events. This factor is considered to
    /// modulate the contribution of each event at its coordinates.
    EventCubeProcessor(timestamp delta_t, int event_input_width, int event_input_height, int num_utbins,
                       bool split_polarity, float max_incr_per_pixel, float clip_value_after_normalization = 0.f,
                       float width_scale = 1.f, float height_scale = 1.f);

private:
    inline void set_value(float *buff, const std::size_t buff_size, const int bin, const int p, const int x,
                          const int y, const float val) const;

    void compute(const timestamp cur_frame_start_ts, InputIt begin, InputIt end, Tensor &tensor) const override;

    const float normalization_factor_;
    const bool split_polarity_;
    const int num_utbins_;
    const float clip_value_after_normalization_;
    const int num_polarities_;
    const float num_utbins_over_delta_t_;
    const int w_h_;   // network_input_width * network_input_height
    const int w_h_p_; // network_input_width * network_input_height * num_polarities
    const int width_;
};

} // namespace Metavision

#include <metavision/sdk/core/preprocessors/detail/event_cube_processor_impl.h>

#endif // METAVISION_SDK_CORE_EVENT_CUBE_PROCESSOR_H
