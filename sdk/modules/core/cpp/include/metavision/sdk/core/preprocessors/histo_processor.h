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

#ifndef METAVISION_SDK_CORE_HISTO_PROCESSOR_H
#define METAVISION_SDK_CORE_HISTO_PROCESSOR_H

#include "metavision/sdk/core/preprocessors/event_preprocessor.h"

namespace Metavision {

/// @brief Class used to compute a histogram from a stream of EventCD
/// @tparam InputIt The type of the input iterator for the range of events to process
template<typename InputIt>
class HistoProcessor : public EventPreprocessor<InputIt> {
public:
    HistoProcessor() = default;

    /// @brief Constructor
    /// @param event_input_width Maximum width of input events
    /// @param event_input_height Maximum height of input events
    /// @param max_incr_per_pixel Maximum number of increments per pixel. This is used to normalize the contribution of
    /// each event
    /// @param clip_value_after_normalization Clipping value to apply after normalization (typically: 1.)
    /// @param use_CHW Boolean to define frame dimension order,
    ///     True if the fields' frame order is (Channel, Height, Width)
    /// @param width_scale Scale on the width previously applied to input events. This factor is considered to modulate
    /// the contribution of each event at its coordinates.
    /// @param height_scale Scale on the height previously applied to input events. This factor is considered to
    /// modulate the contribution of each event at its coordinates.
    HistoProcessor(int event_input_width, int event_input_height, float max_incr_per_pixel,
                   float clip_value_after_normalization, bool use_CHW = true, float width_scale = 1.f,
                   float height_scale = 1.f);

private:
    void compute(const timestamp cur_frame_start_ts, InputIt begin, InputIt end, Tensor &tensor) const override;
    bool is_CHW(const Tensor &t) const;

    float increment_;
    const float clip_value_after_normalization_;
    const int width_, height_, channels_;
};

} // namespace Metavision

#include <metavision/sdk/core/preprocessors/detail/histo_processor_impl.h>

#endif //  METAVISION_SDK_CORE_HISTO_PROCESSOR_H
