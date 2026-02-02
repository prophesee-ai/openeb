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

#ifndef METAVISION_SDK_CORE_EVENTS_INTEGRATION_ALGORITHM_H
#define METAVISION_SDK_CORE_EVENTS_INTEGRATION_ALGORITHM_H

#include <opencv2/core/core.hpp>
#include "metavision/sdk/base/events/event_cd.h"

namespace Metavision {

/// @brief Class to integrate events into a grayscale frame
class EventsIntegrationAlgorithm {
public:
    /// @brief Constructor
    /// @param width Width of the input event stream.
    /// @param height Height of the input event stream.
    /// @param decay_time Time constant for the exponential decay of the integrated grayscale values.
    /// @param contrast_on Contrast value for ON events.
    /// @param contrast_off Contrast value for OFF events. If non-positive, the contrast is set to the inverse of
    /// the @p contrast_on value.
    /// @param tonemapping_max_ev_count Maximum number of events to consider for tonemapping to 8-bits range.
    /// @param gaussian_blur_kernel_radius Radius of the Gaussian blur kernel. If non-positive, no blur is applied.
    /// @param diffusion_weight Weight for slowly diffusing 4-neighboring intensities into the central ones, to smooth
    /// reconstructed intensities in the case of static camera. Clamped to [0; 0.25], 0 meaning no diffusion and 0.25
    /// meaning ignoring central intensity.
    EventsIntegrationAlgorithm(unsigned int width, unsigned int height, timestamp decay_time = 1'000'000,
                               float contrast_on = 1.2f, float contrast_off = -1, int tonemapping_max_ev_count = 5,
                               int gaussian_blur_kernel_radius = 1, float diffusion_weight = 0.f);

    /// @brief Processes a range of events
    /// @tparam InputIt Iterator type.
    /// @param it_begin Iterator pointing to the first event to process.
    /// @param it_end Iterator pointing to the end of the range of events to process.
    template<typename InputIt>
    void process_events(InputIt it_begin, InputIt it_end);

    /// @brief Generates the grayscale frame at the timestamp of the last received event.
    void generate(cv::Mat &grayscale_frame);

    /// @brief Resets the internal state
    void reset();

private:
    void integrate_event(const EventCD &e);
    void diffuse_intensities();

    const unsigned int width_;
    const unsigned int height_;
    const int gaussian_blur_kernel_radius_;
    const float diffusion_weight_;
    const timestamp decay_time_;
    const std::array<float, 2> log_contrast_;
    const float tonemapping_factor_;
    const std::vector<float> exp_decay_lut_;

    struct PxState {
        timestamp last_t = 0;
        float logI       = 0;
    };
    std::vector<PxState> states_;
    timestamp last_t_;
};

} // namespace Metavision

#include "metavision/sdk/core/algorithms/detail/events_integration_algorithm_impl.h"

#endif // METAVISION_SDK_CORE_EVENTS_INTEGRATION_ALGORITHM_H
