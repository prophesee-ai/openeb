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

#include <sstream>
#include <opencv2/imgproc.hpp>
#include "metavision/sdk/core/algorithms/events_integration_algorithm.h"
#include "metavision/sdk/core/utils/fast_math_functions.h"

namespace Metavision {

EventsIntegrationAlgorithm::EventsIntegrationAlgorithm(unsigned int width, unsigned int height, timestamp decay_time,
                                                       float contrast_on, float contrast_off,
                                                       int tonemapping_max_ev_count, int gaussian_blur_kernel_radius,
                                                       float diffusion_weight) :
    width_(width),
    height_(height),
    gaussian_blur_kernel_radius_(gaussian_blur_kernel_radius),
    diffusion_weight_(std::min(0.25f, std::max(0.f, diffusion_weight))),
    decay_time_(decay_time),
    log_contrast_{std::log(contrast_off <= 0 ? 1 / contrast_on : contrast_off), std::log(contrast_on)},
    tonemapping_factor_(std::exp(-tonemapping_max_ev_count * std::log(contrast_on))),
    exp_decay_lut_{Math::init_exp_decay_lut(128)},
    states_(width * height) {}

void EventsIntegrationAlgorithm::generate(cv::Mat &grayscale_frame) {
    grayscale_frame.create(height_, width_, CV_8UC1);
    uchar *pframe = grayscale_frame.ptr();
    auto *pstates = states_.data();
    for (unsigned int y = 0; y < height_; ++y) {
        for (unsigned int x = 0; x < width_; ++x, ++pframe, ++pstates) {
            const float decay_factor =
                Math::fast_exp_decay(exp_decay_lut_, (last_t_ - pstates->last_t) / static_cast<float>(decay_time_));
            pstates->last_t = last_t_;
            pstates->logI   = pstates->logI * decay_factor;
            *pframe         = cv::saturate_cast<uchar>(255 * tonemapping_factor_ * std::exp(pstates->logI));
        }
    }
    if (diffusion_weight_ > 0) {
        diffuse_intensities();
    } else if (gaussian_blur_kernel_radius_ > 0) {
        cv::GaussianBlur(grayscale_frame, grayscale_frame,
                         cv::Size(2 * gaussian_blur_kernel_radius_ + 1, 2 * gaussian_blur_kernel_radius_ + 1), 0);
    }
}

void EventsIntegrationAlgorithm::reset() {
    PxState s_reset{};
    std::fill(states_.begin(), states_.end(), s_reset);
}

void EventsIntegrationAlgorithm::integrate_event(const EventCD &e) {
    if (e.x >= width_ || e.y >= height_) { // e.x<0 || e.y<0 cannot happen since they are unsigned
        return;
    }
    auto &s                  = states_[e.y * width_ + e.x];
    const float decay_factor = Math::fast_exp_decay(exp_decay_lut_, (e.t - s.last_t) / static_cast<float>(decay_time_));
    s.logI                   = s.logI * decay_factor + log_contrast_[e.p];
    s.last_t                 = e.t;
}

void EventsIntegrationAlgorithm::diffuse_intensities() {
    const float stable_weight    = std::max(0.f, 1.f - 4 * diffusion_weight_);
    const float diffusion_weight = std::min(0.25f, diffusion_weight_);
    const int stride             = static_cast<int>(width_);
    auto pstates                 = states_.data();
    for (unsigned int y = 0; y < height_; ++y) {
        for (unsigned int x = 0; x < width_; ++x, ++pstates) {
            if (x == 0 || x == width_ - 1 || y == 0 || y == height_ - 1) {
                continue;
            }
            float sum_logI = stable_weight * pstates->logI;
            sum_logI += diffusion_weight * pstates[-1].logI;
            sum_logI += diffusion_weight * pstates[1].logI;
            sum_logI += diffusion_weight * pstates[-stride].logI;
            sum_logI += diffusion_weight * pstates[stride].logI;
            pstates->logI = sum_logI;
        }
    }
}

} // namespace Metavision