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

#include "metavision/sdk/core/algorithms/contrast_map_generation_algorithm.h"

namespace Metavision {

ContrastMapGenerationAlgorithm::ContrastMapGenerationAlgorithm(unsigned int width, unsigned int height,
                                                               float contrast_on, float contrast_off) :
    width_(width),
    height_(height),
    contrasts_{(contrast_off <= 0 ? (1 / contrast_on) : contrast_off), contrast_on},
    states_(cv::Mat_<float>::ones(height, width)) {}

void ContrastMapGenerationAlgorithm::generate(cv::Mat_<float> &contrast_map) {
    std::swap(states_, contrast_map);
    states_.create(height_, width_);
    states_.setTo(1.f);
}

void ContrastMapGenerationAlgorithm::generate(cv::Mat_<uchar> &contrast_map_tonnemapped, float tonemapping_factor,
                                              float tonemapping_bias) {
    states_.convertTo(contrast_map_tonnemapped, CV_8U, tonemapping_factor, tonemapping_bias);
    states_.setTo(1.f);
}

void ContrastMapGenerationAlgorithm::reset() {
    states_.setTo(1.f);
}

void ContrastMapGenerationAlgorithm::process_event(const EventCD &e) {
    states_.at<float>(e.y, e.x) *= contrasts_[e.p];
}

} // namespace Metavision