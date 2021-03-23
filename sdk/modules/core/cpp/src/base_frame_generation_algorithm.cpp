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

#include <assert.h>
#include <stdexcept>
#include "metavision/sdk/core/algorithms/base_frame_generation_algorithm.h"

namespace Metavision {

const cv::Vec3b &BaseFrameGenerationAlgorithm::bg_color_default() {
    static const cv::Vec3b bg_color_default = get_cv_color(default_palette(), Metavision::ColorType::Background);
    return bg_color_default;
}

const cv::Vec3b &BaseFrameGenerationAlgorithm::on_color_default() {
    static const cv::Vec3b on_color_default = get_cv_color(default_palette(), Metavision::ColorType::Positive);
    return on_color_default;
};

const cv::Vec3b &BaseFrameGenerationAlgorithm::off_color_default() {
    static const cv::Vec3b off_color_default = get_cv_color(default_palette(), Metavision::ColorType::Negative);
    return off_color_default;
};

BaseFrameGenerationAlgorithm::BaseFrameGenerationAlgorithm(int sensor_width, int sensor_height,
                                                           const Metavision::ColorPalette &palette) :
    width_(sensor_width), height_(sensor_height) {
    set_color_palette(palette);
}

void BaseFrameGenerationAlgorithm::set_colors(const cv::Scalar &bg_color, const cv::Scalar &on_color,
                                              const cv::Scalar &off_color, bool colored) {
    for (int i = 0; i < 3; ++i) {
        bg_color_[i]         = static_cast<uchar>(bg_color[i]);
        off_on_colors_[1][i] = static_cast<uchar>(on_color[i]);
        off_on_colors_[0][i] = static_cast<uchar>(off_color[i]);
    }
    colored_ = colored;
}

void BaseFrameGenerationAlgorithm::set_color_palette(const Metavision::ColorPalette &palette) {
    bg_color_         = get_cv_color(palette, Metavision::ColorType::Background);
    off_on_colors_[0] = get_cv_color(palette, Metavision::ColorType::Negative);
    off_on_colors_[1] = get_cv_color(palette, Metavision::ColorType::Positive);
    colored_          = palette != Metavision::ColorPalette::Gray;
}

cv::Vec3b BaseFrameGenerationAlgorithm::get_cv_color(const Metavision::ColorPalette &palette,
                                                     const Metavision::ColorType &type) {
    const Metavision::RGBColor c = Metavision::getColor(palette, type);
    return cv::Vec3b(static_cast<uchar>(c.b * 255 + 0.5), static_cast<uchar>(c.g * 255 + 0.5),
                     static_cast<uchar>(c.r * 255 + 0.5));
}

void BaseFrameGenerationAlgorithm::get_dimension(uint32_t &height, uint32_t &width, uint32_t &channels) const {
    height   = height_;
    width    = width_;
    channels = (colored_ ? 3 : 1);
}

} // namespace Metavision