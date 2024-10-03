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
    static const cv::Vec3b bg_color_default = get_bgr_color(default_palette(), Metavision::ColorType::Background);
    return bg_color_default;
}

const cv::Vec3b &BaseFrameGenerationAlgorithm::on_color_default() {
    static const cv::Vec3b on_color_default = get_bgr_color(default_palette(), Metavision::ColorType::Positive);
    return on_color_default;
};

const cv::Vec3b &BaseFrameGenerationAlgorithm::off_color_default() {
    static const cv::Vec3b off_color_default = get_bgr_color(default_palette(), Metavision::ColorType::Negative);
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
    bg_color_[3]         = 255;
    off_on_colors_[0][3] = 255;
    off_on_colors_[1][3] = 255;
    flags_               = (colored ? Parameters::BGR : Parameters::GRAY);
}

void BaseFrameGenerationAlgorithm::set_color_palette(const Metavision::ColorPalette &palette) {
    bg_color_         = get_bgra_color(palette, Metavision::ColorType::Background);
    off_on_colors_[0] = get_bgra_color(get_color(palette, Metavision::ColorType::Negative));
    off_on_colors_[1] = get_bgra_color(palette, Metavision::ColorType::Positive);
    flags_            = (palette != Metavision::ColorPalette::Gray ? Parameters::BGR : Parameters::GRAY);
}

void BaseFrameGenerationAlgorithm::set_parameters(const cv::Vec4b &bg_color, const cv::Vec4b &on_color,
                                                  const cv::Vec4b &off_color, int flags) {
    for (int i = 0; i < 4; ++i) {
        bg_color_[i]         = bg_color[i];
        off_on_colors_[1][i] = on_color[i];
        off_on_colors_[0][i] = off_color[i];
    }
    flags_ = flags;
}

void BaseFrameGenerationAlgorithm::set_parameters(const Metavision::ColorPalette &palette, int flags) {
    bg_color_         = get_bgra_color(palette, Metavision::ColorType::Background);
    off_on_colors_[0] = get_bgra_color(palette, Metavision::ColorType::Negative);
    off_on_colors_[1] = get_bgra_color(palette, Metavision::ColorType::Positive);
    flags_            = flags;
}

void BaseFrameGenerationAlgorithm::get_dimension(uint32_t &height, uint32_t &width, uint32_t &channels) const {
    height = height_;
    width  = width_;
    if (flags_ & Parameters::GRAY) {
        channels = 1;
    } else if (flags_ & Parameters::RGB || flags_ & Parameters::BGR) {
        channels = 3;
    } else {
        channels = 4;
    }
}

} // namespace Metavision