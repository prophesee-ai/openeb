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
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>

#include "metavision/sdk/core/utils/fast_math_functions.h"
#include "metavision/sdk/core/algorithms/time_decay_frame_generation_algorithm.h"

namespace Metavision {

namespace detail {

void init_colormap(Metavision::ColorPalette palette, std::vector<cv::Vec3b> &colormap) {
    const RGBColor rgb_bkg       = get_color(palette, ColorType::Background);
    const RGBColor rgb_pos       = get_color(palette, ColorType::Positive);
    const RGBColor rgb_neg       = get_color(palette, ColorType::Negative);
    auto rgb_interp_to_vec3b_fct = [](const RGBColor &rgb0, const RGBColor &rgb1, double f) {
        cv::Vec3f s0_rgb(rgb0.r, rgb0.g, rgb0.b), s1_rgb(rgb1.r, rgb1.g, rgb1.b);
        cv::Vec3f s0_lab, s1_lab;
        cv::Mat m0_rgb(1, 1, CV_32FC3, &s0_rgb), m1_rgb(1, 1, CV_32FC3, &s1_rgb);
        cv::Mat m0_lab(1, 1, CV_32FC3, &s0_lab), m1_lab(1, 1, CV_32FC3, &s1_lab);
        cv::cvtColor(m0_rgb, m0_lab, cv::COLOR_RGB2Lab);
        cv::cvtColor(m1_rgb, m1_lab, cv::COLOR_RGB2Lab);
        cv::Vec3f si_lab(s0_lab(0) * (1 - f) + s1_lab(0) * f, s0_lab(1) * (1 - f) + s1_lab(1) * f,
                         s0_lab(2) * (1 - f) + s1_lab(2) * f);
        cv::Vec3f si_rgb;
        cv::Mat mi_lab(1, 1, CV_32FC3, &si_lab), mi_rgb(1, 1, CV_32FC3, &si_rgb);
        cv::cvtColor(mi_lab, mi_rgb, cv::COLOR_Lab2RGB);
        return cv::Vec3b(cvRound(255 * si_rgb(2)), cvRound(255 * si_rgb(1)), cvRound(255 * si_rgb(0)));
    };
    const size_t kMidBin = 128, kTotalBins = 2 * kMidBin + 1;
    colormap.resize(kTotalBins);
    for (std::size_t i = 0; i < kTotalBins; ++i) {
        if (i < kMidBin) {
            colormap[i] = rgb_interp_to_vec3b_fct(rgb_neg, rgb_bkg, i / static_cast<double>(kMidBin - 1));
        } else if (i == kMidBin) {
            colormap[i] = cv::Vec3b(cvRound(255 * rgb_bkg.b), cvRound(255 * rgb_bkg.g), cvRound(255 * rgb_bkg.r));
        } else {
            colormap[i] =
                rgb_interp_to_vec3b_fct(rgb_bkg, rgb_pos, (i - kMidBin - 1) / static_cast<double>(kMidBin - 1));
        }
    }
}

cv::Vec3b apply_colormap(const std::vector<cv::Vec3b> &colormap, float v) {
    assert(-1.f <= v && v <= 1.f);
    const int idx = cvRound(0.5f * (1 + v) * (colormap.size() - 1));
    return colormap[idx];
}

uchar apply_colormap_grayscale(float v) {
    assert(-1.f <= v && v <= 1.f);
    return cv::saturate_cast<uchar>(0.5f * (1 + v) * 255);
}

} // namespace detail

TimeDecayFrameGenerationAlgorithm::TimeDecayFrameGenerationAlgorithm(int width, int height,
                                                                     timestamp exponential_decay_time_us,
                                                                     Metavision::ColorPalette palette) :
    exp_decay_lut_(Math::init_exp_decay_lut(32)), time_surface_(height, width, 2) {
    set_color_palette(palette);
    set_exponential_decay_time_us(exponential_decay_time_us);
    reset();
}

void TimeDecayFrameGenerationAlgorithm::generate(cv::Mat &frame, bool allocate) {
    if (allocate) {
        if (colored_) {
            frame.create(time_surface_.rows(), time_surface_.cols(), CV_8UC3);
        } else {
            frame.create(time_surface_.rows(), time_surface_.cols(), CV_8UC1);
        }
    }
    if (frame.rows != time_surface_.rows() || frame.cols != time_surface_.cols()) {
        std::ostringstream ss;
        ss << "Incompatible matrix size, must be (" << time_surface_.rows() << ", " << time_surface_.cols() << ").";
        throw std::invalid_argument(ss.str());
    }
    if (frame.depth() != CV_8U || frame.channels() != (colored_ ? 3 : 1)) {
        std::ostringstream ss;
        ss << "Incompatible matrix type, must be CV_8UC" << (colored_ ? 3 : 1) << ".";
        throw std::invalid_argument(ss.str());
    }

    if (colored_) {
        for (int y = 0; y < time_surface_.rows(); ++y) {
            for (int x = 0; x < time_surface_.cols(); ++x) {
                const auto dt_n = last_ts_ - time_surface_.at(y, x, 0), dt_p = last_ts_ - time_surface_.at(y, x, 1),
                           dt          = std::min(dt_n, dt_p);
                const bool is_positive = (dt_n > dt_p);
                const float f =
                    (is_positive ? 1 : -1) *
                    Math::fast_exp_decay(exp_decay_lut_, dt / static_cast<float>(exponential_decay_time_us_));
                frame.at<cv::Vec3b>(y, x) = detail::apply_colormap(colormap_, f);
            }
        }
    } else {
        for (int y = 0; y < time_surface_.rows(); ++y) {
            for (int x = 0; x < time_surface_.cols(); ++x) {
                const auto dt_n = last_ts_ - time_surface_.at(y, x, 0), dt_p = last_ts_ - time_surface_.at(y, x, 1),
                           dt          = std::min(dt_n, dt_p);
                const bool is_positive = (dt_n > dt_p);
                const float f =
                    (is_positive ? 1 : -1) *
                    Math::fast_exp_decay(exp_decay_lut_, dt / static_cast<float>(exponential_decay_time_us_));
                frame.at<uchar>(y, x) = detail::apply_colormap_grayscale(f);
            }
        }
    }
}

void TimeDecayFrameGenerationAlgorithm::set_exponential_decay_time_us(timestamp exponential_decay_time_us) {
    if (exponential_decay_time_us <= 0)
        throw std::invalid_argument("exponential decay time must be strictly positive.");

    exponential_decay_time_us_ = exponential_decay_time_us;
}

timestamp TimeDecayFrameGenerationAlgorithm::get_exponential_decay_time_us() const {
    return exponential_decay_time_us_;
}

void TimeDecayFrameGenerationAlgorithm::set_color_palette(Metavision::ColorPalette palette) {
    colored_ = (palette != Metavision::ColorPalette::Gray);
    if (colored_) {
        detail::init_colormap(palette, colormap_);
    } else {
        colormap_.clear();
    }
}

void TimeDecayFrameGenerationAlgorithm::reset() {
    time_surface_.set_to(0);
    last_ts_ = 0;
}

} // namespace Metavision
