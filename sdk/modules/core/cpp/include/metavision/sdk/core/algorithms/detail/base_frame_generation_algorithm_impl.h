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

#ifndef METAVISION_SDK_CORE_BASE_FRAME_GENERATION_ALGORITHM_IMPL_H
#define METAVISION_SDK_CORE_BASE_FRAME_GENERATION_ALGORITHM_IMPL_H

#include <opencv2/core/mat.hpp>

#include "metavision/sdk/core/utils/colors.h"

namespace Metavision {

namespace detail {
inline cv::Vec4b bgra(const cv::Vec3b &v) {
    return {v[0], v[1], v[2], 255};
}
inline cv::Vec4b rgba(const cv::Vec3b &v) {
    return {v[2], v[1], v[0], 255};
}
inline cv::Vec4b rgba(const cv::Vec4b &v) {
    return {v[2], v[1], v[0], v[3]};
}
inline cv::Vec3b bgr(const cv::Vec4b &v) {
    return {v[0], v[1], v[2]};
}
inline cv::Vec3b rgb(const cv::Vec4b &v) {
    return {v[2], v[1], v[0]};
}
inline cv::Vec3b rgb(const cv::Vec3b &v) {
    return {v[2], v[1], v[0]};
}
} // namespace detail

template<typename EventIt>
void BaseFrameGenerationAlgorithm::generate_frame_from_events(EventIt it_begin, EventIt it_end, cv::Mat &frame,
                                                              const uint32_t accumulation_time_us,
                                                              const Metavision::ColorPalette &palette) {
    const cv::Vec4b bg_color = get_bgra_color(palette, Metavision::ColorType::Background);
    const std::array<cv::Vec4b, 2> off_on_colors{get_bgra_color(palette, Metavision::ColorType::Negative),
                                                 get_bgra_color(palette, Metavision::ColorType::Positive)};
    int flags = (palette != Metavision::ColorPalette::Gray ? Parameters::BGR : Parameters::GRAY);

    // Process the entire range of events if the accumulation time is set to zero, or if there's no events.
    // Otherwise, find the first event to process in the desired time interval [t-dt, t[
    if (std::distance(it_begin, it_end) != 0 && accumulation_time_us != 0)
        it_begin = std::lower_bound(it_begin, it_end, std::prev(it_end)->t - accumulation_time_us,
                                    [](const auto &lhs, auto rhs) { return lhs.t < rhs; });

    generate_frame_from_events(it_begin, it_end, frame, bg_color, off_on_colors, flags);
}

template<typename EventIt>
void BaseFrameGenerationAlgorithm::generate_frame_from_events(EventIt it_begin, EventIt it_end, cv::Mat &frame,
                                                              const cv::Vec4b &bg_color,
                                                              const std::array<cv::Vec4b, 2> &off_on_colors,
                                                              int flags) {
    std::ostringstream ss;
    ss << "Incompatible matrix type. Must be ";
    int cv_type;
    std::string cv_type_str;
    if (flags & Parameters::GRAY) {
        cv_type     = CV_8UC1;
        cv_type_str = "CV_8UC1";
    } else if (flags & Parameters::RGB || flags & Parameters::BGR) {
        cv_type     = CV_8UC3;
        cv_type_str = "CV_8UC3";
    } else {
        cv_type     = CV_8UC4;
        cv_type_str = "CV_8UC4";
    }
    ss << cv_type_str << ".";
    if (frame.type() != cv_type) {
        throw std::invalid_argument(ss.str());
    }

    cv::Vec3b _bg_color3;
    cv::Vec4b _bg_color4;
    std::array<cv::Vec3b, 2> _off_on_colors3;
    std::array<cv::Vec4b, 2> _off_on_colors4;
    if (flags & Parameters::BGR || flags & Parameters::BGRA) {
        _bg_color3      = detail::bgr(bg_color);
        _off_on_colors3 = {detail::bgr(off_on_colors[0]), detail::bgr(off_on_colors[1])};
        _bg_color4      = bg_color;
        _off_on_colors4 = off_on_colors;
    } else {
        _bg_color3      = detail::rgb(bg_color);
        _off_on_colors3 = {detail::rgb(off_on_colors[0]), detail::rgb(off_on_colors[1])};
        _bg_color4      = detail::rgba(bg_color);
        _off_on_colors4 = {detail::rgba(off_on_colors[0]), detail::rgba(off_on_colors[1])};
    }

    if (flags & Parameters::GRAY) {
        frame.setTo(bg_color[0]);
        for (auto it = it_begin; it != it_end; ++it)
            frame.at<uint8_t>(it->y, it->x) = off_on_colors[it->p][0];
    } else if (flags & Parameters::RGB || flags & Parameters::BGR) {
        frame.setTo(_bg_color3);
        for (auto it = it_begin; it != it_end; ++it)
            frame.at<cv::Vec3b>(it->y, it->x) = _off_on_colors3[it->p];
    } else {
        frame.setTo(_bg_color4);
        for (auto it = it_begin; it != it_end; ++it)
            frame.at<cv::Vec4b>(it->y, it->x) = _off_on_colors4[it->p];
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_BASE_FRAME_GENERATION_ALGORITHM_IMPL_H