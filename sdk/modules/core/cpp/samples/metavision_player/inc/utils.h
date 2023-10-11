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

#ifndef METAVISION_PLAYER_UTILS_H
#define METAVISION_PLAYER_UTILS_H

#include <ctime>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <functional>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/core/utils/colors.h>

template<typename T>
T clip(const T &val, const T &minVal, const T &maxVal) {
    return std::min(std::max(val, minVal), maxVal);
}

inline void addTrackBar(const std::string &label, const std::string &window, int minValue, int maxValue,
                        void (*cb)(int, void *) = nullptr, void *data = nullptr) {
    cv::createTrackbar(label, window, nullptr, maxValue, cb, data);
    cv::setTrackbarMax(label, window, maxValue);
    cv::setTrackbarMin(label, window, minValue);
    cv::setTrackbarPos(label, window, minValue);
}

inline void addText(cv::Mat &frame, const std::string &text, const cv::Point &pos, const cv::Scalar &color) {
    cv::putText(frame, text, pos, cv::FONT_HERSHEY_PLAIN, 1, color, 1, cv::LINE_AA);
}

inline cv::Vec3b getCVColor(const Metavision::ColorPalette &palette, const Metavision::ColorType &type) {
    const Metavision::RGBColor c = Metavision::get_color(palette, type);
    return cv::Vec3b(c.b * 255 + 0.5, c.g * 255 + 0.5, c.r * 255 + 0.5);
}

inline cv::Size getCameraSize(const Metavision::Camera &camera) {
    return cv::Size(camera.geometry().width(), camera.geometry().height());
}

template<typename It>
std::pair<It, It> getSlice(It begin, It end, Metavision::timestamp t_begin, Metavision::timestamp t_end) {
    //
    // Look for the first element whose timestamp is > t_begin.
    auto slice_begin = std::lower_bound(begin, end, t_begin, [](auto evt, auto ts) { return evt.t <= ts; });

    // Look for the first element whose timestamp is >= t_end.
    // Note that we begin at *slice_begin*, this makes that for an inverted range, the output is empty, as intended.
    auto slice_end = std::upper_bound(slice_begin, end, t_end, [](auto ts, auto evt) { return evt.t > ts; });

    return std::pair<It, It>(slice_begin, slice_end);
}

template<typename InputIt>
inline size_t makeSliceImage(cv::Mat &frame_bgr, InputIt begin, InputIt end, Metavision::timestamp current_ts_us,
                             Metavision::timestamp accumulation_time_us, Metavision::timestamp frame_duration_us,
                             int display_fps, const Metavision::ColorPalette &palette) {
    auto slice_it = getSlice(begin, end, current_ts_us - accumulation_time_us, current_ts_us);

    const cv::Vec3b bg_color  = getCVColor(palette, Metavision::ColorType::Background);
    const cv::Vec3b pos_color = getCVColor(palette, Metavision::ColorType::Positive);
    const cv::Vec3b neg_color = getCVColor(palette, Metavision::ColorType::Negative);
    frame_bgr.setTo(bg_color);

    for (auto ev = slice_it.first; ev != slice_it.second; ++ev) {
        if (ev->p == 1) {
            frame_bgr.at<cv::Vec3b>(ev->y, ev->x) = pos_color;
        } else if (ev->p == 0) {
            frame_bgr.at<cv::Vec3b>(ev->y, ev->x) = neg_color;
        }
    }

    return std::distance(slice_it.first, slice_it.second);
}

inline std::string timeInSeconds(Metavision::timestamp t) {
    std::ostringstream oss;
    oss << std::setprecision(3) << std::fixed << (t / 1.e6) << "s";
    return oss.str();
}

inline std::string makeSliceImageOverlayText(size_t num_events, Metavision::timestamp current_ts_us,
                                             Metavision::timestamp accumulation_time_us,
                                             Metavision::timestamp frame_duration_us, int display_fps) {
    std::string msg = "Time : " + timeInSeconds(current_ts_us);
    msg += " Rate : " + std::to_string(num_events * 1'000 / accumulation_time_us) + "kev/s";
    msg += " Acc. : " + std::to_string(accumulation_time_us) + "us";

    const float fps          = (1e6 / frame_duration_us);
    const float speed_factor = display_fps / fps;
    std::stringstream ss;
    ss << " FPS : " << std::fixed << std::setprecision(1) << fps << " (" << std::setprecision(6) << std::defaultfloat
       << speed_factor << "X)";
    msg += ss.str();
    return msg;
}

inline std::string makeRawFilename(const std::string &basename) {
    char buf[1024];
    std::time_t t = std::time(nullptr);
    std::strftime(buf, 1024, "%Y-%m-%d_%H-%M-%S", std::localtime(&t));
    std::string filename(basename + "_" + buf + ".raw");
    return filename;
}

inline int compute_frame_period(int fps) {
    return static_cast<int>(1.e6 / fps + 0.5);
}

inline int compute_accumulation_time(int accumulation_ratio, int frame_period_us) {
    return static_cast<int>(accumulation_ratio * frame_period_us / 100. + 0.5);
}

#endif // METAVISION_PLAYER_UTILS_H
