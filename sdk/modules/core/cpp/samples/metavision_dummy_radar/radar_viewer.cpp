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

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "radar_viewer.h"

RadarViewer::RadarViewer(const Config &conf, int sensor_width, int sensor_height) :
    conf_(conf),
    width_(sensor_width),
    height_(sensor_height),
    bin_width_(width_ / conf_.n_bins_x),
    bin_height_(height_ / conf_.n_bins_y),
    lateral_fov_(conf.lateral_fov) {
    initialize_linear_grid();
    compute_radar_maps();
    tmp_bin_heights_.resize(conf_.n_bins_x);
}

void RadarViewer::compute_view(std::vector<float> &histo, cv::Mat &radar) {
    const cv::Size s(width_, height_);

    auto max_val = std::max_element(histo.cbegin(), histo.cend());
    if (*max_val == 0)
        return;

    const int i             = std::distance(histo.cbegin(), max_val);
    const float ev_rate_bin = histo[i];
    const bool draw_max     = (ev_rate_bin >= conf_.min_ev_rate && ev_rate_bin < conf_.max_ev_rate);
    if (draw_max) {
        const float ratio   = (ev_rate_bin - conf_.min_ev_rate) / (conf_.max_ev_rate - conf_.min_ev_rate);
        const int max_val_y = static_cast<int>(ratio * conf_.n_bins_y) * bin_height_;
        tmp_bin_heights_[i] = max_val_y;
        cv::rectangle(linear_bins_, cv::Point(i * bin_width_, max_val_y),
                      cv::Point((i + 1) * bin_width_ - 1, height_ - 1), cv::Scalar(255, 255, 255));
    }
    cv::remap(linear_bins_, radar, mapx_, mapy_, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);

    if (draw_max) {
        // Draw back initial rectangle grid
        const int max_val_y = tmp_bin_heights_[i];
        cv::rectangle(linear_bins_, cv::Point(i * bin_width_, max_val_y),
                      cv::Point((i + 1) * bin_width_ - 1, height_ - 1), cv::Scalar(0, 0, 0));

        reset_grid();
    }
}

void RadarViewer::compute_radar_maps() {
    const cv::Size img_size(width_, height_);

    // Initialize remapping maps with polar data
    cv::Mat tmp_mapx_(img_size, CV_32FC1);
    cv::Mat tmp_mapy_(img_size, CV_32FC1);
    tmp_mapx_.setTo(-1.f);
    tmp_mapy_.setTo(-1.f);

    const float x_c     = width_ / 2.f;
    const float y_c     = height_;
    const float width_f = static_cast<float>(width_);
    for (int i = 0; i < height_; ++i) {
        for (int j = 0; j < width_; ++j) {
            const float theta = (j / width_f - 0.5f) * lateral_fov_;
            const int x_pol   = static_cast<int>(std::round(x_c + (height_ - i) * sin(theta)));
            const int y_pol   = static_cast<int>(std::round(y_c - (height_ - i) * cos(theta)));
            if (x_pol >= 0 && x_pol < width_ && y_pol >= 0 && y_pol < height_) {
                // Some pixels will be written several times if the cast to int provides
                // same value (in particular for neighboring pixels
                tmp_mapx_.at<float>(y_pol, x_pol) = j;
                tmp_mapy_.at<float>(y_pol, x_pol) = i;
            }
        }
    }

    // Reset grid lines coordinates which may have been overridden by neighboring pixels
    cv::Mat grid(img_size, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < conf_.n_bins_x; ++i) {
        for (int k = 0; k < conf_.n_bins_y; ++k) {
            cv::rectangle(grid, cv::Point(i * bin_width_, k * bin_height_),
                          cv::Point((i + 1) * bin_width_ - 1, (k + 1) * bin_height_), cv::Scalar(255));
        }
    }
    for (int i = 0; i < height_; ++i) {
        for (int j = 0; j < width_; ++j) {
            if (grid.at<std::uint8_t>(i, j)) {
                const float theta = (j / width_f - 0.5f) * lateral_fov_;
                const int x_pol   = static_cast<int>(std::round(x_c + (height_ - i) * sin(theta)));
                const int y_pol   = static_cast<int>(std::round(y_c - (height_ - i) * cos(theta)));
                if (x_pol >= 0 && x_pol < width_ && y_pol >= 0 && y_pol < height_) {
                    tmp_mapx_.at<float>(y_pol, x_pol) = j;
                    tmp_mapy_.at<float>(y_pol, x_pol) = i;
                }
            }
        }
    }
    cv::convertMaps(tmp_mapx_, tmp_mapy_, mapx_, mapy_, CV_16SC2, true);
}

void RadarViewer::initialize_linear_grid() {
    linear_bins_.create(height_, width_, CV_8UC3);
    linear_bins_.setTo(0);

    reset_grid();
}

void RadarViewer::reset_grid() {
    for (int i = 0; i < conf_.n_bins_x; ++i) {
        for (int k = 0; k < conf_.n_bins_y; ++k) {
            cv::rectangle(linear_bins_, cv::Point(i * bin_width_, k * bin_height_),
                          cv::Point((i + 1) * bin_width_ - 1, (k + 1) * bin_height_), cv::Scalar(0, 255, 0));
        }
    }
}
