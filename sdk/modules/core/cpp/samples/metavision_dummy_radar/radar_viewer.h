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

#ifndef RADAR_VIEWER_H
#define RADAR_VIEWER_H

#include <opencv2/core.hpp>

/// @brief Display class to show a "radar"-like view of the most prominent data
/// visible in the event stream
class RadarViewer {
public:
    struct Config {
        std::uint8_t n_bins_x = 10;         ///< Number of bins in the X direction
        std::uint8_t n_bins_y = 5;          ///< Number of bins in the Y direction
        float lateral_fov     = M_PI / 2.f; ///< Field of view in the X direction
        float min_ev_rate     = 40e3f;      ///< Minimum event rate computed in a bin for it to be displayed
        float max_ev_rate     = 100e6f;     ///< Maximum event rate computed in a bin for it to be displayed
    };

    /// @brief Constructor
    /// @param conf Display parameters
    /// @param sensor_width Sensor width in pixels
    /// @param sensor_height Sensor height in pixels
    RadarViewer(const Config &conf, int sensor_width, int sensor_height);

    /// @brief Destructor
    ~RadarViewer() {}

    /// @brief Computes the radar view from the event rate vector
    /// @param[in] histo Vector containing the event rate for each input bin
    /// @param[out] radar Image of the radar view
    void compute_view(std::vector<float> &histo, cv::Mat &radar);

private:
    void compute_radar_maps();
    void initialize_linear_grid();
    void reset_grid();

    const Config conf_;
    const int width_, height_;
    const int bin_width_, bin_height_;
    const float lateral_fov_;
    cv::Mat mapx_, mapy_;
    cv::Mat linear_bins_;
    std::vector<int> tmp_bin_heights_;
};

#endif // RADAR_VIEWER_H
