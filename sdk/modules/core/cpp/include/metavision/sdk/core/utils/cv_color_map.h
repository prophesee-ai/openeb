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

#ifndef METAVISION_SDK_CORE_CV_COLOR_MAP_H
#define METAVISION_SDK_CORE_CV_COLOR_MAP_H

#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace Metavision {

/// @brief Wrapper around cv::applyColorMap to avoid reallocating memory for the colormap at each call
///
/// The colormap look-up-table is created only once, when the class is constructed.
/// Then the parenthesis operator allows to apply it on an input image.
///
/// See implementation of "void ColorMap::operator()(InputArray _src, OutputArray _dst) const" in
/// https://github.com/opencv/opencv/blob/master/modules/imgproc/src/colormap.cpp
class CvColorMap {
public:
    /// @brief Initializes the colormap
    /// @param cmap Colormap to apply. See cv::ColormapTypes. It's cv::COLORMAP_JET by default
    inline CvColorMap(unsigned int cmap = 2) {
        cv::Mat gray(1, 256, CV_8UC1);
        uchar *p = gray.ptr();
        for (int i = 0; i < 256; ++i)
            p[i] = i;

        cv::applyColorMap(gray, lut_, cmap);
    }

    /// @brief Applies the colormap on a given image
    /// @param src Source image, grayscale or colored of type CV_8UC1 or CV_8UC3
    /// @param dst Result is the colormapped source image. Note: cv::Mat::create is called on dst
    inline void operator()(const cv::Mat &src, cv::Mat &dst) const {
        if (src.type() != CV_8UC1 && src.type() != CV_8UC3)
            throw std::invalid_argument("Invalid matrix type. Must be either CV_8UC1 or CV_8UC3");

        // Turn a BGR matrix into its grayscale representation
        if (src.type() == CV_8UC3) {
            cv::cvtColor(src, gray_img_tmp_, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray_img_tmp_, color_img_tmp_, cv::COLOR_GRAY2BGR);
        } else {
            cv::cvtColor(src, color_img_tmp_, cv::COLOR_GRAY2BGR);
        }
        cv::LUT(color_img_tmp_, lut_, dst);
    }

private:
    cv::Mat lut_;                                  ///< Colormap look-up-table
    mutable cv::Mat color_img_tmp_, gray_img_tmp_; ///< Temporary images
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_CV_COLOR_MAP_H
