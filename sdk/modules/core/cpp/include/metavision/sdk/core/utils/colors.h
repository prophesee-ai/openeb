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

#ifndef METAVISION_SDK_CORE_COLORS_H
#define METAVISION_SDK_CORE_COLORS_H

#include <unordered_map>
#include <string>
#include <opencv2/core.hpp>

namespace Metavision {

/// @brief Struct that represents a color in RGB colorspace
struct RGBColor {
    double r; ///< a fraction between 0 and 1
    double g; ///< a fraction between 0 and 1
    double b; ///< a fraction between 0 and 1
};

/// @brief Struct that represents a color in HSV colorspace
struct HSVColor {
    double h; ///< angle in degrees
    double s; ///< a fraction between 0 and 1
    double v; ///< a fraction between 0 and 1
};

/// @brief Utility function to convert from HSV to RGB colorspace
/// @param hsv A color in HSV colorspace
/// @return @ref RGBColor the color converted in RGB colorspace
///
inline RGBColor hsv2rgb(HSVColor hsv);

/// @brief Utility function to convert from RGB to HSV colorspace
/// @param rgb A color in RBB colorspace
/// @return @ref HSVColor the color converted in HSV colorspace
///
inline HSVColor rgb2hsv(RGBColor rgb);

/// @brief Enum class representing available color palettes
enum class ColorPalette { Light, Dark, CoolWarm, Gray };

/// @brief Enum class representing one of the possible type of colors
enum class ColorType { Background, Positive, Negative, Auxiliary };

/// @brief Gets a color given a palette and the color type
/// @param palette The requested color palette
/// @param type The requested color type
/// @return The color associated to the given palette and type
inline const RGBColor &get_color(const ColorPalette &palette, const ColorType &type);

/// @brief Gets a color given a palette and the color name
/// @param palette The requested color palette
/// @param name The requested color name
/// @return The color associated to the given palette and name
inline const RGBColor &get_color(const ColorPalette &palette, const std::string &name);

/// @brief Converts a RGBColor into a 8-bit BGR color in OpenCV format
/// @param c A color in RBG colorspace
/// @return The color in 8-bit BGR OpenCV format
inline cv::Vec3b get_bgr_color(const RGBColor &c);

/// @brief Gets a color in 8-bit BGR OpenCV format given a palette and the color type
/// @param palette The requested color palette
/// @param type The requested color type
/// @return The color in 8-bit BGR OpenCV format associated to the given palette and type
inline cv::Vec3b get_bgr_color(const ColorPalette &palette, const ColorType &type);

/// @brief Converts a RGBColor into a 8-bit BGRA color in OpenCV format
/// @param c A color in RBG colorspace
/// @return The color in 8-bit BGRA OpenCV format
inline cv::Vec4b get_bgra_color(const RGBColor &c);

/// @brief Gets a color in 8-bit BGRA OpenCV format given a palette and the color type
/// @param palette The requested color palette
/// @param type The requested color type
/// @return The color in 8-bit BGRA OpenCV format associated to the given palette and type
inline cv::Vec4b get_bgra_color(const ColorPalette &palette, const ColorType &type);

} // namespace Metavision

#include "detail/colors_impl.h"

#endif // METAVISION_SDK_CORE_COLORS_H
