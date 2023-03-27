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

#ifndef METAVISION_SDK_CORE_DETAIL_COLORS_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_COLORS_IMPL_H

#include <cmath>

namespace Metavision {

inline HSVColor rgb2hsv(RGBColor in) {
    HSVColor out;
    double min, max, delta;

    min = in.r < in.g ? in.r : in.g;
    min = min < in.b ? min : in.b;

    max = in.r > in.g ? in.r : in.g;
    max = max > in.b ? max : in.b;

    out.v = max; // v
    delta = max - min;
    if (delta < 0.00001) {
        out.s = 0;
        out.h = 0; // undefined, maybe nan?
        return out;
    }
    if (max > 0.0) {           // NOTE: if Max is == 0, this divide would cause a crash
        out.s = (delta / max); // s
    } else {
        // if max is 0, then r = g = b = 0
        // s = 0, h is undefined
        out.s = 0.0;
        out.h = NAN; // its now undefined
        return out;
    }
    if (in.r >= max)                   // > is bogus, just keeps compiler happy
        out.h = (in.g - in.b) / delta; // between yellow & magenta
    else if (in.g >= max)
        out.h = 2.0 + (in.b - in.r) / delta; // between cyan & yellow
    else
        out.h = 4.0 + (in.r - in.g) / delta; // between magenta & cyan

    out.h *= 60.0; // degrees

    if (out.h < 0.0)
        out.h += 360.0;

    return out;
}

inline RGBColor hsv2rgb(HSVColor in) {
    double hh, p, q, t, ff;
    long i;
    RGBColor out;

    if (in.s <= 0.0) { // < is bogus, just shuts up warnings
        out.r = in.v;
        out.g = in.v;
        out.b = in.v;
        return out;
    }
    hh = in.h;
    if (hh >= 360.0)
        hh = 0.0;
    hh /= 60.0;
    i  = (long)hh;
    ff = hh - i;
    p  = in.v * (1.0 - in.s);
    q  = in.v * (1.0 - (in.s * ff));
    t  = in.v * (1.0 - (in.s * (1.0 - ff)));

    switch (i) {
    case 0:
        out.r = in.v;
        out.g = t;
        out.b = p;
        break;
    case 1:
        out.r = q;
        out.g = in.v;
        out.b = p;
        break;
    case 2:
        out.r = p;
        out.g = in.v;
        out.b = t;
        break;

    case 3:
        out.r = p;
        out.g = q;
        out.b = in.v;
        break;
    case 4:
        out.r = t;
        out.g = p;
        out.b = in.v;
        break;
    case 5:
    default:
        out.r = in.v;
        out.g = p;
        out.b = q;
        break;
    }
    return out;
}

namespace detail {
static std::unordered_map<std::string, RGBColor> LightColorPaletteMap{
    {"background", RGBColor{1.0f, 1.0f, 1.0f}},
    {"positive", RGBColor{0.25f, 0.4921875f, 0.78515625f}},
    {"negative", RGBColor{0.1171875f, 0.14453125f, 0.203125f}},
    {"auxiliary", RGBColor{1.0f, 0.5607843f, 0.4235294f}},
};
static std::unordered_map<std::string, RGBColor> DarkColorPaletteMap{
    {"background", RGBColor{0.1171875f, 0.14453125f, 0.203125f}},
    {"positive", RGBColor{1.0f, 1.0f, 1.0f}},
    {"negative", RGBColor{0.25f, 0.4921875f, 0.78515625f}},
    {"auxiliary", RGBColor{1.0f, 0.5607843f, 0.4235294f}},
};
static std::unordered_map<std::string, RGBColor> CoolWarmColorPaletteMap{
    {"background", RGBColor{0.8509804f, 0.8784314f, 0.9294118f}},
    {"positive", RGBColor{1.f, 0.4431373f, 0.4588235f}},
    {"negative", RGBColor{0.3411765f, 0.4823529f, 0.7764706f}},
    {"auxiliary", RGBColor{0.1372549f, 0.1411765f, 0.1764706f}},
};
static std::unordered_map<std::string, RGBColor> GrayColorPaletteMap{
    {"background", RGBColor{0.5f, 0.5f, 0.5f}},
    {"positive", RGBColor{1.0f, 1.0f, 1.0f}},
    {"negative", RGBColor{0.0f, 0.0f, 0.0f}},
    {"auxiliary", RGBColor{1.0f, 1.0f, 0.0f}},
};

inline const std::string &colorTypeToName(const ColorType &type) {
    static std::unordered_map<ColorType, std::string> m{{ColorType::Background, "background"},
                                                        {ColorType::Positive, "positive"},
                                                        {ColorType::Negative, "negative"},
                                                        {ColorType::Auxiliary, "auxiliary"}};
    auto it = m.find(type);
    if (it == m.end()) {
        throw std::runtime_error("Unknown color type " + std::to_string(static_cast<int>(type)));
    }
    return it->second;
}
} // namespace detail

inline const RGBColor &get_color(const ColorPalette &palette, const ColorType &type) {
    switch (palette) {
    case ColorPalette::Light: {
        auto it = detail::LightColorPaletteMap.find(detail::colorTypeToName(type));
        return it->second;
    }
    case ColorPalette::Dark: {
        auto it = detail::DarkColorPaletteMap.find(detail::colorTypeToName(type));
        return it->second;
    }
    case ColorPalette::CoolWarm: {
        auto it = detail::CoolWarmColorPaletteMap.find(detail::colorTypeToName(type));
        return it->second;
    }
    case ColorPalette::Gray: {
        auto it = detail::GrayColorPaletteMap.find(detail::colorTypeToName(type));
        return it->second;
    }
    default:
        break;
    }
    throw std::runtime_error("Unknown color palette " + std::to_string(static_cast<int>(palette)));
}

inline const RGBColor &get_color(const ColorPalette &palette, const std::string &name) {
    switch (palette) {
    case ColorPalette::Light: {
        auto it = detail::LightColorPaletteMap.find(name);
        if (it == detail::LightColorPaletteMap.end()) {
            throw std::runtime_error("Unknown color name " + name);
        }
        return it->second;
    }
    case ColorPalette::Dark: {
        auto it = detail::DarkColorPaletteMap.find(name);
        if (it == detail::DarkColorPaletteMap.end()) {
            throw std::runtime_error("Unknown color name " + name);
        }
        return it->second;
    }
    case ColorPalette::CoolWarm: {
        auto it = detail::CoolWarmColorPaletteMap.find(name);
        if (it == detail::CoolWarmColorPaletteMap.end()) {
            throw std::runtime_error("Unknown color name " + name);
        }
        return it->second;
    }
    case ColorPalette::Gray: {
        auto it = detail::GrayColorPaletteMap.find(name);
        if (it == detail::GrayColorPaletteMap.end()) {
            throw std::runtime_error("Unknown color name " + name);
        }
        return it->second;
    }
    default:
        break;
    }
    throw std::runtime_error("Unknown color palette " + std::to_string(static_cast<int>(palette)));
}

inline cv::Vec3b get_bgr_color(const RGBColor &c) {
    return cv::Vec3b(static_cast<uchar>(c.b * 255 + 0.5), static_cast<uchar>(c.g * 255 + 0.5),
                     static_cast<uchar>(c.r * 255 + 0.5));
}

inline cv::Vec3b get_bgr_color(const ColorPalette &palette, const ColorType &type) {
    return get_bgr_color(get_color(palette, type));
}

inline cv::Vec4b get_bgra_color(const RGBColor &c) {
    return cv::Vec4b(static_cast<uchar>(c.b * 255 + 0.5), static_cast<uchar>(c.g * 255 + 0.5),
                     static_cast<uchar>(c.r * 255 + 0.5), 255);
}

inline cv::Vec4b get_bgra_color(const ColorPalette &palette, const ColorType &type) {
    return get_bgra_color(get_color(palette, type));
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_COLORS_IMPL_H
