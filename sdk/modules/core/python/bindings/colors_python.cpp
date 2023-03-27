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

#include <pybind11/pybind11.h>

#include "metavision/sdk/core/utils/colors.h"
#include "pb_doc_core.h"

namespace py = pybind11;

namespace Metavision {

void export_colors(py::module &m) {
    using namespace pybind11::literals;

    py::enum_<ColorPalette>(m, "ColorPalette")
        .value("Light", ColorPalette::Light)
        .value("Dark", ColorPalette::Dark)
        .value("CoolWarm", ColorPalette::CoolWarm)
        .value("Gray", ColorPalette::Gray)
        .export_values();

    py::enum_<ColorType>(m, "ColorType")
        .value("Background", ColorType::Background)
        .value("Positive", ColorType::Positive)
        .value("Negative", ColorType::Negative)
        .value("Auxiliary", ColorType::Auxiliary)
        .export_values();

    m.def(
        "rgb2hsv",
        [](double r, double g, double b) {
            const HSVColor hsv = Metavision::rgb2hsv({r, g, b});
            return py::make_tuple(hsv.h, hsv.s, hsv.v);
        },
        py::arg("r"), py::arg("g"), py::arg("b"));

    m.def(
        "hsv2rgb",
        [](double h, double s, double v) {
            const RGBColor rgb = Metavision::hsv2rgb({h, s, v});
            return py::make_tuple(rgb.r, rgb.g, rgb.b);
        },
        py::arg("h"), py::arg("s"), py::arg("v"));

    m.def(
        "getColor",
        [](const ColorPalette &palette, const ColorType &type) {
            const RGBColor rgb = Metavision::get_color(palette, type);
            return py::make_tuple(rgb.r, rgb.g, rgb.b);
        },
        py::arg("palette"), py::arg("type"));
}

} // namespace Metavision