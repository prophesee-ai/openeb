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

#ifndef METAVISION_UTILS_PYBIND_PY_ARRAY_TO_CV_MAT_H
#define METAVISION_UTILS_PYBIND_PY_ARRAY_TO_CV_MAT_H

#include <opencv2/core/mat.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

namespace Metavision {

/// @brief Creates a cv::Mat view using the memory stored inside a py::array
/// @param py_image Input image in the py::array_t<std::uint8_t> format
/// @param output_cv_mat Output image in cv::Mat format (Either CV_8UC3 or CV_8UC1)
/// @param colored True if the image is of type CV_8UC3. False if it's of type CV_8UC1
inline void py_array_to_cv_mat(const py::array &py_image, cv::Mat &output_cv_mat, bool colored) {
    if (!py::isinstance<py::array_t<std::uint8_t>>(py_image))
        throw std::invalid_argument("Incompatible input dtype. Must be np.ubyte.");

    const size_t num_channels = (colored ? 3 : 2);
    if (static_cast<size_t>(py_image.ndim()) != num_channels) {
        std::stringstream ss;
        ss << "Incompatible dimensions number. Must be a " << num_channels << " dimensional image.";
        throw std::invalid_argument(ss.str());
    }

    const auto &shape   = py_image.shape();
    const auto &strides = py_image.strides();

    output_cv_mat = cv::Mat(shape[0], shape[1], (colored ? CV_8UC3 : CV_8UC1), py_image.request().ptr, strides[0]);
}

} // namespace Metavision

#endif // METAVISION_UTILS_PYBIND_PY_ARRAY_TO_CV_MAT_H
