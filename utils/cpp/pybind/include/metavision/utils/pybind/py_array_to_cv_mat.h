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

#include <iostream>

namespace py = pybind11;

namespace Metavision {

/// @brief Creates a cv::Mat view using the memory stored inside a py::array (only for images)
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

/// @brief Creates a cv::Mat view using the memory stored inside a py::array (general type)
/// @param in Input array. Should be 2 or 3 dimensions: (H, W) or (H, W, C)
inline cv::Mat to_cv_mat(const py::array &in) {
    const auto num_dims = static_cast<size_t>(in.ndim());
    const auto &shape   = in.shape();
    const auto &strides = in.strides();
    const auto &dtype   = in.dtype();

    std::size_t num_channels = 0;
    if (num_dims == 2) {
        num_channels = 1;
    } else if (num_dims == 3) {
        num_channels = shape[2];
    } else {
        std::ostringstream oss;
        oss << "Invalid number of dimensions (should be either 2 or 3): " << num_dims;
        throw std::invalid_argument(oss.str());
    }

    int cv_depth;
    if (dtype.is(py::dtype::of<std::uint8_t>())) {
        cv_depth = CV_8U;
    } else if (dtype.is(py::dtype::of<std::int8_t>())) {
        cv_depth = CV_8S;
    } else if (dtype.is(py::dtype::of<std::uint16_t>())) {
        cv_depth = CV_16U;
    } else if (dtype.is(py::dtype::of<std::int16_t>())) {
        cv_depth = CV_16S;
    } else if (dtype.is(py::dtype::of<std::int32_t>())) {
        cv_depth = CV_32S;
    } else if (dtype.is(py::dtype::of<float>())) {
        cv_depth = CV_32F;
    } else if (dtype.is(py::dtype::of<double>())) {
        cv_depth = CV_64F;
    } else {
        throw std::runtime_error("This depth not implemented in the python bindings");
    }

    return cv::Mat(shape[0], shape[1], CV_MAKETYPE(cv_depth, num_channels), in.request().ptr, strides[0]);
}

/// @brief Creates a cv::Mat_<T> view using the memory stored inside a py::array_t<T>
/// @param in Input array. Should be 2 dimensions & monochannel: (H, W)
template<typename T>
inline cv::Mat_<T> to_cv_mat_(const py::array &in) {
    if (!py::isinstance<py::array_t<T>>(in)) {
        throw std::invalid_argument("Incompatible input dtype.");
    }
    const int num_dims = static_cast<size_t>(in.ndim());
    if (num_dims != 2) {
        std::ostringstream oss;
        oss << "Invalid number of dimensions (should be 2): " << num_dims;
        throw std::invalid_argument(oss.str());
    }

    const auto &shape   = in.shape();
    const auto &strides = in.strides();

    return cv::Mat_<T>(shape[0], shape[1], reinterpret_cast<T *>(in.request().ptr), strides[0]);
}

} // namespace Metavision

#endif // METAVISION_UTILS_PYBIND_PY_ARRAY_TO_CV_MAT_H
