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
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "metavision/utils/pybind/pod_event_buffer.h"
#include "metavision/sdk/base/events/event_cd.h"

namespace py = pybind11;

namespace { // anonymous

unsigned long buffer_info_get_ptr_int(const py::buffer_info &buf_info) {
    return (unsigned long)(buf_info.ptr);
}

std::string buffer_info_get_ptr_hex(const py::buffer_info &buf_info) {
    std::ostringstream oss;
    oss << "0x" << std::hex << buffer_info_get_ptr_int(buf_info);
    return oss.str();
}

py::dict buffer_info_dict(const py::buffer_info &buf_info) {
    py::dict d;
    d["ptr"]      = buffer_info_get_ptr_int(buf_info);
    d["ptr_hex"]  = buffer_info_get_ptr_hex(buf_info);
    d["itemsize"] = buf_info.itemsize;
    d["size"]     = buf_info.size;
    d["format"]   = buf_info.format;
    d["ndim"]     = buf_info.ndim;
    d["shape"]    = buf_info.shape;
    d["strides"]  = buf_info.strides;
    return d;
}

py::buffer_info get_np_buffer_info(const py::array &arr) {
    return arr.request();
}

} // namespace

namespace Metavision {

void export_debug_buffer_info(py::module &m) {
    py::class_<py::buffer_info>(m, "_BufferInfo")
        .def(py::init<>())
        .def("ptr", &buffer_info_get_ptr_int)
        .def("ptr_hex", &buffer_info_get_ptr_hex)
        .def_readwrite("itemsize", &py::buffer_info::itemsize)
        .def_readwrite("size", &py::buffer_info::size)
        .def_readwrite("format", &py::buffer_info::format)
        .def_readwrite("ndim", &py::buffer_info::ndim)
        .def_readwrite("shape", &py::buffer_info::shape)
        .def_readwrite("strides", &py::buffer_info::strides)
        .def("to_dict", &buffer_info_dict);

    m.def("_buffer_info", &get_np_buffer_info);
}

} // namespace Metavision
