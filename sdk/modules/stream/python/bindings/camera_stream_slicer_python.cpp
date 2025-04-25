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

#include "metavision/sdk/stream/camera_stream_slicer.h"
#include "pb_doc_stream.h"
#include "rvalue_camera.h"

namespace py = pybind11;

namespace Metavision {

void export_camera_stream_slicer(py::module &m) {
    py::class_<Slice>(m, "Slice", pybind_doc_stream["Metavision::Slice"])
        .def(py::init<>())
        .def("__eq__", &Slice::operator==, pybind_doc_stream["Metavision::Slice::operator==(const Slice &other) const"],
             py::arg("other"))
        .def_property_readonly(
            "status", [](const Slice &self) { return self.status; }, pybind_doc_stream["Metavision::Slice::status"])
        .def_property_readonly(
            "t", [](const Slice &self) { return self.t; }, pybind_doc_stream["Metavision::Slice::t"])
        .def_property_readonly(
            "n_events", [](const Slice &self) { return self.n_events; },
            pybind_doc_stream["Metavision::Slice::n_events"])
        .def_property_readonly(
            "events",
            [](const Slice &self) {
                if (self.events->empty()) {
                    return py::array_t<EventCD>(self.events->size(), self.events->data());
                }
                auto capsule = py::capsule(self.events->data(), [](void *v) {});
                return py::array_t<EventCD>(self.events->size(), self.events->data(), capsule);
            },
            pybind_doc_stream["Metavision::Slice::events"])
        .def_property_readonly(
            "triggers",
            [](const Slice &self) {
                if (self.triggers->empty()) {
                    return py::array_t<EventExtTrigger>(self.triggers->size(), self.triggers->data());
                }
                auto capsule = py::capsule(self.triggers->data(), [](void *v) {});
                return py::array_t<EventExtTrigger>(self.triggers->size(), self.triggers->data(), capsule);
            },
            pybind_doc_stream["Metavision::Slice::triggers"]);

    py::enum_<Detail::ReslicingConditionType>(m, "ReslicingConditionType")
        .value("IDENTITY", Detail::ReslicingConditionType::IDENTITY)
        .value("N_EVENTS", Detail::ReslicingConditionType::N_EVENTS)
        .value("N_US", Detail::ReslicingConditionType::N_US)
        .value("MIXED", Detail::ReslicingConditionType::MIXED)
        .export_values();

    py::enum_<Detail::ReslicingConditionStatus>(m, "ReslicingConditionStatus")
        .value("NOT_MET", Detail::ReslicingConditionStatus::NOT_MET)
        .value("MET_AUTOMATIC", Detail::ReslicingConditionStatus::MET_AUTOMATIC)
        .value("MET_N_EVENTS", Detail::ReslicingConditionStatus::MET_N_EVENTS)
        .value("MET_N_US", Detail::ReslicingConditionStatus::MET_N_US)
        .export_values();

    py::class_<CameraStreamSlicer::SliceCondition>(m, "SliceCondition")
        .def(py::init<>())
        .def_readwrite("type", &CameraStreamSlicer::SliceCondition::type)
        .def_readwrite("delta_ts", &CameraStreamSlicer::SliceCondition::delta_ts)
        .def_readwrite("delta_n_events", &CameraStreamSlicer::SliceCondition::delta_n_events)
        .def("is_tracking_events_count", &CameraStreamSlicer::SliceCondition::is_tracking_events_count)
        .def("is_tracking_duration", &CameraStreamSlicer::SliceCondition::is_tracking_duration)
        .def_static("make_identity", &CameraStreamSlicer::SliceCondition::make_identity)
        .def_static("make_n_events", &CameraStreamSlicer::SliceCondition::make_n_events, py::arg("delta_n_events"))
        .def_static("make_n_us", &CameraStreamSlicer::SliceCondition::make_n_us, py::arg("delta_ts"))
        .def_static("make_mixed", &CameraStreamSlicer::SliceCondition::make_mixed, py::arg("delta_ts"),
                    py::arg("delta_n_events"));

    py::class_<CameraStreamSlicer>(m, "CameraStreamSlicer", pybind_doc_stream["Metavision::CameraStreamSlicer"])
        .def(py::init([](RValueCamera &rvalue_camera, const CameraStreamSlicer::SliceCondition &slice_condition,
                         std::size_t max_queue_size) {
                 if (!rvalue_camera.camera.has_value()) {
                     throw std::runtime_error("RValue Camera was already moved");
                 }

                 Camera camera = std::move(*rvalue_camera.camera);
                 rvalue_camera.camera.reset();

                 return CameraStreamSlicer(std::move(camera), slice_condition, max_queue_size);
             }),
             py::arg("rvalue_camera"), py::arg("slice_condition") = CameraStreamSlicer::SliceCondition::make_n_us(1000),
             py::arg("max_queue_size") = 5)
        .def("begin", &CameraStreamSlicer::begin, pybind_doc_stream["Metavision::CameraStreamSlicer::begin"])
        .def("camera", &CameraStreamSlicer::camera, pybind_doc_stream["Metavision::CameraStreamSlicer::camera"],
             py::return_value_policy::reference_internal)
        .def(
            "__iter__", [](CameraStreamSlicer &slicer) { return py::make_iterator(slicer.begin(), slicer.end()); },
            py::keep_alive<0, 1>());
}
} // namespace Metavision
