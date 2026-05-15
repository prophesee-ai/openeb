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

#include "metavision/sdk/stream/synced_camera_streams_slicer.h"
#include "metavision/sdk/stream/synced_camera_system_builder.h"

#include "pb_doc_stream.h"
#include "rvalue_camera.h"

namespace py = pybind11;

namespace Metavision {

void export_synced_cameras_stream_slicer(py::module &m) {
    py::class_<SyncedSlice>(m, "SyncedSlice", pybind_doc_stream["Metavision::SyncedSlice"])
        .def(py::init<>(), pybind_doc_stream["Metavision::SyncedSlice::SyncedSlice()=default"])
        .def("__eq__", &SyncedSlice::operator==,
             pybind_doc_stream["Metavision::SyncedSlice::operator==(const SyncedSlice &other) const"], py::arg("other"))
        .def_property_readonly(
            "status", [](const SyncedSlice &self) { return self.status; },
            pybind_doc_stream["Metavision::SyncedSlice::status"])
        .def_property_readonly(
            "t", [](const SyncedSlice &self) { return self.t; }, pybind_doc_stream["Metavision::SyncedSlice::t"])
        .def_property_readonly(
            "n_events", [](const SyncedSlice &self) { return self.n_events; },
            pybind_doc_stream["Metavision::SyncedSlice::n_events"])
        .def_property_readonly(
            "master_events",
            [](const SyncedSlice &self) {
                if (self.master_events->empty()) {
                    return py::array_t<EventCD>(self.master_events->size(), self.master_events->data());
                }
                auto capsule = py::capsule(self.master_events->data(), [](void *v) {});
                return py::array_t<EventCD>(self.master_events->size(), self.master_events->data(), capsule);
            },
            pybind_doc_stream["Metavision::SyncedSlice::master_events"])
        .def_property_readonly(
            "master_triggers",
            [](const SyncedSlice &self) {
                if (self.master_triggers->empty()) {
                    return py::array_t<EventExtTrigger>(self.master_triggers->size(), self.master_triggers->data());
                }
                auto capsule = py::capsule(self.master_triggers->data(), [](void *v) {});
                return py::array_t<EventExtTrigger>(self.master_triggers->size(), self.master_triggers->data(),
                                                    capsule);
            },
            pybind_doc_stream["Metavision::SyncedSlice::master_triggers"])
        .def_property_readonly(
            "slave_events",
            [](const SyncedSlice &self) {
                py::list slave_events;

                auto capsule = py::capsule(self.slave_events.data(), [](void *v) {});

                for (const auto &slave_event : self.slave_events) {
                    slave_events.append(py::array_t<EventCD>(slave_event->size(), slave_event->data(), capsule));
                }

                return slave_events;
            },
            pybind_doc_stream["Metavision::SyncedSlice::slave_events"]);

    py::class_<SyncedCameraStreamsSlicer>(m, "SyncedCameraStreamsSlicer",
                                          pybind_doc_stream["Metavision::SyncedCameraStreamsSlicer"])
        .def(py::init([](RValueCamera &rvalue_master, py::list rvalue_slaves,
                         SyncedCameraStreamsSlicer::SliceCondition slice_condition, size_t max_queue_size) {
                 std::vector<Camera> slave_cameras;
                 for (auto &element : rvalue_slaves) {
                     auto &rvalue_slave = py::cast<RValueCamera &>(element);
                     if (!rvalue_slave.camera.has_value()) {
                         throw std::runtime_error("RValue Camera was already moved");
                     }

                     slave_cameras.emplace_back(std::move(*rvalue_slave.camera));
                     rvalue_slave.camera.reset();
                 }

                 if (!rvalue_master.camera.has_value()) {
                     throw std::runtime_error("RValue Camera was already moved");
                 }

                 Camera master = std::move(*rvalue_master.camera);
                 rvalue_master.camera.reset();

                 return SyncedCameraStreamsSlicer(std::move(master), std::move(slave_cameras), slice_condition,
                                                  max_queue_size);
             }),
             py::arg("camera"), py::arg("Cameras"),
             py::arg("slice_condition") = SyncedCameraStreamsSlicer::SliceCondition::make_n_us(1000),
             py::arg("max_queue_size")  = 5)
        .def("begin", &SyncedCameraStreamsSlicer::begin,
             pybind_doc_stream["Metavision::SyncedCameraStreamsSlicer::begin"])
        .def("master", &SyncedCameraStreamsSlicer::master,
             pybind_doc_stream["Metavision::SyncedCameraStreamsSlicer::master"],
             py::return_value_policy::reference_internal)
        .def("slaves_count", &SyncedCameraStreamsSlicer::slaves_count,
             pybind_doc_stream["Metavision::SyncedCameraStreamsSlicer::slaves_count"])
        .def("slave", &SyncedCameraStreamsSlicer::slave, py::arg("i"),
             pybind_doc_stream["Metavision::SyncedCameraStreamsSlicer::slave"],
             py::return_value_policy::reference_internal)
        .def(
            "__iter__",
            [](SyncedCameraStreamsSlicer &slicer) { return py::make_iterator(slicer.begin(), slicer.end()); },
            py::keep_alive<0, 1>());
}
} // namespace Metavision
