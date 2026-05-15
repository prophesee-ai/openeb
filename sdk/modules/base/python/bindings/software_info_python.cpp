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

#include "metavision/sdk/base/utils/software_info.h"

#include "pb_doc_base.h"

namespace py = pybind11;

namespace Metavision {

void export_software_info(py::module &m) {
    py::class_<SoftwareInfo>(m, "SoftwareInfo", pybind_doc_base["Metavision::SoftwareInfo"])
        .def(py::init<int, int, int, const std::string &, const std::string &, const std::string &,
                      const std::string &>(),
             py::arg("version_major"), py::arg("version_minor"), py::arg("version_patch"),
             py::arg("version_suffix_string"), py::arg("vcs_branch"), py::arg("vcs_commit"), py::arg("vcs_date"),
             pybind_doc_base["Metavision::SoftwareInfo::SoftwareInfo(int version_major, int version_minor, int "
                             "version_patch, const std::string &version_suffix_string, const std::string &vcs_branch, "
                             "const std::string &vcs_commit, const std::string &vcs_date)"])
        .def("get_version_major", &Metavision::SoftwareInfo::get_version_major,
             pybind_doc_base["Metavision::SoftwareInfo::get_version_major"])
        .def("get_version_minor", &Metavision::SoftwareInfo::get_version_minor,
             pybind_doc_base["Metavision::SoftwareInfo::get_version_minor"])
        .def("get_version_patch", &Metavision::SoftwareInfo::get_version_patch,
             pybind_doc_base["Metavision::SoftwareInfo::get_version_patch"])
        .def("get_version_suffix", &Metavision::SoftwareInfo::get_version_suffix,
             pybind_doc_base["Metavision::SoftwareInfo::get_version_suffix"])
        .def("get_version", &Metavision::SoftwareInfo::get_version,
             pybind_doc_base["Metavision::SoftwareInfo::get_version"])
        .def("get_vcs_branch", &Metavision::SoftwareInfo::get_vcs_branch,
             pybind_doc_base["Metavision::SoftwareInfo::get_vcs_branch"])
        .def("get_vcs_commit", &Metavision::SoftwareInfo::get_vcs_commit,
             pybind_doc_base["Metavision::SoftwareInfo::get_vcs_commit"])
        .def("get_vcs_date", &Metavision::SoftwareInfo::get_vcs_date,
             pybind_doc_base["Metavision::SoftwareInfo::get_vcs_date"]);
}

} // namespace Metavision
