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
#ifdef _WIN32
#define MODULE_NAME metavision_hal_internal
#else
#define MODULE_NAME metavision_hal
#endif

#include <pybind11/pybind11.h>

#include "metavision/utils/pybind/deprecation_warning_exception.h"
#include "hal_python_binder.h"
#include "metavision/hal/device/device.h"

namespace py = pybind11;

namespace Metavision {
namespace detail {
std::vector<std::function<void(py::module &, py::class_<Device, std::shared_ptr<Device>> &)>> &
    get_device_facility_getters_cbs() {
    static std::vector<std::function<void(py::module &, py::class_<Device, std::shared_ptr<Device>> &)>> s_vcb_device;
    return s_vcb_device;
}

} // namespace detail
} // namespace Metavision

PYBIND11_MODULE(MODULE_NAME, m) {
    PyEval_InitThreads();

    try {
        py::module::import("metavision_sdk_base");
    } catch (const std::exception &e) {
        std::cerr << "Exception Raised while loading metavision_sdk_base: " << e.what() << std::endl;
        throw(e);
    }

    // Register the translation for DeprecationWarningException
    py::register_exception<Metavision::DeprecationWarningException>(m, "Deprecated");

    Metavision::export_python_bindings<Metavision::metavision_hal>(m);

    // Export device and its facilities getter :
    auto device_python            = py::class_<Metavision::Device, std::shared_ptr<Metavision::Device>>(m, "Device");
    const auto &facilities_adders = Metavision::detail::get_device_facility_getters_cbs();
    for (auto &f : facilities_adders) {
        f(m, device_python);
    }
}
