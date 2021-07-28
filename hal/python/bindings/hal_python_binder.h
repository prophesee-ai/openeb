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

#ifndef METAVISION_HAL_PYTHON_BINDER_H
#define METAVISION_HAL_PYTHON_BINDER_H

#include <functional>
#include <vector>

#include "metavision/utils/pybind/python_binder_helper.h"
#include "metavision/hal/device/device.h"

namespace Metavision {

struct metavision_hal {};

using HALGenericPythonBinder = GenericPythonBinderHelper<metavision_hal>;

template<typename ClassName, typename... Args>
using HALFacilityPythonBinder = ClassPythonBinderHelper<metavision_hal, ClassName, std::shared_ptr<ClassName>, Args...>;

template<typename... Args>
using HALClassPythonBinder = ClassPythonBinderHelper<metavision_hal, Args...>;

namespace detail {

std::vector<std::function<void(py::module &, py::class_<Device, std::shared_ptr<Device>> &)>> &
    get_device_facility_getters_cbs();

} // namespace detail

template<typename Facility>
struct DeviceFacilityGetter {
    DeviceFacilityGetter(const std::string &getter_name) {
        detail::get_device_facility_getters_cbs().push_back([getter_name](auto &module, auto &device_python) {
            device_python.def(&getter_name[0], &Device::get_facility<Facility>, py::return_value_policy::reference);
        });
    }
};

} // namespace Metavision

#endif // METAVISION_HAL_PYTHON_BINDER_H
