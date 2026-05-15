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

#ifndef METAVISION_HAL_DEVICE_TRAITS_H
#define METAVISION_HAL_DEVICE_TRAITS_H

#include "devices/utils/device_system_id.h"
#include "geometries/tgeometry.h"

namespace Metavision {

template<typename Device>
struct metavision_device_traits {
    using RawEventFormat         = void;
    using Geometry               = TGeometry<0, 0>;
    static constexpr bool HAS_EM = false;

    static constexpr SystemId SYSTEM_ID_DEFAULT = SystemId::SYSTEM_INVALID_NO_FPGA;
    static constexpr long SUBSYSTEM_ID_DEFAULT  = -1;
};

} // namespace Metavision

#endif // METAVISION_HAL_DEVICE_TRAITS_H
