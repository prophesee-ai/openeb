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

#ifndef METAVISION_HAL_GEN4CD_DEVICE_H
#define METAVISION_HAL_GEN4CD_DEVICE_H

#include "metavision_device_traits.h"
#include "evt3_raw_format.h"
#include "geometries/hd_geometry.h"

namespace Metavision {

struct Gen4CDDevice {};

template<>
struct metavision_device_traits<Gen4CDDevice> {
    using RawEventFormat = Evt3RawFormat;
    using Geometry       = HDGeometry;

    static constexpr SystemId SYSTEM_ID_DEFAULT = SystemId::SYSTEM_CCAM3_GEN4;
    static constexpr long SUBSYSTEM_ID_DEFAULT  = -1;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN4CD_DEVICE_H
