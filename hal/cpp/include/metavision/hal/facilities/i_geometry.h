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

#ifndef METAVISION_HAL_I_GEOMETRY_H
#define METAVISION_HAL_I_GEOMETRY_H

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Class to access information on the size of the sensor.
class I_Geometry : public I_RegistrableFacility<I_Geometry> {
public:
    /// @brief Returns width of the sensor in pixels
    /// @return Sensor's width
    virtual int get_width() const = 0;

    /// @brief Returns height of the sensor in pixels
    /// @return Sensor's height
    virtual int get_height() const = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_GEOMETRY_H
