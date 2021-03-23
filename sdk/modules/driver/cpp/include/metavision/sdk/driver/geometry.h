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

#ifndef METAVISION_SDK_DRIVER_GEOMETRY_H
#define METAVISION_SDK_DRIVER_GEOMETRY_H

#include "metavision/hal/facilities/i_geometry.h"

namespace Metavision {

/// @brief Facility class to get the geometry of a Device
class Geometry {
public:
    /// @brief Constructor
    Geometry(I_Geometry *geom);

    /// @brief Destructor
    ~Geometry();

    /// @brief Width of the sensor
    int width() const;

    /// @brief Height of the sensor
    int height() const;

    /// @brief Gets corresponding facility in HAL library
    I_Geometry *get_facility() const;

private:
    I_Geometry *pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_GEOMETRY_H
