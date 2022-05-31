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

#ifndef METAVISION_HAL_TGEOMETRY_H
#define METAVISION_HAL_TGEOMETRY_H

#include "metavision/hal/facilities/i_geometry.h"

namespace Metavision {

template<int WIDTH, int HEIGHT>
class TGeometry : public I_Geometry {
public:
    virtual int get_width() const override final {
        return width_;
    }

    virtual int get_height() const override final {
        return height_;
    }

public:
    static constexpr int width_  = WIDTH;
    static constexpr int height_ = HEIGHT;
};

template<int WIDTH, int HEIGHT>
constexpr int TGeometry<WIDTH, HEIGHT>::width_;
template<int WIDTH, int HEIGHT>
constexpr int TGeometry<WIDTH, HEIGHT>::height_;

} // namespace Metavision

#endif // METAVISION_HAL_TGEOMETRY_H
