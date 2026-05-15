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

#ifndef METAVISION_HAL_GENX320_ROI_PIXEL_RESET_H
#define METAVISION_HAL_GENX320_ROI_PIXEL_RESET_H

#include <vector>
#include "metavision/hal/facilities/i_registrable_facility.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_roi_driver.h"

namespace Metavision {

class GenX320RoiPixelReset : public I_RegistrableFacility<GenX320RoiPixelReset> {
public:
    GenX320RoiPixelReset(const std::shared_ptr<GenX320RoiDriver> &driver);

    void set(const bool &reset);

private:
    std::shared_ptr<GenX320RoiDriver> driver_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GENX320_ROI_PIXEL_RESET_H
