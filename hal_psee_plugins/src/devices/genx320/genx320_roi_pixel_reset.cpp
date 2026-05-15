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

#include "metavision/psee_hw_layer/devices/genx320/genx320_roi_pixel_reset.h"

namespace Metavision {

GenX320RoiPixelReset::GenX320RoiPixelReset(const std::shared_ptr<GenX320RoiDriver> &driver) : driver_(driver) {}

void GenX320RoiPixelReset::set(const bool &reset) {
    if (driver_->get_driver_mode() != GenX320RoiDriver::DriverMode::IO) {
        driver_->set_driver_mode(GenX320RoiDriver::DriverMode::IO);
    }
    driver_->pixel_reset(reset);
}

} // namespace Metavision