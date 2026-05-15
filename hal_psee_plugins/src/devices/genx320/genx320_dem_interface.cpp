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

#include <algorithm>
#include <memory>
#include <sstream>
#include <iomanip>

#include "metavision/psee_hw_layer/devices/genx320/genx320_dem_interface.h"

namespace Metavision {

GenX320DemInterface::GenX320DemInterface(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix) :
    mask_ctrl_(std::make_shared<GenX320DemDriver>(regmap, prefix)) {
    for (unsigned int i = 0; i < NUM_MASK_REGISTERS_; i++) {
        pixel_masks_.push_back(std::make_shared<GenX320PixelMask>(mask_ctrl_, i));
    }
}

const std::vector<I_DigitalEventMask::I_PixelMaskPtr> &GenX320DemInterface::get_pixel_masks() const {
    return pixel_masks_;
}

GenX320DemInterface::GenX320PixelMask::GenX320PixelMask(const std::shared_ptr<GenX320DemDriver> &driver,
                                                        const unsigned int id) :
    driver_(driver), id_(id) {}

bool GenX320DemInterface::GenX320PixelMask::set_mask(uint32_t x, uint32_t y, bool enabled) {
    auto vmask = driver_->vectorize(x, y);

    driver_->set_mask(vmask, id_, enabled);
    return true;
}

std::tuple<uint32_t, uint32_t, bool> GenX320DemInterface::GenX320PixelMask::get_mask() const {
    auto slot  = driver_->get_mask(id_);
    auto coord = driver_->extract_coord(slot.vmask_);
    auto valid = slot.is_valid();

    if (std::get<0>(coord) != -1) {
        return std::make_tuple(std::get<0>(coord), std::get<1>(coord), valid);
    } else {
        return std::make_tuple(0, 0, false);
    }
}

} // namespace Metavision
