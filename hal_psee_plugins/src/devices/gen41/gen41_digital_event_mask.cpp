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

#include "metavision/psee_hw_layer/devices/gen41/gen41_digital_event_mask.h"

namespace Metavision {

Gen41DigitalEventMask::Gen41DigitalEventMask(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix) :
    regmap_(regmap), prefix_(prefix), pixel_masks_(NUM_MASK_REGISTERS_) {
    size_t id = 0;
    std::generate_n(pixel_masks_.begin(), NUM_MASK_REGISTERS_, [&id, this]() {
        std::ostringstream oss;
        oss << prefix_ << std::setw(2) << std::setfill('0') << id++;
        const auto &mask_register = (*regmap_)[oss.str()];
        return std::make_shared<Gen41PixelMask>(mask_register);
    });
}

const std::vector<I_DigitalEventMask::I_PixelMaskPtr> &Gen41DigitalEventMask::get_pixel_masks() const {
    return pixel_masks_;
}

Gen41DigitalEventMask::Gen41PixelMask::Gen41PixelMask(const Register &reg) : reg_(reg) {}

bool Gen41DigitalEventMask::Gen41PixelMask::set_mask(uint32_t x, uint32_t y, bool enabled) {
    reg_["x"].write_value(x);
    reg_["y"].write_value(y);
    reg_["valid"].write_value(enabled);
    return true;
}

std::tuple<uint32_t, uint32_t, bool> Gen41DigitalEventMask::Gen41PixelMask::get_mask() const {
    int x        = reg_["x"].read_value();
    int y        = reg_["y"].read_value();
    bool enabled = reg_["valid"].read_value();
    return std::make_tuple(x, y, enabled);
}

} // namespace Metavision
