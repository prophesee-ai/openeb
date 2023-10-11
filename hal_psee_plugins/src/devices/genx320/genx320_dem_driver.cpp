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

#include "metavision/psee_hw_layer/devices/genx320/genx320_dem_driver.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {

GenX320DemDriver::MaskSlot::MaskSlot(const RegAcc &reg_ctrl, const RegAcc &reg_data) :
    reg_ctrl_(reg_ctrl), reg_data_(reg_data), empty_(true) {
    vmask_ = {0, 0, 0};
};

void GenX320DemDriver::MaskSlot::update(bool enable) {
    reg_ctrl_.write_value(vfield({{"x_group", vmask_.x_group}, {"y", vmask_.y}, {"valid", enable}}));
    reg_data_["data"].write_value(vmask_.vector);
}

bool GenX320DemDriver::MaskSlot::is_valid() {
    return reg_ctrl_["valid"].read_value();
}

GenX320DemDriver::GenX320DemDriver(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix) :
    regmap_(regmap), prefix_(prefix) {
    mslots_.clear();

    for (unsigned int i = 0; i < NUM_MASK_SLOTS_; i++) {
        std::ostringstream reg_mask_ctrl, reg_mask_data;
        reg_mask_ctrl << "ro/crazy_pixel_ctrl" << std::dec << std::setw(2) << std::setfill('0') << i;
        reg_mask_data << "ro/crazy_pixel_data" << std::dec << std::setw(2) << std::setfill('0') << i;

        RegAcc reg_ctrl = (*regmap_)[reg_mask_ctrl.str()];
        RegAcc reg_data = (*regmap_)[reg_mask_data.str()];

        MaskSlot slot(reg_ctrl, reg_data);
        mslots_.push_back(slot);
    }
}

GenX320DemDriver::VectorMask GenX320DemDriver::vectorize(uint32_t x, uint32_t y) {
    // Translate coordinate to vector
    uint32_t vector_id  = uint32_t(x / 32);
    uint32_t vector_val = uint32_t(x % 32);

    GenX320DemDriver::VectorMask vmask = {y, vector_id, uint32_t(1 << vector_val)};
    return vmask;
}

int GenX320DemDriver::is_power_of_two(unsigned int n) {
    return n && (!(n & (n - 1)));
}

int GenX320DemDriver::find_position(unsigned int n) {
    if (!is_power_of_two(n))
        return -1;

    unsigned i = 1, pos = 1;

    // Iterate through bits of n till we find a set bit
    // i&n will be non-zero only when 'i' and 'n' have a set bit
    // at same position
    while (!(i & n)) {
        // Unset current bit and set the next bit in 'i'
        i = i << 1;

        // increment position
        ++pos;
    }

    return pos;
}

std::tuple<int, int> GenX320DemDriver::extract_coord(VectorMask vmask) {
    auto pos = find_position(vmask.vector);

    if (pos != -1) {
        return std::make_tuple(vmask.x_group * 32 + (pos - 1), vmask.y);
    } else {
        return std::make_tuple(-1, -1);
    }
}

bool GenX320DemDriver::set_pixel_filter(uint32_t x, uint32_t y, bool enabled) {
    VectorMask vmask = vectorize(x, y);

    // Search if pixel's group is already assigned a slot  vector in masks list
    std::vector<MaskSlot>::iterator it = find_if(mslots_.begin(), mslots_.end(), [vmask](MaskSlot slot) {
        return ((slot.empty_ == false) && (slot.vmask_.x_group == vmask.x_group) && (slot.vmask_.y == vmask.y));
    });

    if (it != mslots_.end()) {
        // Found matching slot already setup

        if (enabled) {
            // Update slot's vector
            it->vmask_.vector |= vmask.vector;
            it->reg_data_["data"].write_value(it->vmask_.vector);
        } else {
            // Update slot's vector
            it->vmask_.vector &= ~vmask.vector;
            it->reg_data_["data"].write_value(it->vmask_.vector);

            if (it->vmask_.vector == 0) {
                // Remove entry from mask list to free slot
                it->empty_ = true;
                it->reg_ctrl_["valid"].write_value(0);
            }
        }

    } else {
        // New slot needs to be assigned

        // Search for first empty slot in masks list
        std::vector<MaskSlot>::iterator first_free =
            find_if(mslots_.begin(), mslots_.end(), [](MaskSlot slot) { return (slot.empty_ == true); });

        if (enabled) {
            if (first_free == mslots_.end()) {
                //  No more slots available
                MV_HAL_LOG_WARNING() << "Cannot set new pixel mask. No more slots available";
                return false;
            } else {
                // Use first empty slot
                first_free->vmask_ = vmask;
                first_free->empty_ = false;

                // Update registers
                first_free->reg_ctrl_.write_value(
                    vfield({{"x_group", first_free->vmask_.x_group}, {"y", first_free->vmask_.y}, {"valid", 1}}));

                first_free->reg_data_["data"].write_value(first_free->vmask_.vector);
            }
        }
    }

    return true;
}

bool GenX320DemDriver::set_mask(VectorMask vmask, uint32_t id, bool enable) {
    // Update slot
    mslots_[id].vmask_ = vmask;
    mslots_[id].empty_ = false;
    mslots_[id].update(enable);

    return true;
}

const std::vector<GenX320DemDriver::MaskSlot> &GenX320DemDriver::get_masks() const {
    unsigned int i = 0;

    std::for_each(mslots_.begin(), mslots_.end(), [&i](const MaskSlot slot) {
        if (slot.empty_ == false) {
            std::cout << "Slot " << std::dec << i << ": y=" << slot.vmask_.y << ", x=" << slot.vmask_.x_group
                      << ", vector=0x" << std::hex << slot.vmask_.vector << std::dec << std::endl;
        } else {
            std::cout << "Slot " << std::dec << i << ": empty" << std::dec << std::endl;
        }
        i++;
    });
    return mslots_;
}

GenX320DemDriver::MaskSlot GenX320DemDriver::get_mask(const unsigned int id) const {
    return mslots_[id];
}

bool GenX320DemDriver::is_pixel_filtered(uint32_t x, uint32_t y) {
    VectorMask filter_slot = vectorize(x, y);

    // Search if pixel's group is already assigned a slot  vector in masks list
    std::vector<MaskSlot>::iterator it = find_if(mslots_.begin(), mslots_.end(), [filter_slot](MaskSlot n) {
        return ((n.empty_ == false) && (n.vmask_.x_group == filter_slot.x_group) && (n.vmask_.y == filter_slot.y));
    });

    if (it != mslots_.end()) {
        // Found matching slot already setup

        if ((it->vmask_.vector && filter_slot.vector) != 0) {
            // Pixel is set in the vector
            return true;
        } else {
            // Pixel not selected for filtering
            return false;
        }
    } else {
        // No slot setup for pixel coordinates
        return false;
    }
}

} // namespace Metavision
