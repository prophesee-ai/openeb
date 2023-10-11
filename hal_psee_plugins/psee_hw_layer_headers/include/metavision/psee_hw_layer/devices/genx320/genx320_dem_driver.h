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

#ifndef METAVISION_HAL_GENX320_DEM_DRIVER_H
#define METAVISION_HAL_GENX320_DEM_DRIVER_H

#include "metavision/psee_hw_layer/utils/register_map.h"
#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Digital Event Mask implementation for GenX320
class GenX320DemDriver : public I_RegistrableFacility<GenX320DemDriver> {
    using RegAcc = RegisterMap::RegisterAccess;

public:
    struct VectorMask {
        uint32_t y;
        uint32_t x_group;
        uint32_t vector;
    };

    class MaskSlot {
    public:
        MaskSlot(const RegAcc &reg_ctrl, const RegAcc &reg_data);
        void update(bool enable);
        bool is_valid();

        bool empty_;
        VectorMask vmask_;
        RegAcc reg_ctrl_;
        RegAcc reg_data_;
    };

    GenX320DemDriver(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix);

    bool set_pixel_filter(uint32_t x, uint32_t y, bool enabled);
    bool is_pixel_filtered(uint32_t x, uint32_t y);

    const std::vector<MaskSlot> &get_masks() const;

    bool set_mask(VectorMask mask, uint32_t id, bool enable);
    MaskSlot get_mask(const unsigned int id) const;

    static VectorMask vectorize(uint32_t x, uint32_t y);
    static std::tuple<int, int> extract_coord(VectorMask vmask);

private:
    std::shared_ptr<RegisterMap> regmap_;
    std::string prefix_;
    std::vector<MaskSlot> mslots_;
    constexpr static size_t NUM_MASK_SLOTS_ = 16;

    static int is_power_of_two(unsigned int n);
    static int find_position(unsigned int n);
};

} // namespace Metavision

#endif // METAVISION_HAL_GENX320_DEM_DRIVER_H
