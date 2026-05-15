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

#include "metavision/psee_hw_layer/devices/genx320/genx320_roi_pixel_mask_interface.h"

namespace Metavision {

GenX320RoiPixelMaskInterface::GenX320RoiPixelMaskInterface(const std::shared_ptr<GenX320RoiDriver> &driver) :
    driver_(driver) {}

bool GenX320RoiPixelMaskInterface::set_pixel(const unsigned int &column, const unsigned int &row, const bool &enable) {
    return driver_->set_pixel(column, row, enable);
}

std::vector<std::pair<unsigned int, unsigned int>> GenX320RoiPixelMaskInterface::get_pixels() const {
    std::vector<std::pair<unsigned int, unsigned int>> pixel_list;

    GenX320RoiDriver::Grid grid = driver_->get_grid();
    auto grid_size              = grid.get_size();
    auto x_max                  = std::get<0>(grid_size);
    auto y_max                  = std::get<1>(grid_size);

    for (unsigned int y = 0; y < y_max; y++) {
        for (unsigned int x = 0; x < x_max; x++) {
            uint32_t vector = grid.get_vector(x, y);
            for (unsigned int i = 0; i < 32; i++) {
                auto valid = (1 << i) & vector;
                if (!valid) {
                    pixel_list.push_back(std::make_pair(x * 32 + i, y));
                }
            }
        }
    }

    return pixel_list;
}

void GenX320RoiPixelMaskInterface::apply_pixels() {
    driver_->set_driver_mode(GenX320RoiDriver::DriverMode::LATCH);
    driver_->enable(true);
}

void GenX320RoiPixelMaskInterface::reset_pixels() {
    GenX320RoiDriver::Grid default_grid(10, 320);
    driver_->set_grid(default_grid);
    driver_->reset_to_full_roi();
}

} // namespace Metavision