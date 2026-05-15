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

#include "metavision/hal/utils/hal_exception.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_digital_crop.h"

using namespace Metavision;

GenX320DigitalCrop::GenX320DigitalCrop(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix) :
    enable_((*regmap)[prefix + "ro/oor_ctrl"]["oor_crop_enable"]),
    drop_((*regmap)[prefix + "ro/oor_ctrl"]["oor_rm_td"]),
    reset_orig_((*regmap)[prefix + "ro/oor_ctrl"]["oor_crop_reset_orig"]),
    start_x_((*regmap)[prefix + "ro/oor_start_pos"]["oor_crop_start_x"]),
    start_y_((*regmap)[prefix + "ro/oor_start_pos"]["oor_crop_start_y"]),
    end_x_((*regmap)[prefix + "ro/oor_end_pos"]["oor_crop_end_x"]),
    end_y_((*regmap)[prefix + "ro/oor_end_pos"]["oor_crop_end_y"]) {
    drop_.write_value(1);
}

bool GenX320DigitalCrop::enable(bool state) {
    enable_.write_value(state);
    return true;
}

bool GenX320DigitalCrop::is_enabled() const {
    return enable_.read_value();
}

bool GenX320DigitalCrop::set_window_region(const Region &region, bool reset_origin) {
    uint32_t start_x, start_y, end_x, end_y;
    std::tie(start_x, start_y, end_x, end_y) = region;

    if (start_x > end_x) {
        throw HalException(HalErrorCode::InvalidArgument,
                           "X coordinate of the region end pixel can't be smaller than the X start pixel");
    }
    if (start_y > end_y) {
        throw HalException(HalErrorCode::InvalidArgument,
                           "Y coordinate of the region end pixel can't be smaller than the Y start pixel");
    }

    start_x_.write_value(start_x);
    start_y_.write_value(start_y);
    end_x_.write_value(end_x);
    end_y_.write_value(end_y);

    reset_orig_.write_value(reset_origin);

    return true;
}

GenX320DigitalCrop::Region GenX320DigitalCrop::get_window_region() const {
    return {start_x_.read_value(), start_y_.read_value(), end_x_.read_value(), end_y_.read_value()};
}
