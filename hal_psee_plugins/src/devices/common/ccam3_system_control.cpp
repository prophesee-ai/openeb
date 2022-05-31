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

#include <map>

#include "devices/common/ccam3_system_control.h"
#include "utils/register_map.h"

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {

CCam3SystemControl::CCam3SystemControl(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix) :
    SystemControl(regmap, prefix), prefix_(prefix), register_map_(regmap) {}

void CCam3SystemControl::imu_control(bool enable) {
    /*Control the IMU i/f logic.

    Args:
        enable (int): IMU state (0 or 1)
    */
    (*register_map_)[prefix_ + "IMU_CONTROL"]["ENABLE"].write_value(enable);
}

} // namespace Metavision
