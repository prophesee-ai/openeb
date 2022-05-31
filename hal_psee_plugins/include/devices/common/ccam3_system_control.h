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

#ifndef METAVISION_HAL_CCAM3_SYSTEM_CONTROL_H
#define METAVISION_HAL_CCAM3_SYSTEM_CONTROL_H

#include "devices/common/system_control.h"

namespace Metavision {

class CCam3SystemControl : public SystemControl {
public:
    CCam3SystemControl(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix);
    void imu_control(bool enable);

private:
    std::string prefix_;
    std::shared_ptr<RegisterMap> register_map_;
};

} // namespace Metavision

#endif // METAVISION_HAL_CCAM3_SYSTEM_CONTROL_H
