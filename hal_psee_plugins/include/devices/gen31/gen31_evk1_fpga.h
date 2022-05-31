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

#ifndef METAVISION_HAL_GEN31_EVK1_FPGA_H
#define METAVISION_HAL_GEN31_EVK1_FPGA_H

#include "devices/gen31/gen31_fpga.h"
#include "devices/gen31/gen31_system_control.h"
#include "devices/gen31/gen31_sensor_if_ctrl.h"

namespace Metavision {

class RegisterMap;

class Gen31Evk1Fpga : public Gen31Fpga {
public:
    Gen31Evk1Fpga(const std::shared_ptr<RegisterMap> &regmap, bool is_em);

    virtual void init() override;
    virtual void start() override;
    virtual void stop() override;
    virtual void destroy() override;

protected:
    std::shared_ptr<RegisterMap> register_map_;
    Gen31SystemControl sys_ctrl_;
    Gen31SensorIfCtrl sensor_if_;
    bool is_em_ = false;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN31_EVK1_FPGA_H
