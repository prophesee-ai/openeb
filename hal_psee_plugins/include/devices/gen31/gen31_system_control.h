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

#ifndef METAVISION_HAL_GEN31_SYSTEM_CONTROL_H
#define METAVISION_HAL_GEN31_SYSTEM_CONTROL_H

#include <string>
#include <memory>

namespace Metavision {

class RegisterMap;

class Gen31SystemControl {
public:
    Gen31SystemControl(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix);

    void sensor_atis_control_clear(void);
    void sensor_prepowerup(void);
    void sensor_prepowerdown(void);
    void sensor_roi_td_rstn(bool);
    void sensor_em_rstn(bool);
    void sensor_soft_reset(bool);
    void sensor_enable_vddc(bool);
    void sensor_enable_vddd(bool);
    void sensor_enable_vdda(bool);
    void sensor_powerdown(void);
    void sensor_powerup(void);
    void soft_reset(std::string reg_obj);
    void hvga_remap_control(bool enable);
    void no_blocking_control(bool enable);
    void host_if_control(bool enable);
    void timebase_control(bool enable);
    void timebase_config(bool ext_sync, bool master);

private:
    std::string prefix_;
    std::shared_ptr<RegisterMap> register_map_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN31_SYSTEM_CONTROL_H
