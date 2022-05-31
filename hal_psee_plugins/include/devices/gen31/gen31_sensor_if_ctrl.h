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

#ifndef METAVISION_HAL_GEN31_SENSOR_IF_CONTROL_H
#define METAVISION_HAL_GEN31_SENSOR_IF_CONTROL_H

#include <string>
#include <memory>

namespace Metavision {

class RegisterMap;

class Gen31SensorIfCtrl {
public:
    Gen31SensorIfCtrl(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix);
    void enable_test_pattern(uint32_t n_period, uint32_t n_valid_ratio, uint32_t p_period, uint32_t p_valid_ratio,
                             bool enable);
    void sensor_turn_on_clock();
    void sensor_turn_off_clock();
    void self_pattern_config(uint32_t ratio);
    void self_pattern_control(bool enable);
    void trigger_fwd_config(uint32_t channel_id);
    void trigger_fwd_control(bool enable);

private:
    std::string prefix_;
    std::shared_ptr<RegisterMap> register_map_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN31_SENSOR_IF_CONTROL_H
