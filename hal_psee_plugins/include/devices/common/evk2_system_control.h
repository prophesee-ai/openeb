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

#ifndef METAVISION_HAL_EVK2_SYSTEM_CONTROL_H
#define METAVISION_HAL_EVK2_SYSTEM_CONTROL_H

#include <string>
#include <memory>

namespace Metavision {

class RegisterMap;

class Evk2SystemControl {
public:
    Evk2SystemControl(const std::shared_ptr<RegisterMap> &regmap);

    bool apply_resets();
    void set_evt_format(uint32_t fmt);
    void clk_control(bool enable);
    void time_base_config(bool ext_sync, bool master, bool master_sel, bool fwd_up, bool fwd_down);
    void time_base_control(bool enable);
    void merge_config(bool bypass, int source);
    void merge_control(bool enable);
    void th_recovery_config(bool bypass);
    void th_recovery_control(bool enable);
    void out_th_recovery_config(bool bypass);
    void out_th_recovery_control(bool enable);
    void data_formatter_config(bool bypass);
    void data_formatter_control(bool enable);
    void set_mode(int mode);
    void monitoring_merge_config(bool bypass, int source);
    void monitoring_merge_control(bool enable);
    void ts_checker_config(bool bypass);
    void sync_out_pin_config(bool trig_out_override);
    bool sync_out_pin_control(bool enable);
    bool get_sync_out_pin_fault_alert();
    bool is_trigger_out_enabled();

private:
    std::shared_ptr<RegisterMap> register_map_;
    std::string sys_ctrl_regbank_;
    std::string sys_mon_regbank_;
    std::string ps_host_if_regbank_;
};

} // namespace Metavision

#endif // METAVISION_HAL_EVK2_SYSTEM_CONTROL_H
