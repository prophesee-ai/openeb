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

#ifndef METAVISION_HAL_SYSTEM_CONTROL_H
#define METAVISION_HAL_SYSTEM_CONTROL_H

#include <string>
#include <memory>

namespace Metavision {

class RegisterMap;

class SystemControl {
public:
    SystemControl(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix);
    ~SystemControl();
    bool apply_resets();
    void set_evt_format(uint32_t fmt);
    void clk_control(bool enable);
    void host_if_control(bool enable);
    void time_base_config(bool ext_sync, bool master, bool master_sel, bool fwd_up, bool fwd_down);
    void time_base_control(bool enable);
    void merge_config(bool bypass, int source);
    void merge_control(bool enable);
    void th_recovery_config(bool bypass);
    void th_recovery_control(bool enable);
    void data_formatter_config(bool bypass);
    void data_formatter_control(bool enable);
    void set_mode(int mode);
    void sync_out_pin_control(bool trig_out_override);
    void oob_filter_control(bool enable);
    void oob_filter_origin(int x, int y);
    void oob_filter_size(int width, int height);

private:
    std::string prefix_;
    std::shared_ptr<RegisterMap> register_map_;
};

} // namespace Metavision

#endif // METAVISION_HAL_SYSTEM_CONTROL_H
