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

#ifndef METAVISION_HAL_GEN31_CCAM5_FPGA_H
#define METAVISION_HAL_GEN31_CCAM5_FPGA_H

#include <string>
#include <memory>

#include "devices/gen31/gen31_fpga.h"
#include "devices/gen31/gen31_system_control.h"
#include "devices/gen31/gen31_sensor_if_ctrl.h"

namespace Metavision {

class RegisterMap;

class Gen31CCam5Fpga : public Gen31Fpga {
public:
    Gen31CCam5Fpga(const std::shared_ptr<RegisterMap> &regmap, const std::string &root_prefix,
                   const std::string &sensor_if_prefix);

    virtual void init() override;
    virtual void start() override;
    virtual void stop() override;
    virtual void destroy() override;

    void set_timebase_master(bool enable);
    void set_timebase_ext_sync(bool enable);

    bool get_timebase_master() const;
    bool get_timebase_ext_sync() const;

    std::string get_root_prefix() const;
    std::string get_sensor_if_prefix() const;
    std::string get_system_config_prefix() const;
    std::string get_system_control_prefix() const;
    std::string get_system_monitor_prefix() const;

private:
    const std::string root_prefix_;
    const std::string sensor_if_prefix_;

    bool timebase_master_   = true;
    bool timebase_ext_sync_ = false;

    std::shared_ptr<RegisterMap> register_map_;
    Gen31SystemControl sys_ctrl_;
    Gen31SensorIfCtrl sensor_if_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN31_CCAM5_FPGA_H
