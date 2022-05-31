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

#include <math.h>
#include <map>
#include <sstream>
#include <iostream>

#include "devices/common/ccam_monitoring.h"
#include "metavision/hal/facilities/i_hw_register.h"

namespace Metavision {

CCamMonitoring::CCamMonitoring(const std::shared_ptr<I_HW_Register> &i_hw_register, const std::string &fpga_prefix,
                               const std::string &sensor_prefix) :
    PseeMonitoring(i_hw_register), fpga_prefix_(fpga_prefix), sensor_prefix_(sensor_prefix) {
    // Enable temperature reading
    get_hw_register()->write_register(fpga_prefix_ + "SYSTEM_MONITOR/TEMP_VCC_MONITOR/EXT_TEMP_CONTROL",
                                      "EXT_TEMP_MONITOR_SPI_EN", 1);
}

int CCamMonitoring::get_temperature() {
    auto r = get_hw_register()->read_register(fpga_prefix_ + "SYSTEM_MONITOR/TEMP_VCC_MONITOR/EVK_EXT_TEMP_VALUE");
    if (r != decltype(r)(-1))
        return r / 4096;
    return -1;
}

int CCamMonitoring::get_illumination() {
    auto hw_register = get_hw_register();
    hw_register->write_register(sensor_prefix_ + "lifo_ctrl", "lifo_en", 0);
    hw_register->write_register(sensor_prefix_ + "lifo_ctrl", "lifo_cnt_en", 0);
    hw_register->write_register(sensor_prefix_ + "lifo_ctrl", "lifo_en", true);
    hw_register->write_register(sensor_prefix_ + "lifo_ctrl", "lifo_cnt_en", true);
    bool valid       = false;
    uint16_t retries = 0;
    uint32_t counter = 0;
    while (valid == false && retries < 10) {
        auto reg_val = hw_register->read_register(sensor_prefix_ + "lifo_ctrl");
        valid        = reg_val & 1 << 29;
        counter      = reg_val & ((1 << 27) - 1);
        retries += 1;
    }

    if (!valid) {
        return -1;
    }

    if (counter != decltype(counter)(-1)) {
        float t = float(counter) / 100.;
        return powf(10, 3.5 - logf(t * 0.37) / logf(10));
    }
    return -1;
}

} // namespace Metavision
