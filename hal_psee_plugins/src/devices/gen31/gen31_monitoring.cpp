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

#include "devices/gen31/gen31_monitoring.h"
#include "metavision/hal/facilities/i_hw_register.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"

namespace Metavision {

Gen31Monitoring::Gen31Monitoring(const std::shared_ptr<I_HW_Register> &i_hw_register) : PseeMonitoring(i_hw_register) {
    get_hw_register()->write_register("SENSOR_IF/GEN31/lifo_ctrl", "lifo_en", true);
}

int Gen31Monitoring::get_temperature() {
    auto r = get_hw_register()->read_register("SYSTEM_MONITOR/TEMP_VCC_MONITOR/EVK_EXT_TEMP_VALUE");
    if (r != decltype(r)(-1))
        return r / 4096;
    return -1;
}

int Gen31Monitoring::get_illumination() {
    auto hw_register = get_hw_register();
    hw_register->write_register("SENSOR_IF/GEN31/lifo_ctrl", 0);
    hw_register->write_register("SENSOR_IF/GEN31/lifo_ctrl", "lifo_en", true);
    hw_register->write_register("SENSOR_IF/GEN31/lifo_ctrl", "lifo_cnt_en", true);
    bool valid       = false;
    uint16_t retries = 0;
    uint32_t counter = 0;
    while (valid == false && retries < 10) {
        auto reg_val = hw_register->read_register("SENSOR_IF/GEN31/lifo_ctrl");
        reg_val      = hw_register->read_register("SENSOR_IF/GEN31/lifo_ctrl");
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
