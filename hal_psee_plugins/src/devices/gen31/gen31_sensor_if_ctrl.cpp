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

#include "devices/gen31/gen31_sensor_if_ctrl.h"
#include "utils/register_map.h"

namespace Metavision {
using vfield = std::map<std::string, uint32_t>;

Gen31SensorIfCtrl::Gen31SensorIfCtrl(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix) :
    prefix_(prefix), register_map_(register_map) {}

void Gen31SensorIfCtrl::enable_test_pattern(uint32_t n_period = 0x0C00, uint32_t n_valid_ratio = 10,
                                            uint32_t p_period = 0x1400, uint32_t p_valid_ratio = 25,
                                            bool enable = true) {
    (*register_map_)[prefix_ + "TEST_PATTERN_N_PERIOD"].write_value(
        vfield{{"LENGTH", n_period}, {"VALID_RATIO", n_valid_ratio}});
    (*register_map_)[prefix_ + "TEST_PATTERN_P_PERIOD"].write_value(
        vfield{{"LENGTH", p_period}, {"VALID_RATIO", p_valid_ratio}});

    (*register_map_)[prefix_ + "TEST_PATTERN_CONTROL"]["EBABLE"] = enable;
}

void Gen31SensorIfCtrl::sensor_turn_on_clock() {
    (*register_map_)[prefix_ + "CONTROL"]["SENSOR_CLK_EN"] = true;
}

void Gen31SensorIfCtrl::sensor_turn_off_clock() {
    (*register_map_)[prefix_ + "CONTROL"]["SENSOR_CLK_EN"] = false;
}

void Gen31SensorIfCtrl::self_pattern_config(uint32_t ratio = 42) {
    (*register_map_)[prefix_ + "TEST_PATTERN_CONTROL"]["ENABLE"] = false;
    (*register_map_)[prefix_ + "TEST_PATTERN_N_PERIOD"].write_value(vfield{{"LENGTH", 0x0C00}, {"VALID_RATIO", ratio}});
    (*register_map_)[prefix_ + "TEST_PATTERN_P_PERIOD"].write_value({{"LENGTH", 0x1400}, {"VALID_RATIO", ratio}});
}

void Gen31SensorIfCtrl::self_pattern_control(bool enable) {
    (*register_map_)[prefix_ + "TEST_PATTERN_CONTROL"]["ENABLE"] = enable;
}

void Gen31SensorIfCtrl::trigger_fwd_config(uint32_t channel_id = 6) {
    (*register_map_)[prefix_ + "TRIGGERS"]["TRIGGER_FWD_ID"] = channel_id;
}

void Gen31SensorIfCtrl::trigger_fwd_control(bool enable) {
    (*register_map_)[prefix_ + "TRIGGERS"]["TRIGGER_FWD_ENABLE"] = enable;
}
} // namespace Metavision
