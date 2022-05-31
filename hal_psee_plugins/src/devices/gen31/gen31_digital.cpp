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

#include "devices/gen31/gen31_digital.h"
#include "utils/register_map.h"

namespace Metavision {
using vfield = std::map<std::string, uint32_t>;

Gen31Digital::Gen31Digital(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix) :
    prefix_(prefix), register_map_(register_map) {}

long long Gen31Digital::get_chip_id() {
    return (*register_map_)[prefix_ + "chip_id"].read_value();
}

void Gen31Digital::init() {
    (*register_map_)[prefix_ + "clk_out_ctrl"].write_value(vfield{
        {"clk_out_en", true},
        {"clk_gate_bypass", false},
    });
}

void Gen31Digital::start() {}

void Gen31Digital::stop() {}

void Gen31Digital::destroy() {
    (*register_map_)[prefix_ + "clk_out_ctrl"].write_value(vfield{
        {"clk_out_en", false},
    });
}
} // namespace Metavision
