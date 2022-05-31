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

#include "devices/gen31/gen31_evk2_interface_control.h"
#include "utils/register_map.h"

namespace Metavision {
namespace {
std::string REGBANK_PREFIX = "PSEE/CCAM5_IF/CONTROL/";
} // namespace

Gen31Evk2InterfaceControl::Gen31Evk2InterfaceControl(const std::shared_ptr<RegisterMap> &regmap) :
    register_map_(regmap) {}

void Gen31Evk2InterfaceControl::enable(bool enable) {
    // Enable bridge if
    (*register_map_)[REGBANK_PREFIX + "CONTROL"]["ENABLE"].write_value(enable);
}

void Gen31Evk2InterfaceControl::bypass(bool bypass) {
    // Bypass bridge if
    (*register_map_)[REGBANK_PREFIX + "CONTROL"]["BYPASS"].write_value(bypass);
}

void Gen31Evk2InterfaceControl::gen_last(bool enable) {
    (*register_map_)[REGBANK_PREFIX + "CONTROL"]["GEN_LAST"].write_value(enable);
}

} // namespace Metavision
