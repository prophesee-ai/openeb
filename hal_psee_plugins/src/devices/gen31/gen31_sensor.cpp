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

#include <stdint.h>

#include "devices/gen31/gen31_sensor.h"
#include "metavision/hal/utils/hal_log.h"
#include "utils/register_map.h"

namespace Metavision {

using vfield = std::map<std::string, uint32_t>;

Gen31Sensor::Gen31Sensor(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix, bool is_em) :
    register_map_(regmap),
    prefix_(prefix),
    analog_(register_map_, prefix, is_em),
    digital_(register_map_, prefix),
    is_em_(is_em) {}

long long Gen31Sensor::get_chip_id() {
    return digital_.get_chip_id();
}

void Gen31Sensor::init() {
    MV_HAL_LOG_TRACE() << "Sensor Gen31 Init";
    digital_.init();
    bgen_init();
    analog_.init();
}

void Gen31Sensor::start() {
    digital_.start();
    // roi
    (*register_map_)[prefix_ + "roi_ctrl"]["roi_td_en"]             = true;
    (*register_map_)[prefix_ + "roi_ctrl"]["roi_td_shadow_trigger"] = true;
    if (is_em_)
        (*register_map_)[prefix_ + "roi_ctrl"]["roi_em_en"] = true;
    analog_.start();
}

void Gen31Sensor::stop() {
    MV_HAL_LOG_TRACE() << "Sensor Gen31 Stop";
    analog_.stop();
    digital_.stop();
}

void Gen31Sensor::destroy() {
    MV_HAL_LOG_TRACE() << "Sensor Gen31 Destroy";
    analog_.destroy();
    digital_.destroy();
}

void Gen31Sensor::bgen_init() {
    (*register_map_)[prefix_ + "bgen_00"] = 0x5900009f;
    (*register_map_)[prefix_ + "bgen_01"] = 0x5900009f;
    (*register_map_)[prefix_ + "bgen_02"] = 0x5900009b;
    (*register_map_)[prefix_ + "bgen_03"] = 0x590000a9;
    (*register_map_)[prefix_ + "bgen_04"] = 0x7900008c;
    (*register_map_)[prefix_ + "bgen_05"] = 0x79000070;
    (*register_map_)[prefix_ + "bgen_06"] = 0x7900008c;
    (*register_map_)[prefix_ + "bgen_07"] = 0x790000c0;
    (*register_map_)[prefix_ + "bgen_08"] = 0x7900003e;
    (*register_map_)[prefix_ + "bgen_09"] = 0x79000036;
    (*register_map_)[prefix_ + "bgen_10"] = 0x590000b7;
    (*register_map_)[prefix_ + "bgen_11"] = 0x79000000;
    (*register_map_)[prefix_ + "bgen_12"] = 0x790000f7;
    (*register_map_)[prefix_ + "bgen_13"] = 0x7102c400;
    (*register_map_)[prefix_ + "bgen_14"] = 0x7107ff00;
    (*register_map_)[prefix_ + "bgen_15"] = 0x71008200;
    (*register_map_)[prefix_ + "bgen_16"] = 0x790000e2;
    (*register_map_)[prefix_ + "bgen_17"] = 0x790000f0;
    (*register_map_)[prefix_ + "bgen_18"] = 0x7900008c;
    (*register_map_)[prefix_ + "bgen_19"] = 0x7900001f;
    (*register_map_)[prefix_ + "bgen_20"] = 0x79000033;
    (*register_map_)[prefix_ + "bgen_21"] = 0x79000029;
    (*register_map_)[prefix_ + "bgen_22"] = 0x71003600;
    (*register_map_)[prefix_ + "bgen_23"] = 0x51034f00;
    (*register_map_)[prefix_ + "bgen_24"] = 0x51004f00;
    (*register_map_)[prefix_ + "bgen_25"] = 0x51000100;
    (*register_map_)[prefix_ + "bgen_26"] = 0x61013100;
}

} // namespace Metavision
