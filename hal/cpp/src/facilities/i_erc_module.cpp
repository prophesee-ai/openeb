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

#include <memory>

#include "metavision/hal/facilities/i_erc_module.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

bool I_ErcModule::set_cd_event_rate(uint32_t events_per_sec) {
    uint32_t count_period = get_count_period();
    return set_cd_event_count(static_cast<uint32_t>(static_cast<uint64_t>(events_per_sec) * count_period / 1000000));
}

uint32_t I_ErcModule::get_cd_event_rate() const {
    uint32_t count_period = get_count_period();
    uint32_t evt_count    = get_cd_event_count();
    return static_cast<uint32_t>(static_cast<uint64_t>(evt_count) * 1000000 / count_period);
}

uint32_t I_ErcModule::get_min_supported_cd_event_rate() const {
    uint32_t count_period = get_count_period();
    uint32_t evt_count    = get_min_supported_cd_event_count();
    return static_cast<uint32_t>(static_cast<uint64_t>(evt_count) * 1000000 / count_period);
}

uint32_t I_ErcModule::get_max_supported_cd_event_rate() const {
    uint32_t count_period = get_count_period();
    uint32_t evt_count    = get_max_supported_cd_event_count();
    return static_cast<uint32_t>(static_cast<uint64_t>(evt_count) * 1000000 / count_period);
}

} // namespace Metavision
