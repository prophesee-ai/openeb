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

#include <vector>
#include <tuple>

#include "devices/gen41/gen41_evk3_regmap_builder.h"
#include "devices/gen41/register_maps/gen41_evk3_registermap.h"
#include "utils/regmap_data.h"
#include "utils/register_map.h"

namespace Metavision {

void build_gen41_evk3_register_map(RegisterMap &regmap) {
    std::vector<std::tuple<RegmapData *, int, std::string, int>> Gen41Evk3RegisterMap_init = {
        std::make_tuple(Gen41Evk3RegisterMap, Gen41Evk3RegisterMapSize, "PSEE", 0),
    };

    init_device_regmap(regmap, Gen41Evk3RegisterMap_init);
    regmap.dump();
}

} // namespace Metavision
