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

#include <tuple>

#include "devices/gen31/gen31_evk2_regmap_builder.h"
#include "utils/regmap_data.h"
#include "devices/gen31/register_maps/gen31_evk2_registermap.h"

namespace Metavision {

void build_gen31_evk2_register_map(RegisterMap &regmap) {
    std::vector<std::tuple<RegmapData *, int, std::string, int>> Gen31Evk2RegisterMap_init = {
        std::make_tuple(Gen31Evk2RegisterMap, Gen31Evk2RegisterMapSize, "PSEE", 0),
    };

    init_device_regmap(regmap, Gen31Evk2RegisterMap_init);
    regmap.dump();
}

} // namespace Metavision
