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

#include "devices/imx636/imx636_evk2_regmap_builder.h"
#include "devices/imx636/register_maps/imx636_evk2_registermap.h"
#include "utils/regmap_data.h"
#include "utils/register_map.h"

namespace Metavision {

void build_imx636_evk2_register_map(RegisterMap &regmap) {
    std::vector<std::tuple<RegmapData *, int, std::string, int>> Imx636Evk2RegisterMap_init = {
        std::make_tuple(Imx636Evk2RegisterMap, Imx636Evk2RegisterMapSize, "PSEE", 0),
    };

    init_device_regmap(regmap, Imx636Evk2RegisterMap_init);
    regmap.dump();
}

} // namespace Metavision
