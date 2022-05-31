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

#include "utils/regmap_data.h"

namespace Metavision {
void init_device_regmap(RegisterMap &regmap,
                        std::vector<std::tuple<RegmapData *, int, std::string, int>> &device_regmap_description) {
    bool is_curreg_valid   = false;
    bool is_curfield_valid = false;
    RegisterMap::Register curreg;
    RegisterMap::Field curfield;
    for (auto sub_desc : device_regmap_description) {
        RegmapData *curdata = std::get<0>(sub_desc);
        size_t size         = std::get<1>(sub_desc);
        std::string sub_name(std::get<2>(sub_desc));
        RegmapData *data_end = curdata + size;
        std::string prefix   = "";

        if (sub_name.length() != 0) {
            prefix = sub_name + "/";
        }

        for (; curdata != data_end; ++curdata) {
            if (curdata->type == R) {
                if (is_curfield_valid) {
                    curreg.add_field(curfield);
                }
                if (is_curreg_valid) {
                    regmap.add_register(curreg);
                }
                curreg            = RegisterMap::Register(prefix + curdata->register_data.name,
                                               curdata->register_data.addr + std::get<3>(sub_desc));
                is_curreg_valid   = true;
                is_curfield_valid = false;
            } else if (curdata->type == F) {
                if (is_curfield_valid) {
                    curreg.add_field(curfield);
                }
                is_curfield_valid = true;
                curfield          = RegisterMap::Field(curdata->field_data.name, curdata->field_data.start,
                                              curdata->field_data.len, curdata->field_data.default_value);
            } else if (curdata->type == A) {
                if (is_curfield_valid) {
                    curfield.add_alias(curdata->alias_data.name, curdata->alias_data.value);
                }
            }
        }
        if (is_curfield_valid) {
            curreg.add_field(curfield);
        }
        if (is_curreg_valid) {
            regmap.add_register(curreg);
        }
    }
}
} // namespace Metavision
