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

#ifndef METAVISION_HAL_REGMAP_DATA_H
#define METAVISION_HAL_REGMAP_DATA_H

#include <stdint.h>
#include <vector>

enum TypeRegmapElement { R, F, A }; // Register, Field, Alias

struct RegmapElement {
    TypeRegmapElement type;
    struct FieldData {
        const char *name;
        uint32_t start;
        uint32_t len;
        uint32_t default_value;
    };
    struct RegisterData {
        const char *name;
        uint32_t addr;
    };
    struct AliasData {
        const char *name;
        uint32_t value;
    };
    union {
        FieldData field_data;
        RegisterData register_data;
        AliasData alias_data;
    };
};

#endif // METAVISION_HAL_REGMAP_DATA_H
