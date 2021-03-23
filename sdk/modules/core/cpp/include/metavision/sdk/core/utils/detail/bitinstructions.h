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

#ifndef METAVISION_SDK_CORE_DETAIL_BITINSTRUCTIONS_H
#define METAVISION_SDK_CORE_DETAIL_BITINSTRUCTIONS_H
#ifdef _MSC_VER
// For Visual Studio, we need the following code.
#include <cstdint>
#include <intrin.h>

uint32_t __inline ctz(const uint32_t value) {
    unsigned long trailing_zero = 0;
    if (_BitScanForward(&trailing_zero, value)) {
        return (uint32_t)trailing_zero;
    } else {
        return 32;
    }
}

uint32_t __inline clz(const uint32_t value) {
    unsigned long leading_zero = 0;
    if (_BitScanReverse(&leading_zero, value)) {
        return (uint32_t)(31 - leading_zero);
    } else {
        return 32;
    }
}
#else
// For GCC/CLANG, we can use the builtin intrinsics.
#define clz(V) __builtin_clz((V))
#define ctz(V) __builtin_ctz((V))
#endif
#endif // METAVISION_SDK_CORE_DETAIL_BITINSTRUCTIONS_H
