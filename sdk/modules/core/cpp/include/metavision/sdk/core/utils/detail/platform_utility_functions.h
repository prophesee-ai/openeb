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

#ifndef METAVISION_SDK_CORE_DETAIL_PLATFORM_UTILITY_FUNCTIONS_H
#define METAVISION_SDK_CORE_DETAIL_PLATFORM_UTILITY_FUNCTIONS_H

#include <boost/config.hpp>
#ifdef _MSC_VER
#include <Intrin.h>
#include <nmmintrin.h>
#endif

namespace Metavision {
namespace detail {
BOOST_FORCEINLINE void cross_platform_prefetch(const void *addr, ...) {
#ifdef _MSC_VER
    _m_prefetch(const_cast<void *>(addr));
#else
    __builtin_prefetch(addr);
#endif
}

BOOST_FORCEINLINE void cross_platform_prefetch(void *addr, ...) {
#ifdef _MSC_VER
    _m_prefetch(addr);
#else
    __builtin_prefetch(addr);
#endif
}

BOOST_FORCEINLINE int cross_platform_popcount(unsigned long long input_num) {
#ifdef _MSC_VER
    return static_cast<int>(_mm_popcnt_u64(input_num));
#else
    return __builtin_popcountll(input_num);
#endif
}

} // namespace detail
} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_PLATFORM_UTILITY_FUNCTIONS_H
