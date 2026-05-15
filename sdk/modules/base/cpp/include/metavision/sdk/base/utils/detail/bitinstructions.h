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

#include <type_traits>
#include <cstdint>

#ifdef _MSC_VER
#include <immintrin.h>
#include <intrin.h>
#endif // _MSC_VER

namespace Metavision {
namespace detail {

template<class T>
struct _32bType {
    static inline unsigned long ctz(T val) {
#ifdef _MSC_VER
#if (_MSC_VER < 1700) || defined(_M_ARM)
        unsigned long trailing_zero = 0;
        _BitScanForward(&trailing_zero, val);
        return trailing_zero;
#else
        return _tzcnt_u32(val);
#endif
#else
        return __builtin_ctz(val);
#endif
    }
    static inline unsigned long clz(T val) {
#ifdef _MSC_VER
#if (_MSC_VER < 1700) || defined(_M_ARM)
        unsigned long leading_zero = 0;
        _BitScanReverse(&leading_zero, val);
        return Metavision::detail::bit_size<T> - 1 - leading_zero;
#else
        return _lzcnt_u32(val);
#endif
#else
        return __builtin_clz(val);
#endif
    }
};

template<class T>
struct _64bType {
    static inline unsigned long ctz(T val) {
#ifdef _MSC_VER
#if (_MSC_VER < 1700) || defined(_M_ARM)
        unsigned long trailing_zero = 0;
        _BitScanForward64(&trailing_zero, val);
        return trailing_zero;
#else
        return _tzcnt_u64(val);
#endif
#else
        return __builtin_ctzll(val);
#endif
    }
    static inline unsigned long clz(T val) {
#ifdef _MSC_VER
#if (_MSC_VER < 1700) || defined(_M_ARM)
        unsigned long leading_zero = 0;
        _BitScanReverse64(&leading_zero, val);
        return Metavision::detail::bit_size<T> - 1 - leading_zero;
#else
        return _lzcnt_u64(val);
#endif
#else
        return __builtin_clzll(val);
#endif
    }
};

template<class T>
constexpr int bit_size = sizeof(T) * 8; /// Size in bits of type T

struct TypeNotSupported {}; /// Helper structure for unsupported implementation

// clang-format off
template<class T>
using TypeClass = 
    std::conditional_t<bit_size<T> == 32, _32bType<T>,
    std::conditional_t<bit_size<T> == 64, _64bType<T>, 
    TypeNotSupported>>;
// clang-format on

template<class T>
constexpr bool is_unsuported_type = std::is_same<TypeClass<T>, Metavision::detail::TypeNotSupported>::value;

} // namespace detail

template<class T>
inline uint32_t clz(T val) {
    static_assert(!Metavision::detail::is_unsuported_type<T>, "clz intrinsic only supports 32 and 64 integers");

    if (val == 0) {
        return Metavision::detail::bit_size<T>;
    }
    return Metavision::detail::TypeClass<T>::clz(val);
}

// careful, only call this function if you know that @p val is never 0
template<class T>
inline uint32_t clz_not_zero(T val) {
    static_assert(!Metavision::detail::is_unsuported_type<T>, "clz intrinsic only supports 32 and 64 integers");
    return Metavision::detail::TypeClass<T>::clz(val);
}

template<class T>
inline uint32_t ctz(T val) {
    static_assert(!Metavision::detail::is_unsuported_type<T>, "ctz intrinsic only supports 32 and 64 integers");

    if (val == 0) {
        return Metavision::detail::bit_size<T>;
    }
    return Metavision::detail::TypeClass<T>::ctz(val);
}

// careful, only call this function if you know that @p val is never 0
template<class T>
inline uint32_t ctz_not_zero(T val) {
    static_assert(!Metavision::detail::is_unsuported_type<T>, "ctz intrinsic only supports 32 and 64 integers");
    return Metavision::detail::TypeClass<T>::ctz(val);
}
} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_BITINSTRUCTIONS_H
