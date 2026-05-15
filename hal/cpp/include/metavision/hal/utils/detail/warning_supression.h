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

#ifndef METAVISION_HAL_DETAIL_WARNING_SUPRESSION_H
#define METAVISION_HAL_DETAIL_WARNING_SUPRESSION_H

#if defined(__GNUC__)

#define SUPRESS_DEPRECATION_WARNING(x)                                                             \
    _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"") \
        x _Pragma("GCC diagnostic pop")

#elif defined(__clang__)

#define SUPRESS_DEPRECATION_WARNING(x)                                                                 \
    _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"") \
        x _Pragma("clang diagnostic pop")

#elif defined(_MSC_VER)

// see reference to:
// https://learn.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warnings-c4200-through-c4399?view=msvc-170
#define SUPRESS_DEPRECATION_WARNING(x) \
    __pragma(warning(push)) __pragma(warning(disable : 4973 4974 4995 4996)) x __pragma(warning(pop))

#else

#define SUPRESS_DEPRECATION_WARNING(x) x

#endif

#endif // METAVISION_HAL_DETAIL_WARNING_SUPRESSION_H