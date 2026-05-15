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

#ifndef METAVISION_SDK_BASE_DETAIL_DEPRECATED_FEATURE_H
#define METAVISION_SDK_BASE_DETAIL_DEPRECATED_FEATURE_H

#if defined(__GNUC__) || defined(__clang__)
#define METAVISION_DEPRECATED_FEATURE(version)                                     \
    __attribute__((deprecated("This feature is deprecated since version " #version \
                              ", and it will be removed in later releases.")))
#elif defined(_MSC_VER)
#define METAVISION_DEPRECATED_FEATURE(version)                                                                      \
    __declspec(deprecated("This feature is deprecated since version " #version ", and it will be removed in later " \
                          "releases."))
#else
// The deprecated feature of this compiler is not yet supported
#define METAVISION_DEPRECATED_FEATURE(version)
#endif

#endif // METAVISION_SDK_BASE_DETAIL_DEPRECATED_FEATURE_H
