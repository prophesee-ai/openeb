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

#ifndef METAVISION_SDK_BASE_DETAIL_STRUCT_PACK_H
#define METAVISION_SDK_BASE_DETAIL_STRUCT_PACK_H

#ifdef SWIG
#define FORCE_PACK(__Declaration__) __Declaration__
#else
#ifdef _MSC_VER
#define FORCE_PACK(__Declaration__) __pragma(pack(push, 1)) __Declaration__ __pragma(pack(pop))
#elif defined(__GNUC__)
#define FORCE_PACK(__Declaration__) __Declaration__ __attribute__((__packed__))
#endif
#endif

#define PACK(__Declaration__)                                                                      \
    static_assert(false, "\n\n"                                                                    \
                         "The use of the 'PACK' macro is deprecated and probably not necessary.\n" \
                         "For packing data with strict bitfield, use the following model:\n\n"     \
                         "struct FOO {\n"                                                         \
                         "\ttype1 data1 : bit_width1\n"                                            \
                         "\ttype2 data2 : bit_width2\n"                                            \
                         "\ttype3 data3 : bit_width3\n"                                            \
                         "\t...\n"                                                                 \
                         "};\n"                                                                    \
                         "static_assert(sizeof(FOO) == ExpectedSizeBytes, \"Error message\");\n\n" \
                         "With sizeof(type1) == sizeof(type2) == sizeof(type3) ...\n"              \
                         "Use the FORCE_PACK macro if you really know what you are doing.\n")

#endif // METAVISION_SDK_BASE_DETAIL_STRUCT_PACK_H
