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

#ifndef METAVISION_HAL_DETAIL_TYPE_CHECK_H
#define METAVISION_HAL_DETAIL_TYPE_CHECK_H

#include <type_traits>

namespace Metavision{
namespace detail{

template <typename T, template <typename...> class C, typename ... Ts>
constexpr auto is_in_type_list (C<Ts...> const &) -> std::disjunction<std::is_same<T, Ts>...>;

template <typename T, typename V>
static constexpr bool is_in_type_list_v  = decltype(is_in_type_list<T>(std::declval<V>()))::value;

}}

#endif //METAVISION_HAL_DETAIL_TYPE_CHECK_H