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

#ifndef METAVISION_SDK_CORE_DETAIL_ITERATOR_TRAITS_H
#define METAVISION_SDK_CORE_DETAIL_ITERATOR_TRAITS_H

#include <iterator>
#include <type_traits>

namespace Metavision {

namespace detail {

template<typename T, typename R>
constexpr bool is_same_v = std::is_same<T, R>::value;

template<typename Iterator>
using category_t = typename std::iterator_traits<Iterator>::iterator_category;

template<typename Iterator>
using difference_t = typename std::iterator_traits<Iterator>::difference_type;

template<typename Iterator>
using value_t = typename std::iterator_traits<Iterator>::value_type;

template<typename Iterator>
using reference_t = typename std::iterator_traits<Iterator>::reference;

template<typename Iterator>
using pointer_t = typename std::iterator_traits<Iterator>::pointer;

template<class, class Enable = void>
struct is_back_inserter_iterator : std::false_type {};

template<class Iterator>
struct is_back_inserter_iterator<
    Iterator, std::enable_if_t<is_same_v<category_t<Iterator>, std::output_iterator_tag> ||
                               is_same_v<value_t<Iterator>, void> || is_same_v<difference_t<Iterator>, void> ||
                               is_same_v<reference_t<Iterator>, void> || is_same_v<pointer_t<Iterator>, void>>>
    : std::true_type {};

template<typename Iterator>
constexpr auto is_back_inserter_iterator_v = is_back_inserter_iterator<Iterator>::value;

template<class Iterator, bool IsBackInserter>
struct iterator_traits_implementation;

template<class Iterator>
struct iterator_traits_implementation<Iterator, false> {
    using value_type        = value_t<Iterator>;
    using difference_type   = difference_t<Iterator>;
    using pointer           = pointer_t<Iterator>;
    using reference         = reference_t<Iterator>;
    using iterator_category = category_t<Iterator>;
};

template<class Iterator>
struct iterator_traits_implementation<Iterator, true> {
    using container_type    = typename Iterator::container_type;
    using value_type        = typename container_type::value_type;
    using difference_type   = typename container_type::difference_type;
    using pointer           = typename container_type::pointer;
    using reference         = typename container_type::reference;
    using iterator_category = detail::category_t<Iterator>;
};

} // namespace detail

template<class Iterator>
using iterator_traits = detail::iterator_traits_implementation<Iterator, detail::is_back_inserter_iterator_v<Iterator>>;

template<typename Iterator, typename TargetValueType>
struct is_iterator_over {
    using iterator_underlying_type =
        typename std::remove_pointer<typename std::iterator_traits<Iterator>::pointer>::type;
    static constexpr bool value = std::is_base_of<TargetValueType, iterator_underlying_type>::value;
};

template<typename Iterator, typename TargetValueType>
struct is_const_iterator_over {
    using iterator_underlying_type =
        typename std::remove_pointer<typename std::iterator_traits<Iterator>::pointer>::type;
    static constexpr bool value =
        is_iterator_over<Iterator, TargetValueType>::value && std::is_const<iterator_underlying_type>::value;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_ITERATOR_TRAITS_H
