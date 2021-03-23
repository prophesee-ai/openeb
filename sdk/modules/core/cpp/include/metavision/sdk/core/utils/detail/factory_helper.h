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

#ifndef METAVISION_SDK_CORE_DETAIL_FACTORY_HELPER_H
#define METAVISION_SDK_CORE_DETAIL_FACTORY_HELPER_H

#include <type_traits>
#include <typeindex>
#include <utility>

namespace Metavision {
namespace detail {
/// @brief Helpers for using the @ref Factory class
namespace factory_helper {

/// @brief Returns type index of class
template<class Object, typename = typename std::enable_if<std::is_class<Object>::value>::type>
inline std::type_index index() {
    return std::type_index(typeid(Object));
}

/// @brief Functor that initializes Derived class and casts it to Base
///
/// Usage:
/// ~~~cpp
///     factory.register_object(0, Metavision::detail::factory_helper::base_class_constructor<Base, Derived>());
///     // Instantiates object of class Derived and returns a pointer to it as Base *
///     factory.create_object(0)
/// ~~~
template<class Base, class Derived>
struct base_class_constructor {
    template<class... Arguments>
    inline Base *operator()(Arguments &&...arguments) const {
        return new Derived(std::forward<Arguments>(arguments)...);
    }
};

} // namespace factory_helper
} // namespace detail
} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_FACTORY_HELPER_H
