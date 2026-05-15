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

#ifndef METAVISION_SDK_CORE_DETAIL_FACTORY_H
#define METAVISION_SDK_CORE_DETAIL_FACTORY_H

#include <map>
#include <type_traits>
#include <typeindex>
#include <memory>
#include <functional>

namespace Metavision {
namespace detail {

/// @brief Factory for instantiating certain objects based on a key
template<class AbstractProduct, typename ProductKey = std::type_index,
         typename ProductCreator = std::function<AbstractProduct *(void)>>
class Factory {
    static_assert(std::is_convertible<AbstractProduct *, typename ProductCreator::result_type>::value,
                  "The given functor does not return the correct abstract project");

public:
    Factory()          = default;
    virtual ~Factory() = default;

    using value_type = AbstractProduct;
    using key_type   = ProductKey;
    using deleter    = std::default_delete<AbstractProduct>;
    using unique_ptr = std::unique_ptr<AbstractProduct, deleter>;

    /// @brief Call constructor of object associated to the key
    /// @param key Referring to the type of object to instantiate
    /// @param args Optional args to pass to the constructor
    template<typename... Args>
    std::unique_ptr<AbstractProduct> create_object(ProductKey const &key, Args &&...args) const;

    /// @brief Register association between key and constructor
    /// @param key Key to access this constructor
    /// @param constructor Constructor to register
    virtual bool register_object(ProductKey const &key, ProductCreator constructor);

    /// @brief Delete association between key and constructor
    /// @param key Key for which to delete the association
    virtual bool unregister_object(ProductKey const &key);

protected:
    virtual AbstractProduct *on_unknown_object(ProductKey const &) const;
    std::map<ProductKey, ProductCreator> associations_{};
};

} // namespace detail
} // namespace Metavision

#include "factory_impl.h"

#endif // METAVISION_SDK_CORE_DETAIL_FACTORY_H
