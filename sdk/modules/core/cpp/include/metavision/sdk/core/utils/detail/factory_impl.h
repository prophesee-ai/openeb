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

#ifndef METAVISION_SDK_CORE_DETAIL_FACTORY_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_FACTORY_IMPL_H

#include <map>
#include <iostream>
#include <type_traits>
#include <typeindex>
#include <functional>
#include <memory>
#include <utility>

#include "metavision/sdk/base/utils/sdk_log.h"

namespace Metavision {
namespace detail {

template<class AbstractProduct, typename ProductKey, typename ProductConstructor>
bool Factory<AbstractProduct, ProductKey, ProductConstructor>::register_object(const ProductKey &key,
                                                                               ProductConstructor constructor) {
    const auto iter      = associations_.find(key);
    const auto not_found = (iter == associations_.end());
    if (not_found) {
        associations_.insert(std::make_pair(key, constructor));
    }
    return not_found;
}

template<class AbstractProduct, typename ProductKey, typename ProductConstructor>
bool Factory<AbstractProduct, ProductKey, ProductConstructor>::unregister_object(const ProductKey &key) {
    const auto iter  = associations_.find(key);
    const auto found = (iter != associations_.end());
    if (found) {
        associations_.erase(iter);
    }
    return found;
}

template<class AbstractProduct, typename ProductKey, typename ProductConstructor>
template<typename... Args>
std::unique_ptr<AbstractProduct>
    Factory<AbstractProduct, ProductKey, ProductConstructor>::create_object(const ProductKey &key,
                                                                            Args &&...args) const {
    const auto iter  = associations_.find(key);
    const auto found = (iter != associations_.end());
    auto ptr         = found ? (iter->second)(std::forward<Args>(args)...) : on_unknown_object(key);
    return std::unique_ptr<AbstractProduct>(ptr);
}

template<class AbstractProduct, typename ProductKey, typename ProductConstructor>
AbstractProduct *Factory<AbstractProduct, ProductKey, ProductConstructor>::on_unknown_object(const ProductKey &) const {
    MV_SDK_LOG_WARNING() << "Unknown object type passed to Factory";
    return nullptr;
}

} // namespace detail
} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_FACTORY_IMPL_H
