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

#ifndef METAVISION_HAL_I_REGISTRABLE_FACILITY_IMPL_H
#define METAVISION_HAL_I_REGISTRABLE_FACILITY_IMPL_H

#include <functional>
#include <string>
#include <typeinfo>

namespace Metavision {

template<typename SelfType, typename BaseType>
std::unordered_set<std::size_t> I_RegistrableFacility<SelfType, BaseType>::registration_info() const {
    static_assert(std::is_base_of<I_RegistrableFacility<BaseType>, BaseType>::value,
                  "BaseType of registrable facility should also be registrable.");
    auto info = BaseType::registration_info();
    info.insert(std::hash<std::string>{}(typeid(SelfType).name()));
    return info;
}

template<typename SelfType>
std::unordered_set<std::size_t> I_RegistrableFacility<SelfType>::registration_info() const {
    return {std::hash<std::string>{}(typeid(SelfType).name())};
}

template<typename SelfType>
std::size_t I_RegistrableFacility<SelfType>::class_registration_info() {
    return std::hash<std::string>{}(typeid(SelfType).name());
}

} // namespace Metavision

#endif // METAVISION_HAL_I_REGISTRABLE_FACILITY_IMPL_H
