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

#ifndef METAVISION_HAL_I_REGISTRABLE_FACILITY_H
#define METAVISION_HAL_I_REGISTRABLE_FACILITY_H

#include <memory>

#include "metavision/hal/facilities/i_facility.h"

namespace Metavision {

/// @brief Class serving as base for facility types that can be registered in a @ref Device
/// @tparam SelfType the actual type of the registrable facility
/// @tparam BaseType optional I_RegistrableFacility derived type that SelfType extends
template<typename SelfType, typename BaseType = void>
struct I_RegistrableFacility : public BaseType,
                               /** @cond */ public I_RegistrableFacility<SelfType> /** @endcond */ {
    virtual std::unordered_set<std::size_t> registration_info() const override;
};

template<typename SelfType>
struct I_RegistrableFacility<SelfType, void>
    : virtual public I_Facility, public std::enable_shared_from_this<I_RegistrableFacility<SelfType, void>> {
    virtual std::unordered_set<std::size_t> registration_info() const override;

    /// @brief Returns information used to lookup the facility in a @ref Device
    /// @returns The facility class' hash used for lookup
    static std::size_t class_registration_info();
};

} // namespace Metavision

#include "detail/i_registrable_facility_impl.h"

#endif // METAVISION_HAL_I_REGISTRABLE_FACILITY_H
