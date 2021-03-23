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

template<typename SelfType>
struct I_RegistrableFacility : public I_Facility, public std::enable_shared_from_this<I_RegistrableFacility<SelfType>> {
    const std::type_info &registration_info() const override final {
        return typeid(SelfType);
    }

    static const std::type_info &class_registration_info() {
        return typeid(SelfType);
    }
};

} // namespace Metavision

#endif // METAVISION_HAL_I_REGISTRABLE_FACILITY_H
