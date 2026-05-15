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

#include "metavision/hal/facilities/detail/facility_wrapper.h"
#include "metavision/hal/facilities/i_facility.h"

namespace Metavision {

FacilityWrapper::FacilityWrapper(const std::shared_ptr<I_Facility> &facility) : facility_(facility) {
    if (facility_) {
        facility_->setup();
    }
}

FacilityWrapper::~FacilityWrapper() {
    if (facility_) {
        facility_->teardown();
    }
}

const std::shared_ptr<I_Facility> &FacilityWrapper::facility() const {
    return facility_;
}

} // namespace Metavision
