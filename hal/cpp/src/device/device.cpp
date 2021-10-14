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

#include "metavision/hal/device/device.h"
#include "metavision/hal/facilities/i_facility.h"
#include "metavision/hal/facilities/i_events_stream.h"

namespace Metavision {

Device::~Device() {
    // TODO TEAM-10620: remove this block of code
    // keep a pointer to the I_EventsStream facility, it must be destroyed after all other facilities
    // to make sure we keep the stream pointer alive for another potential Future::I_EventsStream facility using it
    std::shared_ptr<I_Facility> facility;
    for (auto &f : facilities_) {
        if (dynamic_cast<I_EventsStream *>(f.second->facility().get())) {
            facility = f.second->facility();
            break;
        }
    }
    facilities_.clear();
}

void Device::register_facility(std::unique_ptr<FacilityWrapper> p) {
    auto h         = std::hash<std::string>{}(p->facility()->registration_info().name());
    facilities_[h] = std::move(p);
}

} // namespace Metavision
