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

#include "metavision/sdk/driver/event_trail_filter_module.h"

namespace Metavision {

EventTrailFilterModule::EventTrailFilterModule(I_EventTrailFilterModule *noise_filter) : pimpl_(noise_filter) {}

EventTrailFilterModule::~EventTrailFilterModule() {}

std::set<I_EventTrailFilterModule::Type> EventTrailFilterModule::get_available_types() const {
    return pimpl_->get_available_types();
}

bool EventTrailFilterModule::enable(bool state) {
    return pimpl_->enable(state);
}

I_EventTrailFilterModule *EventTrailFilterModule::get_facility() const {
    return pimpl_;
}

} // namespace Metavision
