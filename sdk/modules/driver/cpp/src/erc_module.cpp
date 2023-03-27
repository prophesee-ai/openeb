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

#include <memory>

#include "metavision/sdk/driver/erc_module.h"

namespace Metavision {

ErcModule::ErcModule(I_ErcModule *erc) : pimpl_(erc) {}

ErcModule::~ErcModule() {}

bool ErcModule::enable(bool b) {
    return pimpl_->enable(b);
}

bool ErcModule::is_enabled() {
    return (pimpl_->is_enabled());
}

void ErcModule::set_cd_event_rate(uint32_t events_per_sec) {
    pimpl_->set_cd_event_rate(events_per_sec);
}

uint32_t ErcModule::get_cd_event_rate() {
    return (pimpl_->get_cd_event_rate());
}

I_ErcModule *ErcModule::get_facility() const {
    return pimpl_;
}

} // namespace Metavision
