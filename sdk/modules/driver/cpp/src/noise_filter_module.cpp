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

#include "metavision/sdk/driver/noise_filter_module.h"

namespace Metavision {

NoiseFilterModule::NoiseFilterModule(I_NoiseFilterModule *noise_filter) : pimpl_(noise_filter) {}

NoiseFilterModule::~NoiseFilterModule() {}

void NoiseFilterModule::enable(I_NoiseFilterModule::Type type, uint32_t threshold) {
    pimpl_->enable(type, threshold);
}

void NoiseFilterModule::disable() {
    pimpl_->disable();
}

I_NoiseFilterModule *NoiseFilterModule::get_facility() const {
    return pimpl_;
}

} // namespace Metavision
