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

#include "metavision/hal/facilities/i_hal_software_info.h"

namespace Metavision {

I_HALSoftwareInfo::I_HALSoftwareInfo(const Metavision::SoftwareInfo &software_info) : pimpl_(software_info) {}

const Metavision::SoftwareInfo &I_HALSoftwareInfo::get_software_info() {
    return pimpl_;
}

} // namespace Metavision
