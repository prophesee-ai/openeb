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

#ifndef METAVISION_HAL_I_HAL_SOFTWARE_INFO_H
#define METAVISION_HAL_I_HAL_SOFTWARE_INFO_H

#include <string>

#include "metavision/sdk/base/utils/software_info.h"
#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Class that provides information about the HAL software
class I_HALSoftwareInfo : public I_RegistrableFacility<I_HALSoftwareInfo> {
public:
    /// @brief Constructor
    /// @param software_info Information about the HAL software version
    I_HALSoftwareInfo(const Metavision::SoftwareInfo &software_info);

    /// @brief Gets plugin's software information
    /// @return The plugin's software information
    const Metavision::SoftwareInfo &get_software_info();

private:
    Metavision::SoftwareInfo pimpl_;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_HAL_SOFTWARE_INFO_H
