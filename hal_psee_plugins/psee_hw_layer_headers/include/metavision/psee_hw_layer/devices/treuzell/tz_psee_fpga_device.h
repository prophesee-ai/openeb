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

#ifndef METAVISION_HAL_TZ_PSEE_FPGA_DEVICE_H
#define METAVISION_HAL_TZ_PSEE_FPGA_DEVICE_H

#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"

namespace Metavision {

class TzPseeFpgaDevice : public virtual TzDevice {
public:
    TzPseeFpgaDevice();
    virtual void get_device_info(I_HW_Identification::SystemInfo &info, std::string prefix);
    uint32_t get_system_id() const;
    uint32_t get_system_version() const;
    uint32_t get_system_build_date() const;
    uint32_t get_system_version_control_id() const;
};

} // namespace Metavision

#endif // METAVISION_HAL_TZ_PSEE_FPGA_DEVICE_H
