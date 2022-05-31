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

#ifndef METAVISION_HAL_CCAM_GEN31_FX3_DEVICE_CONTROL_H
#define METAVISION_HAL_CCAM_GEN31_FX3_DEVICE_CONTROL_H

#include "devices/gen31/gen31_device_control.h"

namespace Metavision {

class Gen31Fx3DeviceControl : public Gen31DeviceControl {
public:
    Gen31Fx3DeviceControl(const std::shared_ptr<RegisterMap> &register_map);

    std::string get_sensor_prefix() const;

protected:
    virtual void initialize() override;
    virtual void destroy() override;

    virtual void enable_interface(bool state) override;
    virtual void reset_ts_internal() override;

private:
    virtual void start_impl() override;
    virtual void stop_impl() override;
};

} // namespace Metavision

#endif // METAVISION_HAL_CCAM_GEN31_FX3_DEVICE_CONTROL_H
