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

#ifndef METAVISION_HAL_CCAM_GEN31_EVK3_DEVICE_CONTROL_H
#define METAVISION_HAL_CCAM_GEN31_EVK3_DEVICE_CONTROL_H

#include "facilities/psee_device_control.h"
#include "devices/common/issd.h"

namespace Metavision {

class RegisterMap;

class Gen31Evk3DeviceControl : public PseeDeviceControl {
public:
    Gen31Evk3DeviceControl(const std::shared_ptr<RegisterMap> &register_map);

    long long get_sensor_id() override;
    virtual void reset() override;

    virtual bool set_mode_standalone_impl() override;
    virtual bool set_mode_slave_impl() override;
    virtual bool set_mode_master_impl() override;
    std::string get_sensor_prefix() const;

protected:
    virtual void initialize() override;
    virtual void destroy() override;

private:
    virtual bool set_evt_format_impl(EvtFormat fmt) override;
    virtual void start_impl() override;
    virtual void stop_impl() override;

    void ApplyRegisterOperationSequence(const std::vector<RegisterOperation> sequence);
    void ApplyRegisterOperation(const RegisterOperation operation);

    std::shared_ptr<RegisterMap> register_map_;
};

} // namespace Metavision

#endif // METAVISION_HAL_CCAM_GEN31_EVK3_DEVICE_CONTROL_H
