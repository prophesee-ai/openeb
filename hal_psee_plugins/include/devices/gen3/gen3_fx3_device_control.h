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

#ifndef METAVISION_HAL_GEN3_FX3_DEVICE_CONTROL_H
#define METAVISION_HAL_GEN3_FX3_DEVICE_CONTROL_H

#include "devices/gen3/gen3_device_control.h"
#include "devices/utils/evt_format.h"

namespace Metavision {

class Gen3Fx3DeviceControl : public Gen3DeviceControl {
public:
    Gen3Fx3DeviceControl(const std::shared_ptr<PseeLibUSBBoardCommand> &board_cmd);

protected:
    virtual void enable_interface(bool state) override;
    virtual void reset_ts_internal() override;

    virtual void initialize() override;
    virtual void destroy() override;

private:
    virtual bool set_evt_format_impl(EvtFormat fmt) override;
    virtual void start_impl() override;
    virtual void stop_impl() override;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN3_FX3_DEVICE_CONTROL_H
