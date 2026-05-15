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

#ifndef METAVISION_HAL_PSEE_PLUGINS_DEVICES_IMX8_TZ_DEVICE_H
#define METAVISION_HAL_PSEE_PLUGINS_DEVICES_IMX8_TZ_DEVICE_H

#include <memory>

#include "devices/treuzell/tz_unknown.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_main_device.h"

namespace Metavision {

class TzImx8Device : public TzMainDevice, public TzUnknownDevice {
public:
    TzImx8Device(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id, std::shared_ptr<TzDevice> parent);

    // TzMainDevice
    bool set_mode_standalone() override;
    bool set_mode_master() override;
    bool set_mode_slave() override;
    I_CameraSynchronization::SyncMode get_mode() const override;
    I_HW_Identification::SensorInfo get_sensor_info() override;

    static std::shared_ptr<TzImx8Device> build(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t id,
                                               std::shared_ptr<TzDevice> parent);
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_PLUGINS_DEVICES_IMX8_TZ_DEVICE_H
