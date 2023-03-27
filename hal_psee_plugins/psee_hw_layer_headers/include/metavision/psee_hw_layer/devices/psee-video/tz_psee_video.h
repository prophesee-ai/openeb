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

#ifndef METAVISION_HAL_TZ_PSEE_VIDEO_H
#define METAVISION_HAL_TZ_PSEE_VIDEO_H

#include "metavision/psee_hw_layer/devices/treuzell/tz_psee_fpga_device.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_main_device.h"

namespace Metavision {

class TzPseeVideo : public TzPseeFpgaDevice, public TzMainDevice {
public:
    TzPseeVideo(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id, std::shared_ptr<TzDevice> parent);
    virtual ~TzPseeVideo();
    virtual void spawn_facilities(DeviceBuilder &device_builder, const DeviceConfig &device_config);
    static std::shared_ptr<TzDevice> build(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id,
                                           std::shared_ptr<TzDevice> parent);

    virtual std::list<StreamFormat> get_supported_formats() const override;
    StreamFormat get_output_format() const override;
    virtual long get_system_id() const;
    virtual bool set_mode_standalone();
    virtual bool set_mode_master();
    virtual bool set_mode_slave();
    virtual I_CameraSynchronization::SyncMode get_mode();
    virtual I_HW_Identification::SensorInfo get_sensor_info() {
        return {0, 0, "Gen0.0"};
    }
};

} // namespace Metavision

#endif // METAVISION_HAL_TZ_PSEE_VIDEO_H
