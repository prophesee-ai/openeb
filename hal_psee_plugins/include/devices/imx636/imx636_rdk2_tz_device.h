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

#ifndef METAVISION_HAL_IMX636_RDK2_TZ_DEVICE_H
#define METAVISION_HAL_IMX636_RDK2_TZ_DEVICE_H

#include "metavision/psee_hw_layer/devices/psee-video/tz_psee_video.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_regmap_device.h"
#include "metavision/psee_hw_layer/facilities/tz_monitoring.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_main_device.h"
#include "devices/common/evk2_system_control.h"

namespace Metavision {

class TzRdk2Imx636 : public TzPseeVideo,
                     public TzDeviceWithRegmap,
                     public TemperatureProvider,
                     public IlluminationProvider {
public:
    TzRdk2Imx636(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id, std::shared_ptr<TzDevice> parent);
    virtual ~TzRdk2Imx636();
    static std::shared_ptr<TzDevice> build(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id,
                                           std::shared_ptr<TzDevice> parent);
    static bool can_build(std::shared_ptr<BoardCommand>, uint32_t dev_id);

    virtual std::list<StreamFormat> get_supported_formats() const override;
    StreamFormat get_output_format() const override;
    virtual long get_system_id();
    virtual bool set_mode_standalone();
    virtual bool set_mode_master();
    virtual bool set_mode_slave();
    virtual I_CameraSynchronization::SyncMode get_mode();
    virtual I_HW_Identification::SensorInfo get_sensor_info();
    long long get_sensor_id();
    virtual int get_illumination();
    virtual int get_temperature();

protected:
    virtual void spawn_facilities(DeviceBuilder &device_builder, const DeviceConfig &device_config);

private:
    void temperature_init();
    void lifo_control(bool enable, bool out_en, bool cnt_en);
    void iph_mirror_control(bool enable);

    Evk2SystemControl sys_ctrl_;
    I_CameraSynchronization::SyncMode sync_mode_;
};

} // namespace Metavision

#endif // METAVISION_HAL_IMX636_RDK2_TZ_DEVICE_H
