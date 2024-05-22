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

#ifndef METAVISION_HAL_GENX320_CX3_TZ_DEVICE_H
#define METAVISION_HAL_GENX320_CX3_TZ_DEVICE_H

#include "devices/treuzell/tz_issd_device.h"
#include "metavision/psee_hw_layer/facilities/tz_monitoring.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_main_device.h"

namespace Metavision {

class TzIssdGenX320Device : public TzIssdDevice {
public:
    TzIssdGenX320Device(const Issd &issd, const std::pair<std::string, uint32_t> &env_var);
    virtual ~TzIssdGenX320Device();
    bool download_firmware() const;
    void start_firmware(bool is_mp) const;
    static std::pair<std::string, uint32_t> parse_env(const char *input);

protected:
    virtual void initialize() override;

private:
    using Firmware = std::vector<std::pair<uint32_t, uint32_t>>;
    Firmware firmware_;
    uint32_t start_address_;
    static Firmware read_firmware(const std::string &filename);

    static constexpr uint32_t DMEM_ADDR     = (0x00300000UL);
    static constexpr uint32_t DMEM_SIZE     = (32 * 1024UL);
    static constexpr uint32_t IMEM_ADDR     = (0x00200000UL);
    static constexpr uint32_t IMEM_SIZE     = (32 * 1024UL);
    static constexpr uint32_t MEM_BANK_SIZE = (64UL * sizeof(uint32_t));

    static constexpr uint32_t GENX_MEM_BANK_NONE = (0x0UL);
    static constexpr uint32_t GENX_MEM_BANK_IMEM = (0x2UL);
    static constexpr uint32_t GENX_MEM_BANK_DMEM = (0x3UL);
};

class TzCx3GenX320 : public TzIssdGenX320Device,
                     public TzMainDevice,
                     public TemperatureProvider,
                     public IlluminationProvider,
                     public PixelDeadTimeProvider {
public:
    TzCx3GenX320(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id, const Issd &issd, bool mp_variant,
                 std::shared_ptr<TzDevice> parent);
    virtual ~TzCx3GenX320();
    static std::shared_ptr<TzDevice> build(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id,
                                           std::shared_ptr<TzDevice> parent);

    static bool can_build(std::shared_ptr<TzLibUSBBoardCommand>, uint32_t dev_id);
    static bool can_build_es(std::shared_ptr<TzLibUSBBoardCommand>, uint32_t dev_id);
    static bool can_build_mp(std::shared_ptr<TzLibUSBBoardCommand>, uint32_t dev_id);
    std::list<StreamFormat> get_supported_formats() const override;
    StreamFormat get_output_format() const override;
    StreamFormat set_output_format(const std::string &format_name) override;
    virtual long get_system_id() const;
    virtual bool set_mode_standalone();
    virtual bool set_mode_master();
    virtual bool set_mode_slave();
    virtual I_CameraSynchronization::SyncMode get_mode() const;
    virtual I_HW_Identification::SensorInfo get_sensor_info();
    long long get_sensor_id();
    virtual int get_temperature();
    virtual int get_illumination();
    virtual int get_pixel_dead_time();

protected:
    virtual void spawn_facilities(DeviceBuilder &device_builder, const DeviceConfig &device_config);

private:
    I_CameraSynchronization::SyncMode sync_mode_;
    void time_base_config(bool external, bool master);
    void iph_mirror_control(bool enable);
    void lifo_control(bool enable, bool cnt_enable);
    std::vector<uint32_t> lifo_acquisition(int expected_wait_time);
    void temperature_init();
    bool is_mp;
};

} // namespace Metavision

#endif // METAVISION_HAL_GENX320_CX3_TZ_DEVICE_H
