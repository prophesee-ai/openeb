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

#include <math.h>

#include "devices/gen31/gen31_ccam5_tz_device.h"
#include "devices/utils/device_system_id.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"
#include "devices/common/issd.h"
#include "devices/gen31/gen31_evk3_issd.h"
#include "devices/treuzell/tz_device_builder.h"
#include "metavision/psee_hw_layer/devices/gen31/gen31_event_rate_noise_filter_module.h"
#include "devices/gen31/gen31_ccam5_trigger_event.h"
#include "devices/gen31/gen31_ccam5_trigger_out.h"
#include "metavision/psee_hw_layer/devices/gen31/gen31_ll_biases.h"
#include "metavision/psee_hw_layer/devices/gen31/gen31_roi_command.h"
#include "metavision/psee_hw_layer/facilities/psee_hw_register.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "geometries/vga_geometry.h"
#include "metavision/hal/utils/device_builder.h"
#include "plugin/psee_plugin.h"
#include "metavision/hal/utils/hal_error_code.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_utils.h"
#include "metavision/psee_hw_layer/utils/regmap_data.h"

#include "devices/gen31/register_maps/ccam5_single_gen31_system_control_registermap.h"
#include "devices/utils/register_maps/common/ccam2_system_monitor_trigger_ext_adc.h"
#include "devices/utils/register_maps/common/system_config_registermap.h"
#include "devices/gen31/register_maps/ccam3_single_gen31_sensorif_registermap.h"
#include "devices/utils/register_maps/common/mipitx_registermap.h"
#include "devices/utils/register_maps/common/spi_flash_master_registermap.h"

namespace Metavision {

namespace {
std::string CCAM5_PREFIX     = "";
std::string SENSOR_IF_PREFIX = "SENSOR_IF/GEN31_IF/";
std::string SENSOR_PREFIX    = "SENSOR_IF/GEN31/";
} // namespace

TzCcam5Gen31::TzCcam5Gen31(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id,
                           std::shared_ptr<TzDevice> parent) :
    TzDevice(cmd, dev_id, parent),
    TzPseeFpgaDevice(),
    TzDeviceWithRegmap(
        {
            std::make_tuple(ccam5_single_gen31_SystemControlRegisterMap,
                            ccam5_single_gen31_SystemControlRegisterMapSize, "SYSTEM_CONTROL/", 0),
            std::make_tuple(CCAM2SystemMonitorTriggerExtADC, CCAM2SystemMonitorTriggerExtADCSize, "SYSTEM_MONITOR/",
                            0x40),
            std::make_tuple(ccam3_single_gen31_Gen31SensorIFRegisterMap,
                            ccam3_single_gen31_Gen31SensorIFRegisterMapSize, "SENSOR_IF/", 0x200),
            std::make_tuple(SystemConfigRegisterMap, SystemConfigRegisterMapSize, "SYSTEM_CONFIG/", 0x800),
            std::make_tuple(MIPITXRegisterMap, MIPITXRegisterMapSize, "MIPI_TX/", 0x1500),
            std::make_tuple(SPIFlashMasterRegisterMap, SPIFlashMasterRegisterMapSize, "FLASH/", 0x1600),
        },
        CCAM5_PREFIX),
    TzIssdDevice(gen31_evk3_sequence) {
    (*register_map)["SENSOR_IF/GEN31/lifo_ctrl"]["lifo_en"] = 0x1;
    sync_mode_                                              = I_CameraSynchronization::SyncMode::STANDALONE;
}

std::shared_ptr<TzDevice> TzCcam5Gen31::build(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id,
                                              std::shared_ptr<TzDevice> parent) {
    if (cmd->read_device_register(dev_id, 0x800)[0] != SYSTEM_EVK3_GEN31_EVT3)
        throw HalException(HalErrorCode::FailedInitialization, "Wrong FPGA system ID");
    return std::make_shared<TzCcam5Gen31>(cmd, dev_id, parent);
}
static TzRegisterBuildMethod method("psee,ccam5_fpga", TzCcam5Gen31::build);

void TzCcam5Gen31::spawn_facilities(DeviceBuilder &device_builder, const DeviceConfig &device_config) {
    device_builder.add_facility(std::make_unique<Gen31Ccam5TriggerEvent>(register_map, shared_from_this()));
    device_builder.add_facility(std::make_unique<Gen31Ccam5TriggerOut>(
        register_map, std::dynamic_pointer_cast<TzCcam5Gen31>(shared_from_this())));

    auto hw_register = std::make_shared<PseeHWRegister>(register_map);
    device_builder.add_facility(std::make_unique<Gen31_LL_Biases>(device_config, hw_register, SENSOR_PREFIX));
    device_builder.add_facility(std::make_unique<Gen31_EventRateNoiseFilterModule>(hw_register, SENSOR_PREFIX));
    auto geometry = VGAGeometry();
    device_builder.add_facility(
        std::make_unique<Gen31ROICommand>(geometry.get_width(), geometry.get_height(), register_map, SENSOR_PREFIX));
    // those facilities are not exposed in the public API yet
    // device_builder.add_facility(std::make_unique<Gen31PatternGenerator>(register_map));
}

TzCcam5Gen31::~TzCcam5Gen31() {}

long TzCcam5Gen31::get_system_id() const {
    return TzPseeFpgaDevice::get_system_id();
}

long long TzCcam5Gen31::get_sensor_id() {
    return (*register_map)["SENSOR_IF/GEN31/chip_id"].read_value();
}

std::list<StreamFormat> TzCcam5Gen31::get_supported_formats() const {
    std::list<StreamFormat> formats;
    formats.push_back(StreamFormat("EVT3;height=480;width=640"));
    return formats;
}

StreamFormat TzCcam5Gen31::get_output_format() const {
    return StreamFormat("EVT3;height=480;width=640");
}

bool TzCcam5Gen31::set_mode_standalone() {
    (*register_map)["SYSTEM_CONTROL/ATIS_CONTROL"]["MASTER_MODE"]   = 0x1;
    (*register_map)["SYSTEM_CONTROL/ATIS_CONTROL"]["USE_EXT_START"] = 0x0;

    sync_mode_ = I_CameraSynchronization::SyncMode::STANDALONE;
    return true;
}

bool TzCcam5Gen31::set_mode_master() {
    (*register_map)["SYSTEM_CONTROL/ATIS_CONTROL"]["MASTER_MODE"]   = 0x1;
    (*register_map)["SYSTEM_CONTROL/ATIS_CONTROL"]["USE_EXT_START"] = 0x1;

    sync_mode_ = I_CameraSynchronization::SyncMode::MASTER;
    return true;
}

bool TzCcam5Gen31::set_mode_slave() {
    (*register_map)["SYSTEM_CONTROL/ATIS_CONTROL"]["MASTER_MODE"]   = 0x0;
    (*register_map)["SYSTEM_CONTROL/ATIS_CONTROL"]["USE_EXT_START"] = 0x1;

    sync_mode_ = I_CameraSynchronization::SyncMode::SLAVE;
    return true;
}

I_CameraSynchronization::SyncMode TzCcam5Gen31::get_mode() {
    return sync_mode_;
}

int TzCcam5Gen31::get_illumination() {
    (*register_map)["SENSOR_IF/GEN31/lifo_ctrl"].write_value(0);
    (*register_map)["SENSOR_IF/GEN31/lifo_ctrl"]["lifo_en"]     = 1;
    (*register_map)["SENSOR_IF/GEN31/lifo_ctrl"]["lifo_cnt_en"] = 1;
    bool valid                                                  = false;
    uint16_t retries                                            = 0;
    uint32_t counter                                            = 0;
    while (valid == false && retries < 10) {
        auto reg_val = (*register_map)["SENSOR_IF/GEN31/lifo_ctrl"].read_value();
        reg_val      = (*register_map)["SENSOR_IF/GEN31/lifo_ctrl"].read_value();
        valid        = reg_val & 1 << 29;
        counter      = reg_val & ((1 << 27) - 1);
        retries += 1;
    }

    if (!valid) {
        return -1;
    }

    if (counter != decltype(counter)(-1)) {
        float t = float(counter) / 100.;
        return powf(10, 3.5 - logf(t * 0.37) / logf(10));
    }
    return -1;
}

} // namespace Metavision
