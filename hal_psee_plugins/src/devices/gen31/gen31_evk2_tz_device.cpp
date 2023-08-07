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

#include "devices/gen31/gen31_evk2_tz_device.h"
#include "devices/utils/device_system_id.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include "devices/common/issd.h"
#include "devices/gen31/gen31_evk2_issd.h"
#include "devices/gen31/register_maps/gen31_evk2_registermap.h"
#include "devices/treuzell/tz_device_builder.h"
#include "metavision/psee_hw_layer/devices/gen31/gen31_event_rate_noise_filter_module.h"
#include "metavision/psee_hw_layer/devices/gen31/gen31_ll_biases.h"
//#include "devices/gen31/gen31_pattern_generator.h"
#include "metavision/psee_hw_layer/devices/gen31/gen31_roi_command.h"
#include "metavision/psee_hw_layer/utils/regmap_data.h"
#include "metavision/psee_hw_layer/devices/common/evk2_tz_trigger_event.h"
#include "metavision/psee_hw_layer/devices/common/evk2_tz_trigger_out.h"
#include "metavision/psee_hw_layer/facilities/psee_hw_register.h"
#include "geometries/vga_geometry.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/utils/device_builder.h"
#include "plugin/psee_plugin.h"
#include "utils/psee_hal_utils.h"

namespace Metavision {
namespace {
std::string ROOT_PREFIX   = "PSEE/";
std::string CCAM5_PREFIX  = "CCAM5_IF/CCAM5/";
std::string SENSOR_PREFIX = CCAM5_PREFIX + "GEN31/";
} // namespace

TzEvk2Gen31::TzEvk2Gen31(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id, std::shared_ptr<TzDevice> parent) :
    TzDevice(cmd, dev_id, parent),
    TzPseeVideo(cmd, dev_id, parent),
    TzDeviceWithRegmap(Gen31Evk2RegisterMap, Gen31Evk2RegisterMapSize, ROOT_PREFIX),
    TzIssdDevice(gen31_evk2_sequence),
    sys_ctrl_(register_map) {
    (*register_map)[CCAM5_PREFIX + "SYSTEM_MONITOR/TEMP_VCC_MONITOR/EXT_TEMP_CONTROL"]["EXT_TEMP_MONITOR_SPI_EN"]
        .write_value(1);
    (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_en"] = 0x1;
    sync_mode_                                              = I_CameraSynchronization::SyncMode::STANDALONE;
}

std::shared_ptr<TzDevice> TzEvk2Gen31::build(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id,
                                             std::shared_ptr<TzDevice> parent) {
    if (can_build(cmd, dev_id))
        return std::make_shared<TzEvk2Gen31>(cmd, dev_id, parent);
    else
        return nullptr;
}

bool TzEvk2Gen31::can_build(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id) {
    return (cmd->read_device_register(dev_id, 0x800)[0] == SYSTEM_EVK2_GEN31);
}
static TzRegisterBuildMethod method("psee,video_gen3.1", TzEvk2Gen31::build, TzEvk2Gen31::can_build);

void TzEvk2Gen31::spawn_facilities(DeviceBuilder &device_builder, const DeviceConfig &device_config) {
    auto hw_register = std::make_shared<PseeHWRegister>(register_map);
    device_builder.add_facility(std::make_unique<Gen31_LL_Biases>(device_config, hw_register, SENSOR_PREFIX));
    device_builder.add_facility(std::make_unique<Gen31_EventRateNoiseFilterModule>(hw_register, SENSOR_PREFIX));

    auto geometry = VGAGeometry();
    device_builder.add_facility(
        std::make_unique<Gen31ROICommand>(geometry.get_width(), geometry.get_height(), register_map, SENSOR_PREFIX));
    // those facilities are not exposed in the public API yet
    // device_builder.add_facility(std::make_unique<Gen31PatternGenerator>(register_map));

    device_builder.add_facility(std::make_unique<Evk2TzTriggerEvent>(register_map, "", shared_from_this()));
    device_builder.add_facility(std::make_unique<Evk2TzTriggerOut>(
        register_map, "", std::dynamic_pointer_cast<TzPseeVideo>(shared_from_this())));
}

TzEvk2Gen31::~TzEvk2Gen31() {}

void TzEvk2Gen31::start() {
    TzIssdDevice::start();
}

void TzEvk2Gen31::stop() {
    TzIssdDevice::stop();
}

long long TzEvk2Gen31::get_sensor_id() {
    return (*register_map)[SENSOR_PREFIX + "chip_id"].read_value();
}

std::list<StreamFormat> TzEvk2Gen31::get_supported_formats() const {
    std::list<StreamFormat> formats;
    formats.push_back(StreamFormat("EVT2;height=480;width=640"));
    return formats;
}

StreamFormat TzEvk2Gen31::get_output_format() const {
    return StreamFormat("EVT2;height=480;width=640");
}

long TzEvk2Gen31::get_system_id() {
    return cmd->read_device_register(tzID, 0x800)[0];
}

bool TzEvk2Gen31::set_mode_standalone() {
    /* time_base_config(ext_sync, master, master_sel, fwd_up, fwd_down)
        - ext_sync   = 1 --|
        - master     = 1 ----> generate internal sync_out
        - master_sel = 0 -> Don't care since master = 1. Time-base counter is auto-generated.
        - fwd_up     = 0 -> IOs sync_out  = 0
        - fwd_down   = 1 -> ccam5 sync_in = internal sync_out
    */
    sys_ctrl_.time_base_config(true, true, false, false, true);

    if (!sys_ctrl_.is_trigger_out_enabled()) {
        // Disabled sync out IO
        sys_ctrl_.sync_out_pin_control(false);
        sys_ctrl_.sync_out_pin_config(false);
    }
    sync_mode_ = I_CameraSynchronization::SyncMode::STANDALONE;
    return true;
}

bool TzEvk2Gen31::set_mode_master() {
    /* time_base_config(ext_sync, master, master_sel, fwd_up, fwd_down)
        - ext_sync   = 1 --|
        - master     = 1 ----> generate internal sync_out
        - master_sel = 0 -> Don't care since master = 1. Time-base counter is auto-generated.
        - fwd_up     = 1 -> IOs sync_out  = internal sync_out
        - fwd_down   = 1 -> ccam5 sync_in = internal sync_out
    */
    if (sys_ctrl_.is_trigger_out_enabled()) {
        MV_HAL_LOG_WARNING() << "Switching to master sync mode. Trigger out will be overridden.";
    }

    sys_ctrl_.time_base_config(true, true, false, true, true);
    sys_ctrl_.sync_out_pin_config(false);
    sys_ctrl_.sync_out_pin_control(true);
    sync_mode_ = I_CameraSynchronization::SyncMode::MASTER;
    return true;
}

bool TzEvk2Gen31::set_mode_slave() {
    /* time_base_config(ext_sync, master, master_sel, fwd_up, fwd_down)
        - ext_sync   = 1 --|
        - master     = 0 ----> Use sync_in from mux master_sel
        - master_sel = 1 -> internal sync_in = IOs sync_in
        - fwd_up     = 0 -> IOs sync_out     = 0
        - fwd_down   = 1 -> ccam5 sync_in    = IOs sync_in
    */
    sys_ctrl_.time_base_config(true, false, true, false, true);

    if (!sys_ctrl_.is_trigger_out_enabled()) {
        // Disabled sync out IO
        sys_ctrl_.sync_out_pin_control(false);
        sys_ctrl_.sync_out_pin_config(false);
    }
    sync_mode_ = I_CameraSynchronization::SyncMode::SLAVE;
    return true;
}

I_CameraSynchronization::SyncMode TzEvk2Gen31::get_mode() {
    return sync_mode_;
}

int TzEvk2Gen31::get_temperature() {
    auto r = (*register_map)[CCAM5_PREFIX + "SYSTEM_MONITOR/TEMP_VCC_MONITOR/EVK_EXT_TEMP_VALUE"].read_value();
    if (r != decltype(r)(-1))
        return r / 4096;
    return -1;
}

int TzEvk2Gen31::get_illumination() {
    (*register_map)[SENSOR_PREFIX + "lifo_ctrl"].write_value(0);
    (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_en"]     = 1;
    (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_cnt_en"] = 1;

    bool valid       = false;
    uint16_t retries = 0;
    uint32_t counter = 0;
    while (valid == false && retries < 10) {
        auto reg_val = (*register_map)[SENSOR_PREFIX + "lifo_ctrl"].read_value();
        reg_val      = (*register_map)[SENSOR_PREFIX + "lifo_ctrl"].read_value();
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
