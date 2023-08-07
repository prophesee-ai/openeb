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
#include <thread>
#include <chrono>

#include "devices/gen41/gen41_evk2_tz_device.h"
#include "devices/utils/device_system_id.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include "devices/common/issd.h"
#include "devices/gen41/gen41_evk2_issd.h"
#include "devices/treuzell/tz_device_builder.h"
#include "metavision/psee_hw_layer/devices/common/antiflicker_filter.h"
#include "metavision/psee_hw_layer/devices/gen41/gen41_digital_event_mask.h"
#include "metavision/psee_hw_layer/devices/gen41/gen41_erc.h"
#include "metavision/psee_hw_layer/devices/gen41/gen41_ll_biases.h"
#include "metavision/psee_hw_layer/devices/common/event_trail_filter.h"
#include "metavision/psee_hw_layer/devices/gen41/gen41_roi_command.h"
#include "devices/gen41/register_maps/gen41_evk2_registermap.h"
#include "metavision/psee_hw_layer/devices/common/evk2_tz_trigger_event.h"
#include "metavision/psee_hw_layer/devices/common/evk2_tz_trigger_out.h"
#include "metavision/psee_hw_layer/facilities/psee_hw_register.h"
#include "geometries/hd_geometry.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/utils/device_builder.h"
#include "plugin/psee_plugin.h"
#include "utils/psee_hal_utils.h"

namespace Metavision {
namespace {
std::string ROOT_PREFIX   = "PSEE/";
std::string SENSOR_PREFIX = "SENSOR_IF/GEN41/";
} // namespace

TzEvk2Gen41::TzEvk2Gen41(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id, std::shared_ptr<TzDevice> parent) :
    TzDevice(cmd, dev_id, parent),
    TzPseeVideo(cmd, dev_id, parent),
    TzDeviceWithRegmap(Gen41Evk2RegisterMap, Gen41Evk2RegisterMapSize, ROOT_PREFIX),
    TzIssdDevice(gen41_evk2_sequence),
    sys_ctrl_(register_map) {
    sync_mode_ = I_CameraSynchronization::SyncMode::STANDALONE;
    temperature_init();
    iph_mirror_control(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    lifo_control(true, true, true);
}

std::shared_ptr<TzDevice> TzEvk2Gen41::build(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id,
                                             std::shared_ptr<TzDevice> parent) {
    if (can_build(cmd, dev_id))
        return std::make_shared<TzEvk2Gen41>(cmd, dev_id, parent);
    else
        return nullptr;
}

bool TzEvk2Gen41::can_build(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id) {
    return (cmd->read_device_register(dev_id, 0x800)[0] == SYSTEM_EVK2_GEN41);
}
static TzRegisterBuildMethod method("psee,video_gen4.1", TzEvk2Gen41::build, TzEvk2Gen41::can_build);

void TzEvk2Gen41::spawn_facilities(DeviceBuilder &device_builder, const DeviceConfig &device_config) {
    device_builder.add_facility(std::make_unique<EventTrailFilter>(
        std::dynamic_pointer_cast<TzDeviceWithRegmap>(shared_from_this()), get_sensor_info(), SENSOR_PREFIX));
    device_builder.add_facility(std::make_unique<AntiFlickerFilter>(
        std::dynamic_pointer_cast<TzDeviceWithRegmap>(shared_from_this()), get_sensor_info(), SENSOR_PREFIX));

    auto erc = device_builder.add_facility(std::make_unique<Gen41Erc>(register_map, SENSOR_PREFIX + "erc/"));
    erc->initialize();
    erc->enable(true);

    auto hw_register = std::make_shared<PseeHWRegister>(register_map);
    device_builder.add_facility(std::make_unique<Gen41_LL_Biases>(device_config, hw_register, SENSOR_PREFIX));

    auto geometry = HDGeometry();
    device_builder.add_facility(
        std::make_unique<Gen41ROICommand>(geometry.get_width(), geometry.get_height(), register_map, SENSOR_PREFIX));

    device_builder.add_facility(std::make_unique<Evk2TzTriggerEvent>(register_map, "", shared_from_this()));
    device_builder.add_facility(std::make_unique<Evk2TzTriggerOut>(
        register_map, "", std::dynamic_pointer_cast<TzPseeVideo>(shared_from_this())));

    device_builder.add_facility(
        std::make_unique<Gen41DigitalEventMask>(register_map, SENSOR_PREFIX + "ro/digital_mask_pixel_"));
}

TzEvk2Gen41::~TzEvk2Gen41() {}

void TzEvk2Gen41::start() {
    TzIssdDevice::start();
}

void TzEvk2Gen41::stop() {
    TzIssdDevice::stop();
}

long long TzEvk2Gen41::get_sensor_id() {
    return (*register_map)[SENSOR_PREFIX + "chip_id"].read_value();
}

std::list<StreamFormat> TzEvk2Gen41::get_supported_formats() const {
    std::list<StreamFormat> formats;
    formats.push_back(StreamFormat("EVT3;height=720;width=1280"));
    return formats;
}

StreamFormat TzEvk2Gen41::get_output_format() const {
    return StreamFormat("EVT3;height=720;width=1280");
}

long TzEvk2Gen41::get_system_id() {
    return cmd->read_device_register(tzID, 0x800)[0];
}

bool TzEvk2Gen41::set_mode_standalone() {
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

bool TzEvk2Gen41::set_mode_master() {
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

bool TzEvk2Gen41::set_mode_slave() {
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

I_CameraSynchronization::SyncMode TzEvk2Gen41::get_mode() {
    return sync_mode_;
}

int TzEvk2Gen41::get_temperature() {
    (*register_map)[SENSOR_PREFIX + "adc_control"]["adc_clk_en"].write_value(1);
    (*register_map)[SENSOR_PREFIX + "adc_misc_ctrl"]["adc_temp"].write_value(1);
    (*register_map)[SENSOR_PREFIX + "adc_control"]["adc_start"].write_value(1);

    bool valid        = false;
    uint16_t retries  = 0;
    uint32_t counter  = 0;
    uint32_t status   = 0;
    uint32_t temp_val = 0;

    while (valid == false && retries < 5) {
        status   = (*register_map)[SENSOR_PREFIX + "adc_status"]["adc_done_dyn"].read_value();
        temp_val = (*register_map)[SENSOR_PREFIX + "adc_status"]["adc_dac_dyn"].read_value();
        valid    = status & 1;
        retries += 1;
    }

    if (!valid) {
        MV_HAL_LOG_ERROR() << "Failed to get temperature";
        return -1;
    }

    (*register_map)[SENSOR_PREFIX + "adc_control"]["adc_clk_en"].write_value(0);

    return ((0.190 * temp_val) - 56);
}

int TzEvk2Gen41::get_illumination() {
    bool valid       = false;
    uint16_t retries = 0;
    uint32_t counter = 0;

    while (valid == false && retries < 10) {
        uint32_t reg_val = (*register_map)[SENSOR_PREFIX + "lifo_status"].read_value();
        valid            = reg_val & 1 << 29;
        counter          = reg_val & ((1 << 27) - 1);
        retries += 1;
    }

    if (!valid) {
        MV_HAL_LOG_ERROR() << "Failed to get illumination";
        return -1;
    }

    if (counter != decltype(counter)(-1)) {
        float t = float(counter) / 100.;
        return powf(10, 3.5 - logf(t * 0.37) / logf(10));
    }
    return -1;
}

void TzEvk2Gen41::temperature_init() {
    // Temperature ADC init
    (*register_map)[SENSOR_PREFIX + "adc_control"]["adc_en"].write_value(1);
    (*register_map)[SENSOR_PREFIX + "adc_control"]["adc_clk_en"].write_value(1);
    (*register_map)[SENSOR_PREFIX + "adc_misc_ctrl"]["adc_buf_cal_en"].write_value(1);
    std::this_thread::sleep_for(std::chrono::microseconds(100));

    // Temperature sensor init
    (*register_map)[SENSOR_PREFIX + "temp_ctrl"]["temp_buf_en"].write_value(1);
    (*register_map)[SENSOR_PREFIX + "temp_ctrl"]["temp_buf_cal_en"].write_value(1);
    std::this_thread::sleep_for(std::chrono::microseconds(100));

    (*register_map)[SENSOR_PREFIX + "adc_control"]["adc_clk_en"].write_value(0);
}

void TzEvk2Gen41::lifo_control(bool enable, bool out_en, bool cnt_en) {
    /* Control the LIFO settings.

    Args:
        enable (bool): Puts the LIFO in ready mode.
        out_en (bool): Turns on the LIFO.
        cnt_en (bool): Turns on the LIFO counter in the digital to start integrating.

    */

    if (enable && out_en) {
        (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_en"].write_value(enable);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_out_en"].write_value(out_en);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } else if (enable && !out_en) {
        (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_en"].write_value(enable);
    } else if (!enable && out_en) {
        (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_out_en"].write_value(out_en);
    } else if (!enable && !out_en) {
        (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_en"].write_value(enable);
        (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_out_en"].write_value(out_en);
    }
    (*register_map)[SENSOR_PREFIX + "lifo_ctrl"]["lifo_cnt_en"].write_value(cnt_en);
}

void TzEvk2Gen41::iph_mirror_control(bool enable) {
    (*register_map)[SENSOR_PREFIX + "iph_mirr_ctrl"]["iph_mirr_en"].write_value(enable);
    std::this_thread::sleep_for(std::chrono::microseconds(20));
    (*register_map)[SENSOR_PREFIX + "iph_mirr_ctrl"]["iph_mirr_amp_en"].write_value(enable);
    std::this_thread::sleep_for(std::chrono::microseconds(20));
}

} // namespace Metavision
