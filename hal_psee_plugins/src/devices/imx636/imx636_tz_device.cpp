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

#ifdef _MSC_VER
#define NOMINMAX
#endif

#include "devices/imx636/imx636_tz_device.h"
#include "devices/utils/device_system_id.h"
#include "boards/treuzell/tz_libusb_board_command.h"
#include "devices/treuzell/tz_device.h"
#include "devices/common/issd.h"
#include "devices/imx636/imx636_evk3_issd.h"
#include "devices/imx636/imx636_ll_biases.h"
#include "devices/gen41/gen41_antiflicker_module.h"
#include "devices/gen41/gen41_erc.h"
#include "devices/gen41/gen41_noise_filter_module.h"
#include "devices/gen41/gen41_roi_command.h"
#include "devices/imx636/imx636_evk3_regmap_builder.h"
#include "devices/imx636/imx636_tz_trigger_event.h"
#include "facilities/psee_hw_register.h"
#include "geometries/hd_geometry.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/utils/device_builder.h"
#include "plugin/psee_plugin.h"
#include <math.h>
#include <thread>
#include <chrono>

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {
namespace {
std::string ROOT_PREFIX   = "PSEE/";
std::string SENSOR_PREFIX = ROOT_PREFIX + "IMX636/";
} // namespace

TzImx636::TzImx636(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id, std::shared_ptr<TzDevice> parent) :
    TzDevice(cmd, dev_id, parent),
    TzIssdDevice(issd_evk3_imx636_sequence),
    TzDeviceWithRegmap(build_imx636_evk3_register_map) {
    sync_mode_ = I_DeviceControl::SyncMode::STANDALONE;
    temperature_init();
    iph_mirror_control(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    lifo_control(true, true, true);
}

std::shared_ptr<TzDevice> TzImx636::build(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id,
                                          std::shared_ptr<TzDevice> parent) {
    if (can_build(cmd, dev_id)) {
        return std::make_shared<TzImx636>(cmd, dev_id, parent);
    } else {
        return nullptr;
    }
}
static TzRegisterBuildMethod method0("psee,ccam5_gen42", TzImx636::build, TzImx636::can_build);
static TzRegisterBuildMethod method1("psee,ccam5_imx636", TzImx636::build, TzImx636::can_build);

bool TzImx636::can_build(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id) {
    return (cmd->read_device_register(dev_id, 0x14)[0] == 0xA0401806);
}

void TzImx636::spawn_facilities(DeviceBuilder &device_builder) {
    device_builder.add_facility(std::make_unique<Gen41NoiseFilterModule>(register_map, SENSOR_PREFIX));
    device_builder.add_facility(std::make_unique<Gen41AntiFlickerModule>(register_map, SENSOR_PREFIX));

    auto erc = device_builder.add_facility(
        std::make_unique<Gen41Erc>(register_map, SENSOR_PREFIX + "erc/", shared_from_this()));
    erc->initialize();
    erc->enable(true);

    auto geometry = HDGeometry();

    auto hw_register = device_builder.add_facility(std::make_unique<PseeHWRegister>(register_map));

    device_builder.add_facility(std::make_unique<Imx636_LL_Biases>(hw_register, SENSOR_PREFIX));

    device_builder.add_facility(
        std::make_unique<Gen41ROICommand>(geometry.get_width(), geometry.get_height(), register_map, SENSOR_PREFIX));

    device_builder.add_facility(
        std::make_unique<Imx636TzTriggerEvent>(register_map, SENSOR_PREFIX, shared_from_this()));
}

TzImx636::~TzImx636() {}

long long TzImx636::get_sensor_id() {
    return (*register_map)[SENSOR_PREFIX + "chip_id"].read_value();
}

TzDevice::StreamFormat TzImx636::get_output_format() {
    return {std::string("EVT3"), std::make_unique<HDGeometry>()};
}

long TzImx636::get_system_id() const {
    return SystemId::SYSTEM_EVK3_IMX636;
}

bool TzImx636::set_mode_standalone() {
    time_base_config(false, true);

    sync_mode_ = I_DeviceControl::SyncMode::STANDALONE;
    return true;
}

bool TzImx636::set_mode_master() {
    time_base_config(true, true);

    sync_mode_ = I_DeviceControl::SyncMode::MASTER;
    return true;
}

bool TzImx636::set_mode_slave() {
    time_base_config(true, false);

    sync_mode_ = I_DeviceControl::SyncMode::SLAVE;
    return true;
}

I_DeviceControl::SyncMode TzImx636::get_mode() {
    return sync_mode_;
}

int TzImx636::get_temperature() {
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

int TzImx636::get_illumination() {
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

void TzImx636::temperature_init() {
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

/**
 * @brief Configure sensor time base settings. By default, the sensor is in monocular mode
 *
 * @param external if true external time base, otherwise, use internal
 * @param master if true, use master mode, else slave mode
 */
void TzImx636::time_base_config(bool external, bool master) {
    (*register_map)[SENSOR_PREFIX + "ro/time_base_ctrl"].write_value(vfield{
        {"time_base_mode", external},       // 0 : Internal, 1 : External
        {"external_mode", master},          // 0 : Slave, 1 : Master (valid when in external mode)
        {"external_mode_enable", external}, // 0 : External mode disabled, 1 : External mode enabled
        {"Reserved_10_4", 100}              // default 100 = 1us
    });

    if (external) {
        if (master) {
            // set SYNCHRO IO to output mode
            (*register_map)[SENSOR_PREFIX + "dig_pad2_ctrl"]["pad_sync"].write_value(0b1100);
        } else {
            // set SYNCHRO IO to input mode
            (*register_map)[SENSOR_PREFIX + "dig_pad2_ctrl"]["pad_sync"].write_value(0b1111);
        }
    }
}

/**
 * @brief Control the LIFO settings
 *
 * @param enable puts the LIFO in ready mode
 * @param out_en turns on the LIFO
 * @param cnt_en turns on the LIFO counter in the digital to start integrating.
 */
void TzImx636::lifo_control(bool enable, bool out_en, bool cnt_en) {
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

void TzImx636::iph_mirror_control(bool enable) {
    (*register_map)[SENSOR_PREFIX + "iph_mirr_ctrl"]["iph_mirr_en"].write_value(enable);
    std::this_thread::sleep_for(std::chrono::microseconds(20));
    (*register_map)[SENSOR_PREFIX + "iph_mirr_ctrl"]["iph_mirr_amp_en"].write_value(enable);
    std::this_thread::sleep_for(std::chrono::microseconds(20));
}

} // namespace Metavision
