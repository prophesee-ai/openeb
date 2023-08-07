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

#include <thread>
#include <chrono>
#include <numeric>

#include "devices/genx320/genx320_cx3_tz_device.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"
#include "devices/treuzell/tz_device_builder.h"
#include "devices/common/issd.h"
#include "devices/genx320/genx320es_cx3_issd.h"
#include "devices/genx320/register_maps/genx320es_registermap.h"
#include "metavision/psee_hw_layer/facilities/psee_hw_register.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "metavision/hal/utils/device_builder.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_tz_trigger_event.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_ll_roi.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_ll_biases.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_erc.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_nfl.h"
#include "metavision/psee_hw_layer/devices/common/antiflicker_filter.h"
#include "metavision/psee_hw_layer/devices/common/event_trail_filter.h"
#include "devices/utils/device_system_id.h"

namespace Metavision {
namespace {
std::string ROOT_PREFIX   = "PSEE/GENX320/";
std::string SENSOR_PREFIX = "";
using vfield              = std::map<std::string, uint32_t>;
} // namespace

TzCx3GenX320::TzCx3GenX320(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id,
                           std::shared_ptr<TzDevice> parent) :
    TzDevice(cmd, dev_id, parent),
    TzIssdDevice(issd_genx320es_cx3_sequence),
    TzDeviceWithRegmap(GenX320ESRegisterMap, GenX320ESRegisterMapSize, ROOT_PREFIX) {
    sync_mode_ = I_CameraSynchronization::SyncMode::STANDALONE;
}

std::shared_ptr<TzDevice> TzCx3GenX320::build(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id,
                                              std::shared_ptr<TzDevice> parent) {
    if (can_build(cmd, dev_id)) {
        return std::make_shared<TzCx3GenX320>(cmd, dev_id, parent);
    } else {
        return nullptr;
    }
}

static TzRegisterBuildMethod method0("psee,cx3_saphir", TzCx3GenX320::build, TzCx3GenX320::can_build);

bool TzCx3GenX320::can_build(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id) {
    return (cmd->read_device_register(dev_id, 0x14)[0] == 0x30501C01);
}

void TzCx3GenX320::spawn_facilities(DeviceBuilder &device_builder, const DeviceConfig &device_config) {
    device_builder.add_facility(
        std::make_unique<GenX320TzTriggerEvent>(register_map, SENSOR_PREFIX));
    device_builder.add_facility(std::make_unique<GenX320LowLevelRoi>(device_config, register_map, SENSOR_PREFIX));
    device_builder.add_facility(std::make_unique<GenX320LLBiases>(register_map, device_config));
    // FIXME: make_shared called on a reference
    device_builder.add_facility(std::make_unique<AntiFlickerFilter>(
        std::make_shared<RegisterMap>(regmap()), get_sensor_info(), SENSOR_PREFIX));
    device_builder.add_facility(std::make_unique<EventTrailFilter>(
        std::make_shared<RegisterMap>(regmap()), get_sensor_info(), SENSOR_PREFIX));
    device_builder.add_facility(std::make_unique<GenX320Erc>(register_map));
    device_builder.add_facility(std::make_unique<GenX320NoiseFilter>(register_map));
}

TzCx3GenX320::~TzCx3GenX320() {}

long long TzCx3GenX320::get_sensor_id() {
    return (*register_map)["chip_id"].read_value();
}

std::list<StreamFormat> TzCx3GenX320::get_supported_formats() const {
    std::list<StreamFormat> formats;
    formats.push_back(StreamFormat("EVT21;height=320;width=320"));
    return formats;
}

StreamFormat TzCx3GenX320::get_output_format() const {
    StreamFormat format("EVT21");
    format["width"]  = "320";
    format["height"] = "320";
    return format;
}

long TzCx3GenX320::get_system_id() const {
    return SystemId::SYSTEM_EVK3_GENX320;
}

bool TzCx3GenX320::set_mode_standalone() {
    time_base_config(false, true);
    sync_mode_ = I_CameraSynchronization::SyncMode::STANDALONE;
    return true;
}

bool TzCx3GenX320::set_mode_master() {
    time_base_config(true, true);

    sync_mode_ = I_CameraSynchronization::SyncMode::MASTER;
    return true;
}

bool TzCx3GenX320::set_mode_slave() {
    time_base_config(true, false);

    sync_mode_ = I_CameraSynchronization::SyncMode::SLAVE;
    return true;
}

I_CameraSynchronization::SyncMode TzCx3GenX320::get_mode() {
    return sync_mode_;
}

/**
 * @brief Configure sensor time base settings. By default, the sensor is in monocular mode
 *
 * @param external if true external time base, otherwise, use internal
 * @param master if true, use master mode, else slave mode
 */
void TzCx3GenX320::time_base_config(bool external, bool master) {
    (*register_map)["ro/time_base_ctrl"].write_value(vfield{
        {"time_base_mode", external},       // 0 : Internal, 1 : External
        {"external_mode", master},          // 0 : Slave, 1 : Master (valid when in external mode)
        {"external_mode_enable", external}, // 0 : External mode disabled, 1 : External mode enabled
        {"us_counter_max", 25}              // Core clock is 25 MHz
    });

    if (external) {
        if (master) {
            // set SYNCHRO IO to output mode
            (*register_map)["io_ctrl2"]["sync_enzi"].write_value(0);
            (*register_map)["io_ctrl2"]["sync_en"].write_value(0);
        } else {
            // set SYNCHRO IO to input mode
            (*register_map)["io_ctrl2"]["sync_enzi"].write_value(1);
            (*register_map)["io_ctrl2"]["sync_en"].write_value(1);
        }
    }
}

int TzCx3GenX320::get_temperature() {
    // ADC enable
    (*register_map)["adc_control"].write_value(vfield({{"adc_en", 1}, {"adc_clk_en", 1}}));
    std::this_thread::sleep_for(std::chrono::microseconds(500));

    // ADC Buf cal
    (*register_map)["adc_misc_ctrl"].write_value(
        vfield({{"adc_buf_cal_en", 1}, {"adc_cmp_cal_en", 1}, {"adc_buf_adj_rng", 0}, {"adc_cmp_adj_rng", 0}}));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // ADC Misc control
    vfield fields = {{"adc_rng", 0}, {"adc_temp", 1}, {"adc_ext_bg", 0}};
    (*register_map)["adc_misc_ctrl"].write_value(fields);

    // Temperature enable
    (*register_map)["temp_ctrl"].write_value(vfield{{"temp_buf_en", 1}, {"temp_ihalf", 0}});
    (*register_map)["temp_ctrl"].write_value(vfield{{"temp_buf_offset_man", 32}, {"temp_buf_adj_rng", 0}});
    std::this_thread::sleep_for(std::chrono::microseconds(500));

    // Temperature buf cal
    (*register_map)["temp_ctrl"].write_value(vfield{{"temp_buf_cal_en", 1}, {"temp_buf_adj_rng", 0}});
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    std::list<uint32_t> temp_meas = {};

    for (int i = 0; i < 10; i++) {
        (*register_map)["adc_control"]["adc_start"].write_value(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(3));

        auto val = (*register_map)["adc_status1"]["adc_dac_dyn"].read_value();
        temp_meas.push_back((val * 0.216) - 54);
    }

    int temp = accumulate(temp_meas.begin(), temp_meas.end(), 0) / 10;

    // ADC Clock disable
    (*register_map)["adc_control"]["adc_clk_en"].write_value(0);

    return temp;
}

} // namespace Metavision
