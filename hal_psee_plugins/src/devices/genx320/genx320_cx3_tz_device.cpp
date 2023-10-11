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
#include <math.h>

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
#include "metavision/psee_hw_layer/devices/genx320/genx320_roi_driver.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_roi_interface.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_roi_pixel_mask_interface.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_ll_biases.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_erc.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_nfl_driver.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_nfl_interface.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_dem_interface.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_digital_crop.h"
#include "metavision/psee_hw_layer/devices/common/antiflicker_filter.h"
#include "metavision/psee_hw_layer/devices/common/event_trail_filter.h"
#include "devices/utils/device_system_id.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {
namespace {
std::string ROOT_PREFIX   = "PSEE/GENX320/";
std::string SENSOR_PREFIX = "";
using vfield              = std::map<std::string, uint32_t>;
} // namespace

uint32_t get_bitfield(uint32_t value, uint8_t idx, uint8_t size) {
    return ((1 << size) - 1) & (value >> idx);
}

TzCx3GenX320::TzCx3GenX320(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id,
                           std::shared_ptr<TzDevice> parent) :
    TzDevice(cmd, dev_id, parent),
    TzIssdDevice(issd_genx320es_cx3_sequence),
    TzDeviceWithRegmap(GenX320ESRegisterMap, GenX320ESRegisterMapSize, ROOT_PREFIX) {
    sync_mode_ = I_CameraSynchronization::SyncMode::STANDALONE;
    iph_mirror_control(true);
    temperature_init();
}

std::shared_ptr<TzDevice> TzCx3GenX320::build(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id,
                                              std::shared_ptr<TzDevice> parent) {
    if (can_build(cmd, dev_id)) {
        return std::make_shared<TzCx3GenX320>(cmd, dev_id, parent);
    } else {
        return nullptr;
    }
}

static TzRegisterBuildMethod method0("psee,cx3_saphir", TzCx3GenX320::build, TzCx3GenX320::can_build);

bool TzCx3GenX320::can_build(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id) {
    return (cmd->read_device_register(dev_id, 0x14)[0] == 0x30501C01);
}

void TzCx3GenX320::spawn_facilities(DeviceBuilder &device_builder, const DeviceConfig &device_config) {
    device_builder.add_facility(
        std::make_unique<GenX320TzTriggerEvent>(register_map, SENSOR_PREFIX, shared_from_this()));

    auto roi_driver = std::make_shared<GenX320RoiDriver>(320, 320, register_map, SENSOR_PREFIX, device_config);

    device_builder.add_facility(std::make_unique<GenX320RoiInterface>(roi_driver));
    device_builder.add_facility(std::make_unique<GenX320RoiPixelMaskInterface>(roi_driver));

    device_builder.add_facility(std::make_unique<GenX320LLBiases>(register_map, device_config));
    device_builder.add_facility(std::make_unique<AntiFlickerFilter>(
        std::dynamic_pointer_cast<TzDeviceWithRegmap>(shared_from_this()), get_sensor_info(), SENSOR_PREFIX));
    device_builder.add_facility(std::make_unique<EventTrailFilter>(
        std::dynamic_pointer_cast<TzDeviceWithRegmap>(shared_from_this()), get_sensor_info(), SENSOR_PREFIX));
    device_builder.add_facility(std::make_unique<GenX320Erc>(register_map));

    auto nfl = std::make_shared<GenX320NflDriver>(register_map);
    device_builder.add_facility(std::make_unique<GenX320NflInterface>(nfl));

    device_builder.add_facility(std::make_unique<GenX320DemInterface>(register_map, SENSOR_PREFIX));
    device_builder.add_facility(std::make_unique<GenX320DigitalCrop>(register_map, SENSOR_PREFIX));
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

I_CameraSynchronization::SyncMode TzCx3GenX320::get_mode() const {
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

void TzCx3GenX320::temperature_init() {
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
}

int TzCx3GenX320::get_temperature() {
    MV_HAL_LOG_DEBUG() << "Temperature measurement";

    std::list<uint32_t> temp_meas = {};
    int meas_samples              = 3;

    // ADC Clock enable
    (*register_map)["adc_control"]["adc_clk_en"].write_value(1);

    for (int i = 0; i < meas_samples; i++) {
        (*register_map)["adc_control"]["adc_start"].write_value(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(3));

        auto val = (*register_map)["adc_status1"]["adc_dac_dyn"].read_value();
        temp_meas.push_back((val * 0.216) - 54);
    }

    int temp = accumulate(temp_meas.begin(), temp_meas.end(), 0) / meas_samples;

    // ADC Clock disable
    (*register_map)["adc_control"]["adc_clk_en"].write_value(0);

    return temp;
}

int TzCx3GenX320::get_illumination() {
    MV_HAL_LOG_DEBUG() << "Illumination measurement";
    bool valid        = false;
    uint16_t measures = 3;
    uint32_t ack_time = 20;
    uint32_t ack_step = 10;

    std::vector<uint32_t> results(3, 0);

    // We follow 20ms->200ms->2s.
    for (int i = 1; i <= measures; i++) {
        results = lifo_acquisition(ack_time);
        if (results[0] != 1) {
            // We failed to converge.
            ack_time = ack_time * ack_step;
        } else {
            valid = true;
            (*register_map)["lifo_ton_status"]["lifo_ton_valid"].write_value(1);
            break;
        }
        RegisterMap::Field *my_field((*register_map)["lifo_ton_status"]["lifo_ton_valid"].get_field());
    }

    if (valid) {
        int illu = (int)round(exp((11.97 - 0.98 * log(results[2]))));

        return illu;
    } else {
        MV_HAL_LOG_ERROR() << "Failed to get illumination";
        return -1;
    }
}

void TzCx3GenX320::iph_mirror_control(bool enable) {
    (*register_map)["iph_mirr_ctrl"].write_value(vfield({{"iph_mirr_en", enable},
                                                         {"iph_mirr_tbus_in_en", 0},
                                                         {"iph_mirr_calib_en", 0},
                                                         {"iph_mirr_calib_x10_en", 0},
                                                         {"iph_mirr_dft_en", 0},
                                                         {"iph_mirr_dft_sel", 0}}));

    if (enable) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void TzCx3GenX320::lifo_control(bool enable, bool cnt_enable) {
    (*register_map)["lifo_ctrl"].write_value(
        vfield({{"lifo_en", enable}, {"lifo_cont_op_en", 1}, {"lifo_dft_mode_en", 0}, {"lifo_timer_en", cnt_enable}}));

    if (enable) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

std::vector<uint32_t> TzCx3GenX320::lifo_acquisition(int expected_wait_time = 20) {
    // The iph mirror needs to be enabled first
    // Default acquisition time is 500 ms to cover low level light condition
    // Assuming 25 MHz operation

    lifo_control(true, false);

    // Wait for specified duration for LIFO ton accumulation
    std::this_thread::sleep_for(std::chrono::milliseconds(expected_wait_time));

    // Read LIFO ton register
    uint32_t ton_stat = (*register_map)["lifo_ton_status"].read_value();

    uint8_t valid_index = 0xFF;
    RegisterMap::Field *my_field((*register_map)["lifo_ton_status"]["lifo_ton_valid"].get_field());

    valid_index        = my_field->get_start();
    auto overrun_index = (*register_map)["lifo_ton_status"]["lifo_ton_overrun"].get_field()->get_start();
    auto ton_cnt_index = (*register_map)["lifo_ton_status"]["lifo_ton"].get_field()->get_start();
    auto ton_cnt_size  = (*register_map)["lifo_ton_status"]["lifo_ton"].get_field()->get_len();

    auto valid   = get_bitfield(ton_stat, valid_index, 1);
    auto overrun = get_bitfield(ton_stat, overrun_index, 1);
    auto ton_cnt = get_bitfield(ton_stat, ton_cnt_index, ton_cnt_size);

    MV_HAL_LOG_DEBUG() << "Ton status =" << std::hex << "0x" << ton_stat << std::endl;
    MV_HAL_LOG_DEBUG() << "Valid bit =" << std::dec << valid << std::endl;
    MV_HAL_LOG_DEBUG() << "Overrun bit =" << std::dec << overrun << std::endl;
    MV_HAL_LOG_DEBUG() << "Ton cnt bit =" << std::dec << ton_cnt << std::endl;

    lifo_control(false, false);

    std::vector<uint32_t> results = {valid, overrun, ton_cnt};

    return results;
}

int TzCx3GenX320::get_pixel_dead_time() {
    MV_HAL_LOG_DEBUG() << "Pixel dead time measurement";
    auto reg          = (*register_map)[SENSOR_PREFIX + "refractory_ctrl"];
    uint32_t refr_val = 0;
    uint32_t valid    = 0;
    uint32_t overrun  = 0;
    uint32_t count    = 0;

    reg.write_value(vfield({
        {"refr_en", 1},
        {"refr_cnt_en", 1},
    }));

    // Erase refractory status bit
    reg["refr_overrun"].write_value(1);

    auto valid_index   = (*register_map)["refractory_ctrl"]["refr_valid"].get_field()->get_start();
    auto overrun_index = (*register_map)["refractory_ctrl"]["refr_overrun"].get_field()->get_start();
    auto cnt_index     = (*register_map)["refractory_ctrl"]["refr_counter"].get_field()->get_start();
    auto cnt_size      = (*register_map)["refractory_ctrl"]["refr_counter"].get_field()->get_len();

    int max_retries = 10;
    while (valid == 0) {
        if (max_retries == 0) {
            throw HalException(HalErrorCode::MaximumRetriesExeeded);
        } else {
            // Read refractory counter
            refr_val = (*register_map)["refractory_ctrl"].read_value();
            valid    = get_bitfield(refr_val, valid_index, 1);
            overrun  = get_bitfield(refr_val, overrun_index, 1);
            count    = get_bitfield(refr_val, cnt_index, cnt_size);
        }
        max_retries--;
    }

    MV_HAL_LOG_DEBUG() << "Refr status =" << std::hex << "0x" << refr_val << std::endl;
    MV_HAL_LOG_DEBUG() << "Valid bit =" << std::dec << valid << std::endl;
    MV_HAL_LOG_DEBUG() << "Overrun bit =" << std::dec << overrun << std::endl;
    MV_HAL_LOG_DEBUG() << "Count bit =" << std::dec << count << std::endl;

    return count / (25 * 2);
}

} // namespace Metavision
