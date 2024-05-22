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
#include <fstream>

#include "devices/genx320/genx320_cx3_tz_device.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"
#include "devices/treuzell/tz_device_builder.h"
#include "devices/common/issd.h"
#include "devices/genx320/genx320es_cx3_issd.h"
#include "devices/genx320/genx320mp_cx3_issd.h"
#include "devices/genx320/register_maps/genx320es_registermap.h"
#include "metavision/psee_hw_layer/facilities/psee_hw_register.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "metavision/hal/utils/device_builder.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_tz_trigger_event.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_roi_driver.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_roi_interface.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_roi_pixel_mask_interface.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_roi_pixel_reset.h"
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

TzIssdGenX320Device::TzIssdGenX320Device(const Issd &issd, const std::pair<std::string, uint32_t> &env_var) :
    TzIssdDevice(issd), firmware_(TzIssdGenX320Device::read_firmware(env_var.first)), start_address_(env_var.second) {}
TzIssdGenX320Device::~TzIssdGenX320Device() {}

TzIssdGenX320Device::Firmware TzIssdGenX320Device::read_firmware(const std::string &filename) {
    Firmware firmware;

    if (filename.empty())
        return Firmware();

    std::ifstream fid(filename);
    if (!fid.is_open()) {
        MV_HAL_LOG_ERROR() << "Failed to load firmware from:" << filename;
        return Firmware();
    } else {
        MV_HAL_LOG_TRACE() << "Loading Risc-V firmware from:" << filename;
    }
    std::istream &input(fid);

    uint32_t v0, v1, v2, v3;
    uint32_t p0, offset = 0, v;

    input >> std::hex;
    if (input) {
        input >> std::ws;
    }
    while (input) {
        if (input.peek() == '@') {
            input.ignore();
            input >> p0;
            offset = 0;
        } else {
            input >> v0 >> v1 >> v2 >> v3;
            v = v0 + (v1 << 8) + (v2 << 16) + (v3 << 24);
            firmware.emplace_back(p0 + offset, v);
            offset += 4;
        }
        input >> std::ws;
    }
    MV_HAL_LOG_TRACE() << "Risc-V Firmware size:" << firmware.size() << " words";
    return firmware;
}

bool TzIssdGenX320Device::download_firmware() const {
    if (!firmware_.empty()) {
        MV_HAL_LOG_TRACE() << "Start Risc-V Firmware programing";

        // Reset RISC-V
        // TODO : MANAGE RESET OF RISC-V TO ALLOW MULTIPLE FLASH
        const uint32_t bank_address = (*register_map)["mem_bank/bank_mem0"].get_address();

        uint32_t prev_bank_id = 0;
        uint32_t prev_mem_id  = GENX_MEM_BANK_NONE;

        (*register_map)["mem_bank/bank_select"]["bank"].write_value(prev_bank_id);
        (*register_map)["mem_bank/bank_select"]["select"].write_value(prev_mem_id);

        for (auto &operation : firmware_) {
            uint32_t address = operation.first;
            uint32_t value   = operation.second;

            uint32_t mem_id;
            if ((DMEM_ADDR <= address) && (address < DMEM_ADDR + DMEM_SIZE)) {
                mem_id = GENX_MEM_BANK_DMEM;
            } else if ((IMEM_ADDR <= address) && (address < IMEM_ADDR + IMEM_SIZE)) {
                mem_id = GENX_MEM_BANK_IMEM;
            } else {
                MV_HAL_LOG_ERROR() << "No memory at 0x" << std::hex << address << std::dec;
                continue;
            }
            uint32_t bank_id     = (address - ((mem_id == GENX_MEM_BANK_DMEM) ? DMEM_ADDR : IMEM_ADDR)) / MEM_BANK_SIZE;
            uint32_t bank_offset = address % MEM_BANK_SIZE;

            if ((bank_id != prev_bank_id) || (mem_id != prev_mem_id)) {
                MV_HAL_LOG_TRACE() << "\tPrograming Mem " << ((mem_id == GENX_MEM_BANK_DMEM) ? "DMEM" : "IMEM")
                                   << " bank:" << bank_id;
                (*register_map)["mem_bank/bank_select"]["bank"].write_value(bank_id);
                (*register_map)["mem_bank/bank_select"]["select"].write_value(mem_id);
                prev_bank_id = bank_id;
                prev_mem_id  = mem_id;
            }
            // MV_HAL_LOG_TRACE() << "\tPrograming @0x " << std::hex << address << "  = 0x" << value;

            (*register_map)[bank_address + bank_offset] = value;
        }
        return true;
    }
    return false;
}
void TzIssdGenX320Device::start_firmware(bool is_mp) const {
    if (is_mp) {
        unsigned int retries = 0;

        (*register_map)["mbx/cmd_ptr"]["cmd_ptr"].write_value(0x70200200);

        while (retries < 10) {
            if (((*register_map)["mbx/cmd_ptr"]["cmd_ptr"].read_value() & 0xff000000) == 0) {
                MV_HAL_LOG_TRACE() << "Jump to IMEM successfull";
                break;
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            retries++;
        }

        if (retries == 10) {
            MV_HAL_LOG_ERROR() << "Failed to jump to IMEM";
        }

    } else {
        if (((DMEM_ADDR <= start_address_) && (start_address_ < DMEM_ADDR + DMEM_SIZE)) ||
            ((IMEM_ADDR <= start_address_) && (start_address_ < IMEM_ADDR + IMEM_SIZE))) {
            MV_HAL_LOG_TRACE() << "Start Risc-V execution at 0x" << std::hex << start_address_;
            // Currently the CPU will always start at 0x200200 (default address) assuming that ROMMODE IO is in low
            // state
            (*register_map)["mbx/cpu_start_en"]["cpu_start_en"].write_value(1);

        } else {
            MV_HAL_LOG_ERROR() << "Start address 0x" << std::hex << start_address_ << std::dec << " is not valid.";
        }
    }
}

void TzIssdGenX320Device::initialize() {
    MV_HAL_LOG_TRACE() << "Device initialization";
    TzIssdDevice::initialize();

    if (download_firmware() == true)
        start_firmware(false);
}
std::pair<std::string, uint32_t> TzIssdGenX320Device::parse_env(const char *input) {
    uint32_t outputValue = 0x200200;
    if (input == nullptr) {
        return std::make_pair("", outputValue); // Default values
    }

    std::istringstream stream(input);
    std::string outputString;

    std::getline(stream, outputString, ':');

    if (stream.fail()) {
        outputString = input;
    } else {
        if (stream.str().find("0x") != std::string::npos) {
            stream >> std::hex >> outputValue;
        } else {
            stream >> outputValue;
        }
    }
    return std::make_pair(outputString, outputValue);
}

TzCx3GenX320::TzCx3GenX320(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id, const Issd &issd,
                           bool mp_variant, std::shared_ptr<TzDevice> parent) :
    TzDevice(cmd, dev_id, parent),
    TzIssdGenX320Device(issd, TzIssdGenX320Device::parse_env(getenv("MV_FLAGS_RISCV_FW_PATH"))),
    TzDeviceWithRegmap(GenX320ESRegisterMap, GenX320ESRegisterMapSize, ROOT_PREFIX),
    is_mp(mp_variant) {
    // Beware the firmware has not been loaded during the initialize operation performed
    // in the parent class TzIssdDevice
    if (download_firmware() == true)
        start_firmware(is_mp);
    sync_mode_ = I_CameraSynchronization::SyncMode::STANDALONE;
    iph_mirror_control(true);
    temperature_init();
}

std::shared_ptr<TzDevice> TzCx3GenX320::build(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id,
                                              std::shared_ptr<TzDevice> parent) {
    if (can_build_es(cmd, dev_id)) {
        return std::make_shared<TzCx3GenX320>(cmd, dev_id, issd_genx320es_cx3_sequence, false, parent);
    } else if (can_build_mp(cmd, dev_id)) {
        return std::make_shared<TzCx3GenX320>(cmd, dev_id, issd_genx320mp_cx3_sequence, true, parent);
    } else {
        return nullptr;
    }
}

static TzRegisterBuildMethod method0("psee,cx3_saphir", TzCx3GenX320::build, TzCx3GenX320::can_build);
static TzRegisterBuildMethod method1("psee,cx3_genx320", TzCx3GenX320::build, TzCx3GenX320::can_build);

bool TzCx3GenX320::can_build(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id) {
    return (can_build_es(cmd, dev_id) || can_build_mp(cmd, dev_id));
}

bool TzCx3GenX320::can_build_es(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id) {
    uint32_t id = cmd->read_device_register(dev_id, 0x14)[0];

    if (id == 0x30501C01) {
        return true;
    } else {
        return false;
    }
}

bool TzCx3GenX320::can_build_mp(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id) {
    uint32_t id = cmd->read_device_register(dev_id, 0x14)[0];

    if (id == 0xB0602003) {
        return true;
    } else {
        return false;
    }
}

void TzCx3GenX320::spawn_facilities(DeviceBuilder &device_builder, const DeviceConfig &device_config) {
    device_builder.add_facility(std::make_unique<GenX320TzTriggerEvent>(register_map, SENSOR_PREFIX));

    auto roi_driver = std::make_shared<GenX320RoiDriver>(320, 320, register_map, SENSOR_PREFIX, device_config);

    device_builder.add_facility(std::make_unique<GenX320RoiInterface>(roi_driver));
    device_builder.add_facility(std::make_unique<GenX320RoiPixelMaskInterface>(roi_driver));
    device_builder.add_facility(std::make_unique<GenX320RoiPixelReset>(roi_driver));

    device_builder.add_facility(std::make_unique<GenX320LLBiases>(register_map, device_config));
    device_builder.add_facility(std::make_unique<AntiFlickerFilter>(register_map, get_sensor_info(), SENSOR_PREFIX));
    device_builder.add_facility(std::make_unique<EventTrailFilter>(register_map, get_sensor_info(), SENSOR_PREFIX));
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

    // /!\ HAL advertizes first value in list as default format
    formats.emplace_back("EVT21;height=320;width=320");
    formats.emplace_back("EVT2;height=320;width=320");

    if (is_mp) {
        formats.emplace_back("EVT3;height=320;width=320");
    }

    return formats;
}

StreamFormat TzCx3GenX320::set_output_format(const std::string &format_name) {
    if (is_mp && (format_name == "EVT3")) {
        (*register_map)["edf/control"]["format"].write_value(1);
        (*register_map)["edf/pipeline_control"].write_value(1);
    } else if (format_name == "EVT2") {
        (*register_map)["edf/control"]["format"].write_value(0);
        (*register_map)["edf/pipeline_control"].write_value(1);
    } else {
        // Default as EVT21
        (*register_map)["edf/control"]["format"].write_value(2);
        (*register_map)["edf/control"]["endianness"].write_value(0);
        (*register_map)["edf/pipeline_control"].write_value(1);
    }
    return get_output_format();
}

StreamFormat TzCx3GenX320::get_output_format() const {
    uint32_t fmt        = (*register_map)["edf/control"]["format"].read_value();
    std::string fmt_str = "";

    switch (fmt) {
    case 0:
        fmt_str = "EVT2";
        break;
    case 1:
        fmt_str = "EVT3";
        break;
    case 2:
        fmt_str = "EVT21;endianness=little";
        break;
    default:
        break;
    }

    StreamFormat format(fmt_str);
    format["width"]  = "320";
    format["height"] = "320";

    return format;
}

long TzCx3GenX320::get_system_id() const {
    if (is_mp) {
        return SystemId::SYSTEM_EVK3_GENX320_MP;
    } else {
        return SystemId::SYSTEM_EVK3_GENX320;
    }
}

I_HW_Identification::SensorInfo TzCx3GenX320::get_sensor_info() {
    if (is_mp) {
        return {320, 1, "GenX320MP"};
    } else {
        return {320, 0, "GenX320"};
    }
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
